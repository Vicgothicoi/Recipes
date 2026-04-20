"""
BGE 路由分类器
基于 BAAI/bge-small-zh-v1.5，冻结 Encoder，只训练分类头。
输入：查询文本
输出：0（不需要图检索）/ 1（需要图检索）
"""

import ast
import logging
import os
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)


class BGEQueryRouter(nn.Module):

    def __init__(self, encoder_name: str = "BAAI/bge-small-zh-v1.5"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)

        # 冻结所有 encoder 参数
        for param in self.encoder.parameters():
            param.requires_grad = False

        hidden_size = self.encoder.config.hidden_size  # bge-small: 512
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        return self.classifier(cls_emb).squeeze(-1)  # (batch,)


class QueryDataset(Dataset):

    def __init__(self, data: List[Tuple[str, int]]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[str, int]:
        return self.data[idx]


def load_data(path: str) -> List[Tuple[str, int]]:
    samples = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = ast.literal_eval(line)
                query, graph_useful = str(row[0]), int(bool(row[1]))
                samples.append((query, graph_useful))
            except Exception as e:
                logger.warning(f"第 {lineno} 行解析失败，跳过: {e}")
    return samples


def collate_fn(tokenizer, max_length: int = 128):
    """返回一个 collate 函数，将 batch 编码为 tensor。"""

    def _collate(batch: List[Tuple[str, int]]):
        texts, labels = zip(*batch)
        encoding = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return (
            encoding["input_ids"],
            encoding["attention_mask"],
            torch.tensor(labels, dtype=torch.float),
        )

    return _collate


def train(
    data_path: str,
    encoder_name: str = "BAAI/bge-small-zh-v1.5",
    save_path: str = "rag_modules/bge_router.pt",
    epochs: int = 20,
    batch_size: int = 8,
    lr: float = 1e-3,
    val_ratio: float = 0.2,
    seed: int = 42,
):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    # 加载数据
    all_data = load_data(data_path)
    if not all_data:
        raise ValueError(f"数据文件为空或解析失败: {data_path}")

    # 划分训练/验证集
    split = int(len(all_data) * (1 - val_ratio))
    indices = torch.randperm(len(all_data)).tolist()
    train_data = [all_data[i] for i in indices[:split]]
    val_data = [all_data[i] for i in indices[split:]]
    logger.info(f"训练集: {len(train_data)}  验证集: {len(val_data)}")

    tokenizer = AutoTokenizer.from_pretrained(encoder_name)
    collate = collate_fn(tokenizer)

    train_loader = DataLoader(
        QueryDataset(train_data),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        QueryDataset(val_data), batch_size=batch_size, shuffle=False, collate_fn=collate
    )

    model = BGEQueryRouter(encoder_name).to(device)

    # 只优化分类头参数
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=lr)
    criterion = nn.BCELoss()

    best_val_acc = 0.0
    for epoch in range(1, epochs + 1):
        # ── 训练 ──
        model.train()
        total_loss = 0.0
        for input_ids, attention_mask, labels in train_loader:
            input_ids, attention_mask, labels = (
                input_ids.to(device),
                attention_mask.to(device),
                labels.to(device),
            )
            optimizer.zero_grad()
            preds = model(input_ids, attention_mask)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # ── 验证 ──
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for input_ids, attention_mask, labels in val_loader:
                input_ids, attention_mask, labels = (
                    input_ids.to(device),
                    attention_mask.to(device),
                    labels.to(device),
                )
                preds = model(input_ids, attention_mask)
                predicted = (preds >= 0.5).float()
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total if total > 0 else 0.0
        logger.info(
            f"Epoch {epoch:02d}/{epochs}  loss={total_loss/len(train_loader):.4f}  val_acc={val_acc:.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            logger.info(f"  → 保存最优模型 (val_acc={best_val_acc:.3f})")

    logger.info(
        f"训练完成，最优验证准确率: {best_val_acc:.3f}，模型保存至: {save_path}"
    )
    return best_val_acc


class BGERouterInference:
    """加载训练好的分类头，提供单条/批量推理接口。"""

    def __init__(
        self,
        model_path: str = "rag_modules/bge_router.pt",
        encoder_name: str = "BAAI/bge-small-zh-v1.5",
        threshold: float = 0.5,
    ):
        self.threshold = threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        self.model = BGEQueryRouter(encoder_name)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, query: str) -> Tuple[bool, float]:
        """返回 (graph_useful, confidence)。"""
        encoding = self.tokenizer(
            query, return_tensors="pt", truncation=True, max_length=128
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        with torch.no_grad():
            score = self.model(input_ids, attention_mask).item()
        return score >= self.threshold, round(score, 4)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    data_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "query", "route.txt"
    )
    train(data_path=data_path)

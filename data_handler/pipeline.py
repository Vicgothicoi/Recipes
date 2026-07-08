"""数据处理全流程编排。"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from data_handler.config import DataHandlerConfig, DEFAULT_CONFIG
from data_handler.export import GraphExporter
from data_handler.parsers import parse_directory, parse_file
from data_handler.schemas import EntitySpan, ParsedDocument, RelationTriple
from data_handler.utils.logging_setup import setup_logging

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    documents: List[ParsedDocument] = field(default_factory=list)
    entities_count: int = 0
    relations_count: int = 0
    skipped_paths: List[str] = field(default_factory=list)
    nodes_csv: Optional[str] = None
    relationships_csv: Optional[str] = None


class DataPipeline:
    """解析 → [图像描述] → NER → RE → 导出 CSV。"""

    def __init__(self, config: Optional[DataHandlerConfig] = None) -> None:
        self.config = config or DEFAULT_CONFIG

    def _save_intermediate(self, doc: ParsedDocument, stage: str) -> None:
        out_dir = self.config.intermediate_dir / "parsed"
        out_dir.mkdir(parents=True, exist_ok=True)
        name = Path(doc.source_path).stem + ".json"
        path = out_dir / f"{stage}_{name}"
        path.write_text(
            json.dumps(doc.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def process_document(
        self,
        path: Path | str,
        *,
        run_vision: bool = False,
        run_ner: bool = True,
        run_re: bool = True,
        use_weak_if_no_checkpoint: bool = True,
        ner_checkpoint: str = "best",
        re_checkpoint: str = "best",
        save_intermediate: bool = False,
    ) -> tuple[ParsedDocument, List[EntitySpan], List[RelationTriple]]:
        doc = parse_file(path)
        if save_intermediate:
            self._save_intermediate(doc, "parse")

        if run_vision:
            try:
                from data_handler.vision import ImageDescriptor

                descriptor = ImageDescriptor(config=self.config)
                doc = descriptor.describe_document(doc)
                if save_intermediate:
                    self._save_intermediate(doc, "vision")
            except Exception as e:
                logger.warning("图像描述跳过: %s", e)

        entities: List[EntitySpan] = []
        if run_ner:
            entities = self._run_ner(
                doc,
                checkpoint=ner_checkpoint,
                use_weak=use_weak_if_no_checkpoint,
            )
            if save_intermediate:
                self._save_intermediate(doc, "ner")

        relations: List[RelationTriple] = []
        if run_re:
            relations = self._run_re(
                doc,
                entities,
                checkpoint=re_checkpoint,
                use_weak=use_weak_if_no_checkpoint,
            )
            if save_intermediate:
                self._save_intermediate(doc, "re")

        return doc, entities, relations

    def _run_ner(
        self,
        doc: ParsedDocument,
        *,
        checkpoint: str,
        use_weak: bool,
    ) -> List[EntitySpan]:
        ckpt = self.config.ner_checkpoint_dir / checkpoint / "meta.json"
        if ckpt.is_file():
            try:
                from data_handler.ner import NerPredictor

                predictor = NerPredictor(checkpoint=checkpoint, config=self.config)
                return predictor.predict_document(doc)
            except Exception as e:
                logger.warning("NER 推理失败，尝试弱标注: %s", e)

        if use_weak:
            from data_handler.ner.weak_labeler import weak_label_document

            spans: List[EntitySpan] = []
            for sample in weak_label_document(doc):
                for ent in sample.entities:
                    spans.append(
                        EntitySpan(
                            start=ent["start"],
                            end=ent["end"],
                            label=ent["label"],
                            text=sample.text[ent["start"] : ent["end"]],
                            section=sample.metadata.get("section"),
                            source_path=doc.source_path,
                            block_index=sample.metadata.get("block_index"),
                        )
                    )
            doc.metadata["entities"] = [s.to_dict() for s in spans]
            logger.debug("使用 NER 弱标注: %s (%d 实体)", doc.source_path, len(spans))
            return spans

        return []

    def _run_re(
        self,
        doc: ParsedDocument,
        entities: List[EntitySpan],
        *,
        checkpoint: str,
        use_weak: bool,
    ) -> List[RelationTriple]:
        ckpt = self.config.re_checkpoint_dir / checkpoint / "meta.json"
        if ckpt.is_file() and entities:
            try:
                from data_handler.re import RePredictor

                predictor = RePredictor(
                    checkpoint=checkpoint,
                    config=self.config,
                    min_score=self.config.re_min_score,
                )
                return predictor.predict_document(doc, entities)
            except Exception as e:
                logger.warning("RE 推理失败，尝试弱标注: %s", e)

        if use_weak:
            from data_handler.re.weak_labeler import weak_label_from_entities

            samples = weak_label_from_entities(doc, entities)
            triples = []
            for s in samples:
                from data_handler.re.labels import relation_name

                triples.append(
                    RelationTriple(
                        head_text=s.head.text,
                        head_label=s.head.label,
                        tail_text=s.tail.text,
                        tail_label=s.tail.label,
                        relation_type=s.relation,
                        relation_name=relation_name(s.relation),
                        properties=dict(s.metadata.get("properties", {})),
                        source_path=doc.source_path,
                    )
                )
            doc.metadata["relations"] = [t.to_dict() for t in triples]
            return triples

        return []

    def run(
        self,
        input_path: Path | str,
        *,
        recursive: bool = True,
        run_vision: bool = False,
        run_ner: bool = True,
        run_re: bool = True,
        export_csv: bool = True,
        merge_existing: bool = True,
        limit: Optional[int] = None,
        **kwargs,
    ) -> PipelineResult:
        input_path = Path(input_path)
        if input_path.is_file():
            docs_paths = [input_path]
        else:
            from data_handler.parsers.registry import supported_suffixes

            allowed = set(supported_suffixes())
            docs_paths = []
            pattern = "**/*" if recursive else "*"
            for p in sorted(input_path.glob(pattern)):
                if p.is_file() and p.suffix.lower() in allowed:
                    docs_paths.append(p)
        if limit:
            docs_paths = docs_paths[:limit]

        batch: List[tuple] = []
        result = PipelineResult()

        for path in docs_paths:
            try:
                doc, ents, rels = self.process_document(
                    path,
                    run_vision=run_vision,
                    run_ner=run_ner,
                    run_re=run_re,
                    **kwargs,
                )
                result.documents.append(doc)
                result.entities_count += len(ents)
                result.relations_count += len(rels)
                batch.append((doc, ents, rels))
            except Exception as e:
                logger.error("处理失败 %s: %s", path, e)

        if export_csv and batch:
            exporter = GraphExporter(config=self.config)
            export_result = exporter.export_batch(batch, skip_existing=merge_existing)
            result.skipped_paths = export_result.skipped
            nodes_p, rels_p = exporter.write_csv(
                export_result, merge_existing=merge_existing
            )
            result.nodes_csv = str(nodes_p)
            result.relationships_csv = str(rels_p)

        logger.info(
            "流水线完成: %d 篇文档, %d 实体, %d 关系",
            len(result.documents),
            result.entities_count,
            result.relations_count,
        )
        return result


def run_pipeline(
    input_path: str | Path,
    config: Optional[DataHandlerConfig] = None,
    **kwargs,
) -> PipelineResult:
    setup_logging()
    pipeline = DataPipeline(config)
    return pipeline.run(input_path, **kwargs)

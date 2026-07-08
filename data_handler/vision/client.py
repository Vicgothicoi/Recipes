"""OpenAI 兼容的视觉 API 客户端。"""

from __future__ import annotations

import base64
import logging
import mimetypes
import os
from pathlib import Path
from typing import List, Optional

from openai import OpenAI

logger = logging.getLogger(__name__)

_DEFAULT_PROMPT = """你是一位烹饪领域的图像分析助手。请用中文描述这张与菜谱相关的图片，重点包括：
1. 可见的食材、调料或厨具
2. 菜肴状态（生/熟、成品/半成品）及外观
3. 若能看出烹饪步骤或操作，请简要说明
描述应客观、简洁，约 80～200 字，不要编造图中没有的内容。"""


class VisionClient:
    """调用兼容 OpenAI Chat Completions 的视觉模型。"""

    def __init__(
        self,
        *,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        default_prompt: Optional[str] = None,
    ) -> None:
        self.model = model or os.getenv("VISION_MODEL") or os.getenv("LLM_MODEL", "")
        if not self.model:
            raise ValueError("请设置 VISION_MODEL 或 LLM_MODEL 环境变量")

        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("请设置 OPENAI_API_KEY 环境变量")

        self.base_url = base_url or os.getenv(
            "OPENAI_BASE_URL", "https://api.siliconflow.cn/v1"
        )
        self.max_tokens = max_tokens or int(os.getenv("VISION_MAX_TOKENS", "512"))
        self.temperature = (
            temperature
            if temperature is not None
            else float(os.getenv("VISION_TEMPERATURE", "0.2"))
        )
        self.default_prompt = default_prompt or os.getenv(
            "VISION_PROMPT", _DEFAULT_PROMPT
        )

        self.client = OpenAI(api_key=api_key, base_url=self.base_url)
        logger.info(
            "VisionClient 已初始化: model=%s, base_url=%s",
            self.model,
            self.base_url,
        )

    def describe_image(
        self,
        image_path: str,
        *,
        prompt: Optional[str] = None,
        context: Optional[str] = None,
    ) -> str:
        """对单张本地或远程图片生成描述。"""
        image_url = self._to_image_url(image_path)
        user_text = prompt or self.default_prompt
        if context:
            user_text = f"{user_text}\n\n上下文（菜谱片段）：\n{context}"

        content: List[dict] = [
            {"type": "text", "text": user_text},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        message = response.choices[0].message
        text = (message.content or "").strip()
        if not text:
            raise RuntimeError(f"视觉模型返回空内容: {image_path}")
        return text

    def _to_image_url(self, image_path: str) -> str:
        if image_path.startswith(("http://", "https://", "data:")):
            return image_path

        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"图像文件不存在: {image_path}")

        mime, _ = mimetypes.guess_type(str(path))
        if not mime or not mime.startswith("image/"):
            mime = "image/jpeg"

        data = base64.standard_b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{data}"

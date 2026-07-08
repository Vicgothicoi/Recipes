"""离线数据处理：文档解析、图谱构建等。"""

from data_handler.schemas import EntitySpan, ParsedDocument, RelationTriple, TextBlock

try:
    from data_handler.vision import ImageDescriptor, VisionClient
except ImportError:
    ImageDescriptor = None  # type: ignore[misc, assignment]
    VisionClient = None  # type: ignore[misc, assignment]

try:
    from data_handler.ner import NerPredictor, NerTrainer, NerSample
except ImportError:
    NerPredictor = None  # type: ignore[misc, assignment]
    NerTrainer = None  # type: ignore[misc, assignment]
    NerSample = None  # type: ignore[misc, assignment]

try:
    from data_handler.re import RePredictor, ReTrainer, ReSample
except ImportError:
    RePredictor = None  # type: ignore[misc, assignment]
    ReTrainer = None  # type: ignore[misc, assignment]
    ReSample = None  # type: ignore[misc, assignment]

from data_handler.export import GraphExporter, IdAllocator
from data_handler.pipeline import DataPipeline, PipelineResult, run_pipeline

__all__ = [
    "ParsedDocument",
    "TextBlock",
    "EntitySpan",
    "RelationTriple",
    "VisionClient",
    "ImageDescriptor",
    "NerTrainer",
    "NerPredictor",
    "NerSample",
    "ReTrainer",
    "RePredictor",
    "ReSample",
    "GraphExporter",
    "IdAllocator",
    "DataPipeline",
    "PipelineResult",
    "run_pipeline",
]

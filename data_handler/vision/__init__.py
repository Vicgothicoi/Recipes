"""视觉大模型：图像定位与描述生成。"""

from data_handler.vision.client import VisionClient
from data_handler.vision.image_descriptor import ImageDescriptor
from data_handler.vision.image_locator import LocatedImage, collect_images, unique_image_paths

__all__ = [
    "VisionClient",
    "ImageDescriptor",
    "LocatedImage",
    "collect_images",
    "unique_image_paths",
]

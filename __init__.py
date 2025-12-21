
"""
NewBie ComfyUI Custom Nodes
为NewBie模型提供ComfyUI集成支持
"""

from .comfy_newbie_clip_loader import NODE_CLASS_MAPPINGS as CLIP_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as CLIP_DISPLAY_MAPPINGS
from .comfy_newbie_unet_loader import NODE_CLASS_MAPPINGS as UNET_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as UNET_DISPLAY_MAPPINGS
from .comfy_newbie_prompt_separator import NODE_CLASS_MAPPINGS as PROMPT_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as PROMPT_DISPLAY_MAPPINGS
from .comfy_newbie_clip_text_encode import NODE_CLASS_MAPPINGS as TEXT_ENCODE_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as TEXT_ENCODE_DISPLAY_MAPPINGS
from .comfy_newbie_model_sampling import ModelSamplingNewbie
from .comfy_newbie_xml_builder_nodes import NODE_CLASS_MAPPINGS as XML_MAPPINGS
from .comfy_newbie_lora_loader import NODE_CLASS_MAPPINGS as LORA_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as LORA_DISPLAY_MAPPINGS
from .comfy_newbie_gemma_chat import NODE_CLASS_MAPPINGS as GEMMA_CHAT_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as GEMMA_CHAT_DISPLAY_MAPPINGS
from .comfy_newbie_xml_prompt_builder import NODE_CLASS_MAPPINGS as XML_PROMPT_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as XML_PROMPT_DISPLAY_MAPPINGS

# 合并所有节点映射
NODE_CLASS_MAPPINGS = {
    **CLIP_MAPPINGS,
    **UNET_MAPPINGS,
    **PROMPT_MAPPINGS,
    **TEXT_ENCODE_MAPPINGS,
    **XML_MAPPINGS,
    **LORA_MAPPINGS,
    **GEMMA_CHAT_MAPPINGS,
    **XML_PROMPT_MAPPINGS,
    "ModelSamplingNewbie": ModelSamplingNewbie,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    **CLIP_DISPLAY_MAPPINGS,
    **UNET_DISPLAY_MAPPINGS,
    **PROMPT_DISPLAY_MAPPINGS,
    **TEXT_ENCODE_DISPLAY_MAPPINGS,
    **LORA_DISPLAY_MAPPINGS,
    **GEMMA_CHAT_DISPLAY_MAPPINGS,
    **XML_PROMPT_DISPLAY_MAPPINGS,
    "ModelSamplingNewbie": "Model Sampling (Newbie)",
}

# 导出节点映射
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

# 版本信息
__version__ = "1.0.0"
__author__ = "NewBie Team"
__description__ = "NewBie model support for ComfyUI"

# ComfyUI节点注册
WEB_DIRECTORY = "./web"

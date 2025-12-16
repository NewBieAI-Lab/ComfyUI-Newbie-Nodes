from .configuration_clip import JinaCLIPConfig, JinaCLIPTextConfig, JinaCLIPVisionConfig
from .modeling_clip import JinaCLIPModel, JinaCLIPTextModel, JinaCLIPVisionModel
from .eva_model import EVAVisionTransformer
from .hf_model import HFTextEncoder
from .rope_embeddings import VisionRotaryEmbeddingFast
from .transform import image_transform, OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from .processing_clip import JinaCLIPProcessor, JinaCLIPImageProcessor

# Register with transformers AutoConfig and AutoModel
try:
    from transformers import AutoConfig, AutoModel
    AutoConfig.register("jina_clip", JinaCLIPConfig)
    AutoModel.register(JinaCLIPConfig, JinaCLIPModel)
except Exception as e:
    print(f"Warning: Could not register JinaCLIP with transformers: {e}")

__all__ = [
    'JinaCLIPConfig',
    'JinaCLIPTextConfig',
    'JinaCLIPVisionConfig',
    'JinaCLIPModel',
    'JinaCLIPTextModel',
    'JinaCLIPVisionModel',
    'EVAVisionTransformer',
    'HFTextEncoder',
    'VisionRotaryEmbeddingFast',
    'image_transform',
    'OPENAI_DATASET_MEAN',
    'OPENAI_DATASET_STD',
    'JinaCLIPProcessor',
    'JinaCLIPImageProcessor',
]

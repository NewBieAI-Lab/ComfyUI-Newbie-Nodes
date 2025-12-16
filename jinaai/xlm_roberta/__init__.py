from .configuration_xlm_roberta import XLMRobertaFlashConfig
from .modeling_xlm_roberta import XLMRobertaModel, XLMRobertaForMaskedLM
from .modeling_lora import XLMRobertaLoRA
from .block import Block
from .embedding import XLMRobertaEmbeddings
from .mha import MHA
from .mlp import Mlp
from .rotary import RotaryEmbedding
from .xlm_padding import unpad_input, pad_input
from .stochastic_depth import StochasticDepth

# Register with transformers AutoConfig and AutoModel
try:
    from transformers import AutoConfig, AutoModel
    AutoConfig.register("xlm-roberta-flash", XLMRobertaFlashConfig)
    AutoModel.register(XLMRobertaFlashConfig, XLMRobertaModel)
except Exception as e:
    print(f"Warning: Could not register XLMRobertaFlash with transformers: {e}")

__all__ = [
    'XLMRobertaFlashConfig',
    'XLMRobertaModel',
    'XLMRobertaLoRA',
    'XLMRobertaForMaskedLM',
    'Block',
    'XLMRobertaEmbeddings',
    'MHA',
    'Mlp',
    'RotaryEmbedding',
    'unpad_input',
    'pad_input',
    'StochasticDepth',
]

import re
import warnings
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, PretrainedConfig
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    BaseModelOutputWithPoolingAndCrossAttentions,
)

# Import and register local xlm_roberta module (no trust_remote_code needed)
try:
    from ..xlm_roberta import XLMRobertaFlashConfig, XLMRobertaLoRA
    _LOCAL_XLM_ROBERTA = True
except ImportError as e:
    _LOCAL_XLM_ROBERTA = False
    print(f"[hf_model] Failed to import local xlm_roberta: {e}")

_HF_ARCH_DICT = {
    # https://huggingface.co/docs/transformers/model_doc/roberta#roberta
    'roberta': {
        'config_names': {
            'context_length': 'max_position_embeddings',
            'vocab_size': 'vocab_size',
            'width': 'hidden_size',
            'heads': 'num_attention_heads',
            'layers': 'num_hidden_layers',
            'layer_attr': 'layer',
            'token_embeddings_attr': 'embeddings',
        },
        'pooler': 'mean_pooler',
    },
    # https://huggingface.co/docs/transformers/model_doc/xlm-roberta#transformers.XLMRobertaConfig
    'xlm-roberta': {
        'config_names': {
            'context_length': 'max_position_embeddings',
            'vocab_size': 'vocab_size',
            'width': 'hidden_size',
            'heads': 'num_attention_heads',
            'layers': 'num_hidden_layers',
            'layer_attr': 'layer',
            'token_embeddings_attr': 'embeddings',
        },
        'pooler': 'mean_pooler',
    },
    # https://huggingface.co/docs/transformers/model_doc/bert
    'bert': {
        'config_names': {
            'context_length': 'max_position_embeddings',
            'vocab_size': 'vocab_size',
            'width': 'hidden_size',
            'heads': 'num_attention_heads',
            'layers': 'num_hidden_layers',
        },
        'pooler': 'cls_pooler',
    },
}

_POOLERS = {}


def _camel2snake(s):
    return re.sub(r'(?<!^)(?=[A-Z])', '_', s).lower()


def register_pooler(cls):
    """Decorator registering pooler class"""
    _POOLERS[_camel2snake(cls.__name__)] = cls
    return cls


@register_pooler
class MeanPooler(nn.Module):
    @staticmethod
    def forward(x: BaseModelOutput, attention_mask: torch.Tensor):
        masked_output = x.last_hidden_state * attention_mask.unsqueeze(-1)
        return masked_output.sum(dim=1) / attention_mask.sum(-1, keepdim=True)


@register_pooler
class MaxPooler(nn.Module):
    @staticmethod
    def forward(x: BaseModelOutput, attention_mask: torch.Tensor):
        masked_output = x.last_hidden_state.masked_fill(
            attention_mask.unsqueeze(-1), -torch.inf
        )
        return masked_output.max(1).values


@register_pooler
class ClsPooler(nn.Module):
    def __init__(self, use_pooler_output: bool = True):
        super().__init__()
        self.cls_token_position = 0
        self.use_pooler_output = use_pooler_output

    def forward(self, x: BaseModelOutput, _: torch.Tensor):
        if (
            self.use_pooler_output
            and isinstance(
                x,
                (
                    BaseModelOutputWithPooling,
                    BaseModelOutputWithPoolingAndCrossAttentions,
                ),
            )
            and (x.pooler_output is not None)
        ):
            return x.pooler_output
        return x.last_hidden_state[:, self.cls_token_position, :]


class HFTextEncoder(nn.Module):
    output_tokens: torch.jit.Final[bool]

    def __init__(
        self,
        model_name_or_path: str,
        output_dim: int,
        config: PretrainedConfig = None,
        pooler_type: str = None,
        proj_type: str = None,
        proj_bias: bool = False,
        pretrained: bool = True,
        output_tokens: bool = False,
        trust_remote_code: bool = False,
        revision: Optional[str] = None,
        code_revision: Optional[str] = None,
        default_instruction_task: Optional[str] = None,
        default_lora_task: Optional[str] = None,
        model_config_kwargs: Optional[Dict] = None,
        local_files_path: Optional[str] = None,
    ):
        super().__init__()
        self.output_tokens = output_tokens
        self.output_dim = output_dim

        model_config_kwargs = model_config_kwargs or {}

        # Check if this is a jinaai model that needs local xlm_roberta module
        # Matches: jinaai/jina-embeddings-v3, jinaai/xlm-roberta-flash-implementation, etc.
        _use_local_xlm = _LOCAL_XLM_ROBERTA and ('jinaai/' in model_name_or_path or 'xlm-roberta-flash' in model_name_or_path)

        if config is None:
            if pretrained:
                if _use_local_xlm:
                    # Use local XLMRobertaLoRA directly (load weights from HF)
                    self.config = XLMRobertaFlashConfig.from_pretrained(model_name_or_path)
                    self.config.update(model_config_kwargs)
                    self.transformer = XLMRobertaLoRA.from_pretrained(
                        model_name_or_path,
                        config=self.config,
                    )
                else:
                    self.transformer = AutoModel.from_pretrained(
                        model_name_or_path,
                        trust_remote_code=trust_remote_code,
                        revision=revision,
                        add_pooling_layer=False,
                        code_revision=code_revision,
                        **model_config_kwargs,
                    )
                    self.config = self.transformer.config
            else:
                if _use_local_xlm:
                    # Use local XLMRobertaLoRA directly (no remote fetch needed)
                    # Config comes from model_config_kwargs passed by JinaCLIP
                    # XLMRobertaLoRA wraps XLMRobertaModel with self.roberta attribute
                    # to match checkpoint weight names: text_model.transformer.roberta.xxx
                    self.config = XLMRobertaFlashConfig(**model_config_kwargs)
                    # Set _name_or_path for tokenizer loading (use local path if available)
                    self.config._name_or_path = local_files_path or model_name_or_path
                    self.transformer = XLMRobertaLoRA(self.config, add_pooling_layer=False)
                else:
                    self.config = AutoConfig.from_pretrained(
                        model_name_or_path,
                        trust_remote_code=trust_remote_code,
                        code_revision=code_revision,
                    )
                    self.config.update(model_config_kwargs)
                    self.transformer = AutoModel.from_config(
                        self.config,
                        trust_remote_code=trust_remote_code,
                        add_pooling_layer=False,
                        code_revision=code_revision,
                    )
            if (
                hasattr(self.config, 'is_encoder_decoder')
                and self.config.is_encoder_decoder
            ):
                self.transformer = self.transformer.encoder

        else:
            self.config = config
            self.config.update(model_config_kwargs)
            if _use_local_xlm:
                # Set _name_or_path for tokenizer loading (use local path if available)
                if local_files_path:
                    self.config._name_or_path = local_files_path
                # Use XLMRobertaLoRA to match checkpoint weight names
                self.transformer = XLMRobertaLoRA(self.config, add_pooling_layer=False)
            else:
                self.transformer = AutoModel.from_config(
                    self.config,
                    trust_remote_code=trust_remote_code,
                    revision=revision,
                    code_revision=code_revision,
                )
        self.vocab_size = getattr(self.config, 'vocab_size', 0)
        self.context_length = getattr(self.config, 'max_position_embeddings', 0)

        pooler_type = pooler_type or _HF_ARCH_DICT[self.config.model_type]['pooler']
        self.pooler = _POOLERS[pooler_type]()

        d_model = getattr(
            self.config, _HF_ARCH_DICT[self.config.model_type]['config_names']['width']
        )
        if (d_model == output_dim) and (proj_type is None):  # do we always need a proj?
            self.proj = nn.Identity()
        elif (d_model != output_dim) or proj_type == 'linear':
            self.proj = nn.Linear(d_model, output_dim, bias=proj_bias)
        elif proj_type == 'mlp':
            hidden_size = (d_model + output_dim) // 2
            self.proj = nn.Sequential(
                nn.Linear(d_model, hidden_size, bias=proj_bias),
                nn.GELU(),
                nn.Linear(hidden_size, output_dim, bias=proj_bias),
            )

        self._task_instructions = {}
        self._lora_adaptation_map = {}
        self._supports_task_instructions = False
        self._supports_lora = False
        if (
            hasattr(self.transformer, '_adaptation_map')
            and len(self.transformer._adaptation_map) > 0
        ):
            self._lora_adaptation_map = self.transformer._adaptation_map
            self._supports_lora = True
        if (
            hasattr(self.transformer, '_task_instructions')
            and len(self.transformer._task_instructions) > 0
        ):
            self._task_instructions = self.transformer._task_instructions
            self._supports_task_instructions = True

        self._default_instruction_task = None
        self._default_lora_task = None
        self._default_instruction = None
        self._default_loraid = None

        if default_instruction_task is not None:
            self._default_instruction_task = default_instruction_task
            self._default_instruction = self.get_instruction_from_task(
                default_instruction_task
            )
        if default_lora_task is not None:
            self._default_lora_task = default_lora_task
            self._default_loraid = self.get_loraid_from_task(default_lora_task)

    @property
    def supports_task_instructions(self) -> bool:
        return self._supports_task_instructions

    @property
    def supports_lora(self) -> bool:
        return self._supports_lora

    @property
    def task_instructions(self) -> Dict[str, str]:
        return self._task_instructions

    @property
    def lora_adaptation_map(self) -> Dict[str, int]:
        return self._lora_adaptation_map

    @property
    def default_instruction(self) -> Optional[str]:
        return self._default_instruction

    @property
    def default_loraid(self) -> Optional[int]:
        return self._default_loraid

    def get_instruction_from_task(self, task: Optional[str]) -> Optional[str]:
        if self._supports_task_instructions:
            if task is None:
                return self._default_instruction
            if task not in self._task_instructions:
                raise ValueError(
                    f'Unsupported task \'{task}\'. Choose one of the following: '
                    f'{", ".join(self._task_instructions)} or set to None to disable '
                    f'task instructions completely'
                )
            return self._task_instructions[task]
        else:
            if task is not None:
                warnings.warn(
                    'Model does not support task instructions, ignoring instruction '
                    f"task '{task}'"
                )
        return None

    def get_loraid_from_task(self, task: Optional[str]) -> Optional[int]:
        if self._supports_lora:
            if task is None:
                return self._default_loraid
            if task not in self._lora_adaptation_map:
                raise ValueError(
                    f'Unsupported task \'{task}\'. Choose one of the following: '
                    f'{", ".join(self._task_instructions)} or set to None to disable '
                    f'the LoRA adapters completely'
                )
            return self._lora_adaptation_map[task]
        else:
            if task is not None:
                warnings.warn(
                    f"Model does not support LoRA adapters, ignoring LoRA task '{task}'"
                )
        return None

    @staticmethod
    def get_adapter_mask_from_loraid(
        batch_size: int, loraid: int, device: Union[str, torch.device]
    ):
        return torch.full((batch_size,), loraid, dtype=torch.int32, device=device)

    @torch.jit.ignore
    def set_grad_checkpointing(self, _=True):
        self.transformer.gradient_checkpointing_enable()

    def init_parameters(self):
        pass

    def forward(self, x: torch.Tensor, adapter_mask: Optional[torch.Tensor] = None):
        if adapter_mask is None:
            default_loraid = self.default_loraid
            if default_loraid is not None:
                adapter_mask = self.get_adapter_mask_from_loraid(
                    x.shape[0], default_loraid, x.device
                )
        else:
            if not self.supports_lora:
                warnings.warn(
                    'Model does not support LoRA adapters, setting adapter_mask to None'
                )
                adapter_mask = None

        attention_mask = (x != self.config.pad_token_id).long()
        lora_kwargs = {}
        if adapter_mask is not None:
            lora_kwargs['adapter_mask'] = adapter_mask

        out = self.transformer(
            input_ids=x, attention_mask=attention_mask, **lora_kwargs
        )
        pooled_out = self.pooler(out, attention_mask)
        projected = self.proj(pooled_out)
        seqlen = out.last_hidden_state.shape[1]
        tokens = (
            out.last_hidden_state[
                :, torch.arange(seqlen) != self.pooler.cls_token_position, :
            ]
            if isinstance(self.pooler, ClsPooler)
            else out.last_hidden_state
        )
        if self.output_tokens:
            return projected, tokens
        return projected

    def lock(self, unlocked_layers: int = 0, freeze_layer_norm: bool = True):
        if not unlocked_layers:
            for n, p in self.transformer.named_parameters():
                p.requires_grad = (
                    (not freeze_layer_norm) if 'LayerNorm' in n.split('.') else False
                )
            return

        encoder = (
            self.transformer.encoder
            if hasattr(self.transformer, 'encoder')
            else self.transformer
        )
        layer_list = getattr(
            encoder, _HF_ARCH_DICT[self.config.model_type]['config_names']['layer_attr']
        )
        print(f'Unlocking {unlocked_layers}/{len(layer_list) + 1} layers of hf model')
        embeddings = getattr(
            self.transformer,
            _HF_ARCH_DICT[self.config.model_type]['config_names'][
                'token_embeddings_attr'
            ],
        )
        modules = [embeddings, *layer_list][:-unlocked_layers]
        # freeze layers
        for module in modules:
            for n, p in module.named_parameters():
                p.requires_grad = (
                    (not freeze_layer_norm) if 'LayerNorm' in n.split('.') else False
                )

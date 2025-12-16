# coding=utf-8
#
# Code mainly copied from:
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/clip/configuration_clip.py
# and adjusted for Jina CLIP

import os
from copy import deepcopy
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import PretrainedConfig, logging

logger = logging.get_logger(__name__)


""" Jina CLIP model configuration """


class JinaCLIPTextConfig(PretrainedConfig):
    model_type = 'jina_clip_text'

    def __init__(
        self,
        embed_dim: int = 768,
        hf_model_name_or_path: str = 'jinaai/jina-bert-flash-implementation',
        hf_model_config_kwargs: Optional[Dict[str, Any]] = None,
        default_instruction_task: Optional[str] = None,
        default_lora_task: Optional[str] = None,
        pooler_type: Optional[str] = None,
        proj_type: Optional[str] = None,
        proj_bias: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.embed_dim = embed_dim
        self.hf_model_name_or_path = hf_model_name_or_path
        self.hf_model_config_kwargs = hf_model_config_kwargs or {}
        self.default_instruction_task = default_instruction_task
        self.default_lora_task = default_lora_task
        self.pooler_type = pooler_type
        self.proj_type = proj_type
        self.proj_bias = proj_bias

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> 'PretrainedConfig':
        cls._set_token_in_kwargs(kwargs)

        configdict, kwargs = cls.get_config_dict(
            pretrained_model_name_or_path, **kwargs
        )
        # get the text config dict if we are loading from JinaCLIPConfig
        if configdict.get('model_type') == 'jina_clip':
            configdict = configdict['text_config']
        if (
            'model_type' in configdict
            and hasattr(cls, 'model_type')
            and configdict['model_type'] != cls.model_type
        ):
            logger.warning(
                f'You are using a model of type {configdict["model_type"]} to '
                f'instantiate a model of type {cls.model_type}. This is not supported '
                'for all configurations of models and can yield errors.'
            )
        return cls.from_dict(configdict, **kwargs)


class JinaCLIPVisionConfig(PretrainedConfig):
    model_type = 'jina_clip_vision'

    def __init__(
        self,
        embed_dim: int = 768,
        width: int = 768,
        image_size: int = 224,
        patch_size: int = 16,
        layers: int = 12,
        head_width: int = 64,
        mlp_ratio: float = 4.0,
        ls_init_value: Optional[float] = None,
        patch_dropout: float = 0.0,
        qkv_bias: bool = True,
        fused_layer_norm: bool = False,
        x_attention: bool = False,
        post_norm: bool = False,
        rope_embeddings: bool = False,
        pt_hw_seq_len: int = 16,
        intp_freq: bool = False,
        naive_swiglu: bool = False,
        subln: bool = False,
        drop_path_rate: float = 0.0,
        proj_type: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.layers = layers
        self.embed_dim = embed_dim
        self.width = width
        self.head_width = head_width
        self.mlp_ratio = mlp_ratio
        self.image_size = image_size
        self.patch_size = patch_size
        self.ls_init_value = ls_init_value
        self.patch_dropout = patch_dropout
        self.qkv_bias = qkv_bias
        self.fused_layer_norm = fused_layer_norm
        self.x_attention = x_attention
        self.post_norm = post_norm
        self.rope_embeddings = rope_embeddings
        self.pt_hw_seq_len = pt_hw_seq_len
        self.intp_freq = intp_freq
        self.naive_swiglu = naive_swiglu
        self.subln = subln
        self.drop_path_rate = drop_path_rate
        self.proj_type = proj_type

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> 'PretrainedConfig':
        cls._set_token_in_kwargs(kwargs)

        configdict, kwargs = cls.get_config_dict(
            pretrained_model_name_or_path, **kwargs
        )
        # get the vision config dict if we are loading from JinaCLIPConfig
        if configdict.get('model_type') == 'jina_clip':
            configdict = configdict['vision_config']
        if (
            'model_type' in configdict
            and hasattr(cls, 'model_type')
            and configdict['model_type'] != cls.model_type
        ):
            logger.warning(
                f'You are using a model of type {configdict["model_type"]} to '
                f'instantiate a model of type {cls.model_type}. This is not supported '
                'for all configurations of models and can yield errors.'
            )
        return cls.from_dict(configdict, **kwargs)


class JinaCLIPConfig(PretrainedConfig):
    model_type = 'jina_clip'
    is_composition = True

    def __init__(
        self,
        text_config: Optional[Dict] = None,
        vision_config: Optional[Dict] = None,
        add_projections: bool = False,
        projection_dim: int = 768,
        logit_scale_init_value: float = 2.6592,
        use_text_flash_attn: Optional[bool] = None,
        use_vision_xformers: Optional[bool] = None,
        matryoshka_dimensions: Optional[List[int]] = None,
        truncate_dim: Optional[int] = None,
        torch_dtype: Optional[Union[str, torch.dtype]] = None,
        **kwargs,
    ):
        # If `_config_dict` exist, we use them for the backward compatibility.
        # We pop out these 2 attributes before calling `super().__init__` to avoid
        # them being saved (which causes a lot of confusion!).

        text_config_dict: Optional[Dict] = kwargs.pop('text_config_dict', None)
        vision_config_dict: Optional[Dict] = kwargs.pop('vision_config_dict', None)
        self.use_text_flash_attn = use_text_flash_attn
        self.use_vision_xformers = use_vision_xformers
        self.matryoshka_dimensions = matryoshka_dimensions
        self.truncate_dim = truncate_dim

        super().__init__(**kwargs)

        if text_config_dict is not None:
            if text_config is None:
                text_config = {}

            # This is the complete result when using `text_config_dict`.
            _text_config_dict = JinaCLIPTextConfig(**text_config_dict).to_dict()

            # Give a warning if the values exist in both `_text_config_dict` and
            # `text_config` but being different.
            for key, value in _text_config_dict.items():
                if (
                    key in text_config
                    and value != text_config[key]
                    and key not in ['transformers_version']
                ):
                    # If specified in `text_config_dict`
                    if key in text_config_dict:
                        message = (
                            f'`{key}` is found in both `text_config_dict` and '
                            f'`text_config` but with different values. '
                            f'The value `text_config_dict["{key}"]` will be used '
                            f'instead.'
                        )
                    # If inferred from default argument values (
                    # just to be super careful)
                    else:
                        message = (
                            f'`text_config_dict` is provided which will be used to '
                            f'initialize `JinaCLIPTextConfig`. The '
                            f'value `text_config["{key}"]` will be overriden.'
                        )
                    logger.info(message)

            # Update all values in `text_config` with the ones in `_text_config_dict`.
            text_config.update(_text_config_dict)

        if vision_config_dict is not None:
            if vision_config is None:
                vision_config = {}

            # This is the complete result when using `vision_config_dict`.
            _vision_config_dict = JinaCLIPVisionConfig(**vision_config_dict).to_dict()
            # convert keys to string instead of integer
            if 'id2label' in _vision_config_dict:
                _vision_config_dict['id2label'] = {
                    str(key): value
                    for key, value in _vision_config_dict['id2label'].items()
                }

            # Give a warning if the values exist in both `_vision_config_dict`
            # and `vision_config` but being different.
            for key, value in _vision_config_dict.items():
                if (
                    key in vision_config
                    and value != vision_config[key]
                    and key not in ['transformers_version']
                ):
                    # If specified in `vision_config_dict`
                    if key in vision_config_dict:
                        message = (
                            f'`{key}` is found in both `vision_config_dict` and '
                            f'`vision_config` but with different '
                            f'values. The value `vision_config_dict["{key}"]` will '
                            f'be used instead.'
                        )
                    # If inferred from default argument values
                    # (just to be super careful)
                    else:
                        message = (
                            f'`vision_config_dict` is provided which will be used to '
                            f'initialize `JinaCLIPVisionConfig`. '
                            f'The value `vision_config["{key}"]` will be overriden.'
                        )
                    logger.info(message)

            # Update all values in `vision_config` with the ones in
            # `_vision_config_dict`.
            vision_config.update(_vision_config_dict)

        if text_config is None:
            text_config = {}
            logger.info(
                '`text_config` is `None`. Initializing the `JinaCLIPTextConfig` with '
                'default values.'
            )

        if vision_config is None:
            vision_config = {}
            logger.info(
                '`vision_config` is `None`. initializing the `JinaCLIPVisionConfig` '
                'with default values.'
            )

        self.text_config = JinaCLIPTextConfig(**text_config)
        self.vision_config = JinaCLIPVisionConfig(**vision_config)

        self.add_projections = add_projections
        self.projection_dim = projection_dim
        self.logit_scale_init_value = logit_scale_init_value
        self.initializer_factor = 1.0

        if not self.add_projections:
            if self.text_config.embed_dim != self.vision_config.embed_dim:
                raise ValueError(
                    'When projections are disabled (`add_projections=False`), text '
                    'and vision towers need to have the same embedding dimensionality. '
                    f'Currently text embedding dim is {self.text_config.embed_dim} != '
                    f'{self.vision_config.embed_dim} of the vision tower. '
                    'Either set the same output dim for both towers, or enable '
                    'projections with `add_projections=True`.'
                )

        if (
            torch_dtype
            and hasattr(torch, torch_dtype)
            and type(getattr(torch, torch_dtype)) is torch.dtype
        ):
            self.torch_dtype = getattr(torch, torch_dtype)
        else:
            self.torch_dtype = torch_dtype

        use_text_flash_attn = (
            self.use_text_flash_attn if self.use_text_flash_attn is not None
            else self.text_config.hf_model_config_kwargs.get('use_flash_attn', False)
        )
        if not use_text_flash_attn or not torch.cuda.is_available():
            self.torch_dtype = torch.float32

    @classmethod
    def from_text_vision_configs(
        cls,
        text_config: JinaCLIPTextConfig,
        vision_config: JinaCLIPVisionConfig,
        **kwargs,
    ):
        return cls(
            text_config=text_config.to_dict(),
            vision_config=vision_config.to_dict(),
            projection_dim=text_config.projection_dim,
            **kwargs,
        )

    def to_dict(self):
        output = deepcopy(self.__dict__)
        output['text_config'] = self.text_config.to_dict()
        output['vision_config'] = self.vision_config.to_dict()
        output['model_type'] = self.__class__.model_type
        return output

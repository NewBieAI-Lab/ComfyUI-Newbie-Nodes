# coding=utf-8
#
# Code mainly copied from:
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/clip/modeling_clip.py
# and adjusted for Jina CLIP

import base64
import importlib.util
import warnings
from functools import partial
from io import BytesIO
from typing import List, Optional, Tuple, Union

import numpy as np
import requests
import torch
import torch.nn.functional as f
import torch.utils.checkpoint
from PIL import Image
from torch import nn
from transformers import (
    AutoImageProcessor,
    AutoTokenizer,
    BatchEncoding,
    BatchFeature,
    PreTrainedModel,
    logging,
)
from transformers.models.clip.modeling_clip import (
    CLIPOutput,
    CLIPTextModelOutput,
    CLIPVisionModelOutput,
    clip_loss,
)

try:
    from tqdm.autonotebook import trange

    has_tqdm = True
except ImportError:
    trange = None
    has_tqdm = False

from .configuration_clip import JinaCLIPConfig, JinaCLIPTextConfig, JinaCLIPVisionConfig
from .eva_model import EVAVisionTransformer
from .hf_model import HFTextEncoder
from .rope_embeddings import VisionRotaryEmbeddingFast  # noqa: F401
from .transform import (  # noqa: F401
    OPENAI_DATASET_MEAN,
    OPENAI_DATASET_STD,
    image_transform,
)

logger = logging.get_logger(__name__)


""" Jina CLIP model implementation """


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm (with cast back to input dtype)."""

    def forward(self, x: torch.Tensor):
        origtype = x.dtype
        x = f.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(origtype)


def _build_text_tower(config: JinaCLIPTextConfig) -> HFTextEncoder:
    # Use local xlm_roberta module (no trust_remote_code needed)
    # Pass local_files_path for tokenizer loading (set by JinaCLIPModel.__init__)
    return HFTextEncoder(
        model_name_or_path=config.hf_model_name_or_path,
        output_dim=config.embed_dim,
        default_instruction_task=config.default_instruction_task,
        default_lora_task=config.default_lora_task,
        pooler_type=config.pooler_type,
        proj_type=config.proj_type,
        proj_bias=config.proj_bias,
        pretrained=False,
        output_tokens=False,
        trust_remote_code=False,  # Using local xlm_roberta module
        revision=None,
        model_config_kwargs=config.hf_model_config_kwargs,
        local_files_path=getattr(config, '_name_or_path', None),
    )


def _build_vision_tower(config: JinaCLIPVisionConfig) -> EVAVisionTransformer:
    norm_layer = partial(LayerNorm, eps=1e-6)

    if config.fused_layer_norm:
        try:
            from apex.normalization import FusedLayerNorm

            norm_layer = partial(FusedLayerNorm, eps=1e-6)
        except (ModuleNotFoundError, ImportError):
            logger.warning('Please install apex to use fused layer norm, ignoring')

    return EVAVisionTransformer(
        img_size=config.image_size,
        patch_size=config.patch_size,
        num_classes=config.embed_dim,
        use_mean_pooling=False,
        init_values=config.ls_init_value,
        patch_dropout=config.patch_dropout,
        embed_dim=config.width,
        depth=config.layers,
        num_heads=config.width // config.head_width,
        mlp_ratio=config.mlp_ratio,
        qkv_bias=config.qkv_bias,
        drop_path_rate=config.drop_path_rate,
        norm_layer=norm_layer,
        xattn=config.x_attention,
        rope=config.rope_embeddings,
        postnorm=config.post_norm,
        pt_hw_seq_len=config.pt_hw_seq_len,
        intp_freq=config.intp_freq,
        naiveswiglu=config.naive_swiglu,
        subln=config.subln,
        proj_type=config.proj_type,
    )


def _resolve_attention_libs(config: JinaCLIPConfig):
    # flash_attn dependency removed - always use PyTorch native attention
    # xformers is optional for vision, check if available
    use_vision_xformers = (
        config.use_vision_xformers
        if config.use_vision_xformers is not None
        else config.vision_config.x_attention
    )

    def _resolve_use_vision_xformers() -> bool:
        if use_vision_xformers:
            if not torch.cuda.is_available():
                warnings.warn('xFormers requires CUDA, disabling')
                return False
            if importlib.util.find_spec('xformers') is None:
                warnings.warn(
                    'xFormers is not installed. Check '
                    'https://github.com/facebookresearch/xformers?'
                    'tab=readme-ov-file#installing-xformers for installation '
                    'instructions, disabling'
                )
                return False
            return True
        return False

    _use_text_flash_attn = False  # Always disabled - using PyTorch native attention
    _use_vision_xformers = _resolve_use_vision_xformers()

    config.use_text_flash_attn = _use_text_flash_attn
    config.use_vision_xformers = _use_vision_xformers
    config.text_config.hf_model_config_kwargs['use_flash_attn'] = _use_text_flash_attn
    config.vision_config.x_attention = _use_vision_xformers

    return config


class JinaCLIPPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for
    downloading and loading pretrained models.
    """

    config_class = JinaCLIPConfig
    base_model_prefix = 'clip'
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, JinaCLIPModel):
            if isinstance(module.text_projection, nn.Linear):
                nn.init.normal_(
                    module.text_projection.weight,
                    std=module.text_embed_dim**-0.5 * self.config.initializer_factor,
                )
            if isinstance(module.text_projection, nn.Linear):
                nn.init.normal_(
                    module.visual_projection.weight,
                    std=module.vision_embed_dim**-0.5 * self.config.initializer_factor,
                )
        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        if 'torch_dtype' not in kwargs:
            kwargs['torch_dtype'] = 'auto'
        return super().from_pretrained(*args, **kwargs)


class JinaCLIPTextModel(JinaCLIPPreTrainedModel):
    config_class = JinaCLIPTextConfig

    def __init__(self, config: JinaCLIPTextConfig):
        super().__init__(config)
        self.text_model = _build_text_tower(config)
        self.post_init()

    def forward(
        self,
        input_ids: Union[None, torch.Tensor, BatchEncoding] = None,
        return_dict: Optional[bool] = None,
        *_,
        **__,
    ) -> Union[Tuple[Optional[torch.FloatTensor], ...], CLIPTextModelOutput]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        x = input_ids.input_ids if isinstance(input_ids, BatchEncoding) else input_ids
        feats = self.text_model(x=x)
        out = CLIPTextModelOutput(text_embeds=feats)
        return out if return_dict else out.to_tuple()


class JinaCLIPVisionModel(JinaCLIPPreTrainedModel):
    config_class = JinaCLIPVisionConfig
    main_input_name = 'pixel_values'

    def __init__(self, config: JinaCLIPVisionConfig):
        super().__init__(config)
        self.vision_model = _build_vision_tower(config)
        self.post_init()

    def forward(
        self,
        pixel_values: Union[None, torch.FloatTensor, BatchFeature] = None,
        return_dict: Optional[bool] = None,
        *_,
        **__,
    ) -> Union[Tuple[Optional[torch.FloatTensor], ...], CLIPVisionModelOutput]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        x = (
            pixel_values.pixel_values
            if isinstance(pixel_values, BatchFeature)
            else pixel_values
        )
        feats = self.vision_model(x=x)
        out = CLIPVisionModelOutput(image_embeds=feats)
        return out if return_dict else out.to_tuple()


class JinaCLIPModel(JinaCLIPPreTrainedModel):
    config_class = JinaCLIPConfig

    def __init__(self, config: JinaCLIPConfig):
        super().__init__(config)

        if not isinstance(config.text_config, JinaCLIPTextConfig):
            raise ValueError(
                'Attribute config.text_config is expected to be of type '
                f'JinaCLIPTextConfig but is of type {type(config.text_config)}.'
            )

        if not isinstance(config.vision_config, JinaCLIPVisionConfig):
            raise ValueError(
                'Attribute config.vision_config is expected to be of type '
                f'JinaCLIPVisionConfig but is of type {type(config.vision_config)}.'
            )

        config = _resolve_attention_libs(config)
        text_config = config.text_config
        vision_config = config.vision_config

        # Pass local path to text_config for tokenizer loading
        # config._name_or_path is set by from_pretrained() to the local model path
        text_config._name_or_path = getattr(config, '_name_or_path', '')

        self.add_projections = config.add_projections
        self.projection_dim = config.projection_dim
        self.text_embed_dim = text_config.embed_dim
        self.vision_embed_dim = vision_config.embed_dim
        self.text_model = _build_text_tower(text_config)
        self.vision_model = _build_vision_tower(vision_config)
        self.logit_scale = nn.Parameter(
            torch.tensor(self.config.logit_scale_init_value)
        )
        if self.add_projections:
            self.visual_projection = nn.Linear(
                self.vision_embed_dim, self.projection_dim, bias=False
            )
            self.text_projection = nn.Linear(
                self.text_embed_dim, self.projection_dim, bias=False
            )
        else:
            self.visual_projection = nn.Identity()
            self.text_projection = nn.Identity()

        self.tokenizer = None
        self.preprocess = None
        self.post_init()

    def get_tokenizer(self):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config._name_or_path, trust_remote_code=True
            )
        return self.tokenizer

    def get_preprocess(self):
        if not self.preprocess:
            self.preprocess = AutoImageProcessor.from_pretrained(
                self.config._name_or_path, trust_remote_code=True
            )
        return self.preprocess

    def get_text_features(
        self,
        input_ids: Union[None, torch.Tensor, BatchEncoding] = None,
        *_,
        **__,
    ) -> torch.FloatTensor:
        x = input_ids.input_ids if isinstance(input_ids, BatchEncoding) else input_ids
        return self.text_projection(self.text_model(x=x))

    def get_image_features(
        self,
        pixel_values: Union[None, torch.FloatTensor, BatchFeature] = None,
        *_,
        **__,
    ) -> torch.FloatTensor:
        x = (
            pixel_values.pixel_values
            if isinstance(pixel_values, BatchFeature)
            else pixel_values
        )
        return self.visual_projection(self.vision_model(x=x))

    def _truncate_embeddings(self, embeddings: torch.Tensor, truncate_dim: int):
        if not self.config.matryoshka_dimensions:
            logger.warning(
                'Model is not trained using Matryoshka Representation Learning, '
                'truncating embeddings will not work optimally.'
            )
        return embeddings[:, :truncate_dim]

    @staticmethod
    def _decode_image_data(image_data_str: str) -> Image:
        header, data = image_data_str.split(',', 1)
        image_data = base64.b64decode(data)
        return Image.open(BytesIO(image_data))

    @torch.inference_mode()
    def encode_image(
        self,
        images: Union[str, List[Union[str, 'Image.Image']]],
        batch_size: int = 32,
        show_progress_bar: Optional[bool] = None,
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        device: Optional[torch.device] = None,
        normalize_embeddings: bool = True,
        truncate_dim: Optional[int] = None,
    ) -> Union[List[torch.Tensor], np.ndarray, torch.Tensor]:
        """
        Computes image embeddings

        Args:
            images(`str` or `List[Union[str, Image.Image]]`):
                Image paths, URLs, PIL images, or data:image/ strings to be encoded
            batch_size(`int`, *optional*, defaults to 32):
                Batch size for the computation
            show_progress_bar(`bool`, *optional*, defaults to None):
                Show a progress bar when encoding images. If set to None, progress bar
                is only shown when `logger.level == logging.INFO` or
                `logger.level == logging.DEBUG`
            convert_to_numpy(`bool`, *optional*, defaults to True):
                If true, the output is a list of numpy vectors. Else, it is a list of
                pytorch tensors
            convert_to_tensor(`bool`, *optional*, defaults to False):
                If true, you get one large tensor as return. Overwrites any setting
                from convert_to_numpy
            device(`torch.device`, *optional*, defaults to None):
                Which torch.device to use for the computation
            normalize_embeddings(`bool`, *optional*, defaults to True):
                If set to true, returned vectors will have length 1. In that case,
                the faster dot-product (util.dot_score) instead of cosine similarity
                can be used
            truncate_dim(`int`, *optional*, defaults to None):
                The dimension to truncate sentence embeddings to. If set to `None`
                no truncation is performed

        Returns:
            By default, a list of tensors is returned. If convert_to_tensor, a stacked
            tensor is returned. If convert_to_numpy, a numpy matrix is returned
        """

        _is_training = self.training
        self.eval()

        self.preprocess = self.get_preprocess()
        all_embeddings = []

        if show_progress_bar is None:
            show_progress_bar = (
                logger.getEffectiveLevel() == logging.INFO
                or logger.getEffectiveLevel() == logging.DEBUG
            )
        if convert_to_tensor:
            convert_to_numpy = False

        _input_was_single_img = False
        if isinstance(images, str) or not hasattr(images, '__len__'):
            images = [images]
            _input_was_single_img = True

        if device is not None:
            self.to(device)

        _permutation = np.argsort([-len(str(i)) for i in images])
        _inverse_permutation = np.argsort(_permutation)
        images = [images[idx] for idx in _permutation]

        if has_tqdm:
            range_iter = trange(
                0,
                len(images),
                batch_size,
                desc='Encoding',
                disable=not show_progress_bar,
            )
        else:
            range_iter = range(0, len(images), batch_size)

        truncate_dim = truncate_dim or self.config.truncate_dim

        for i in range_iter:
            _processed_images = []
            for img in images[i: i + batch_size]:
                if isinstance(img, str):
                    if img.startswith('http'):
                        response = requests.get(img)
                        image = Image.open(BytesIO(response.content)).convert('RGB')
                    elif img.startswith('data:image/'):
                        image = self._decode_image_data(img).convert('RGB')
                    else:
                        image = Image.open(img).convert('RGB')
                elif isinstance(img, Image.Image):
                    image = img.convert('RGB')
                else:
                    raise ValueError('Unsupported image format')
                _processed_images.append(image)

            pixelvals = self.preprocess(_processed_images)
            pixelvals = pixelvals.to(self.device)
            embeddings = self.get_image_features(pixelvals)

            if truncate_dim:
                embeddings = self._truncate_embeddings(embeddings, truncate_dim)
            if normalize_embeddings:
                embeddings = f.normalize(embeddings, p=2, dim=1)
            if convert_to_numpy:
                embeddings = embeddings.cpu()

            all_embeddings.extend(embeddings)

        all_embeddings = [all_embeddings[idx] for idx in _inverse_permutation]

        if convert_to_tensor:
            all_embeddings = torch.stack(all_embeddings)
        elif convert_to_numpy:
            all_embeddings = np.asarray(
                [emb.to(torch.float32).numpy() for emb in all_embeddings]
            )

        if _input_was_single_img:
            all_embeddings = all_embeddings[0]

        self.train(_is_training)
        return all_embeddings

    @torch.inference_mode()
    def encode_text(
        self,
        sentences: Union[str, List[str]],
        task: Optional[str] = None,
        batch_size: int = 32,
        show_progress_bar: Optional[bool] = None,
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        device: Optional[torch.device] = None,
        normalize_embeddings: bool = True,
        truncate_dim: Optional[int] = None,
        **tokenizer_kwargs,
    ) -> Union[List[torch.Tensor], np.ndarray, torch.Tensor]:
        """
        Computes text embeddings

        Args:
            sentences(`str` or `List[str]`):
                Sentence or sentences to be encoded
            task(`str`, *optional*, defaults to `None`):
                Specifies the task for which the encoding is intended. If a `task` is
                provided, a task-specific instruction is added to the beginning of each
                sentence. If `task` is not provided, no instructions are added.
            batch_size(`int`, *optional*, defaults to 32):
                Batch size for the computation
            show_progress_bar(`bool`, *optional*, defaults to None):
                Show a progress bar when encoding sentences. If set to None, progress
                bar is only shown when `logger.level == logging.INFO` or
                `logger.level == logging.DEBUG`
            convert_to_numpy(`bool`, *optional*, defaults to True):
                If true, the output is a list of numpy vectors. Else, it is a list of
                pytorch tensors
            convert_to_tensor(`bool`, *optional*, defaults to False):
                If true, you get one large tensor as return. Overwrites any setting
                from convert_to_numpy
            device(`torch.device`, *optional*, defaults to None):
                Which torch.device to use for the computation
            normalize_embeddings(`bool`, *optional*, defaults to True):
                If set to true, returned vectors will have length 1. In that case,
                the faster dot-product (util.dot_score) instead of cosine similarity
                can be used
            truncate_dim(`int`, *optional*, defaults to None):
                The dimension to truncate sentence embeddings to. If set to `None`
                no truncation is performed
            tokenizer_kwargs(`Dict[str, Any]`, *optional*, defaults to {}):
                Keyword arguments for the tokenizer
        Returns:
            By default, a list of tensors is returned. If convert_to_tensor, a stacked
            tensor is returned. If convert_to_numpy, a numpy matrix is returned.
        """
        _is_training = self.training
        self.eval()

        all_embeddings = []
        self.tokenizer = self.get_tokenizer()

        if show_progress_bar is None:
            show_progress_bar = (
                logger.getEffectiveLevel() == logging.INFO
                or logger.getEffectiveLevel() == logging.DEBUG
            )
        if convert_to_tensor:
            convert_to_numpy = False

        _input_was_string = False
        if isinstance(sentences, str) or not hasattr(sentences, '__len__'):
            sentences = [sentences]
            _input_was_string = True

        if device is not None:
            self.to(device)

        _permutation = np.argsort([-len(i) for i in sentences])
        _inverse_permutation = np.argsort(_permutation)
        sentences = [sentences[idx] for idx in _permutation]

        tokenizer_kwargs['padding'] = tokenizer_kwargs.get('padding', True)
        tokenizer_kwargs['max_length'] = tokenizer_kwargs.get('max_length', 512)
        tokenizer_kwargs['truncation'] = tokenizer_kwargs.get('truncation', True)

        if has_tqdm:
            range_iter = trange(
                0,
                len(sentences),
                batch_size,
                desc='Encoding',
                disable=not show_progress_bar,
            )
        else:
            range_iter = range(0, len(sentences), batch_size)

        truncate_dim = truncate_dim or self.config.truncate_dim

        instruction = self.text_model.get_instruction_from_task(task)
        if instruction:
            sentences = [instruction + sentence for sentence in sentences]

        for i in range_iter:
            tokens = self.tokenizer(
                sentences[i: i + batch_size],
                return_tensors='pt',
                **tokenizer_kwargs,
            ).to(self.device)
            embeddings = self.get_text_features(input_ids=tokens)
            if truncate_dim:
                embeddings = self._truncate_embeddings(embeddings, truncate_dim)
            if normalize_embeddings:
                embeddings = f.normalize(embeddings, p=2, dim=1)
            if convert_to_numpy:
                embeddings = embeddings.cpu()
            all_embeddings.extend(embeddings)

        all_embeddings = [all_embeddings[idx] for idx in _inverse_permutation]

        if convert_to_tensor:
            all_embeddings = torch.stack(all_embeddings)
        elif convert_to_numpy:
            all_embeddings = np.asarray(
                [emb.to(torch.float32).numpy() for emb in all_embeddings]
            )
        if _input_was_string:
            all_embeddings = all_embeddings[0]

        self.train(_is_training)
        return all_embeddings

    def forward(
        self,
        input_ids: Union[None, torch.Tensor, BatchEncoding] = None,
        pixel_values: Union[None, torch.FloatTensor, BatchFeature] = None,
        return_dict: Optional[bool] = None,
        return_loss: Optional[bool] = None,
        *_,
        **__,
    ) -> Union[Tuple[Optional[torch.FloatTensor], ...], CLIPOutput]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        image_embeds = self.get_image_features(pixel_values=pixel_values)
        text_embeds = self.get_text_features(input_ids=input_ids)

        # normalized features
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.t()

        loss = None
        if return_loss:
            loss = clip_loss(logits_per_text)

        if not return_dict:
            output = (
                logits_per_image,
                logits_per_text,
                text_embeds,
                image_embeds,
                None,
                None,
            )
            return ((loss,) + output) if loss is not None else output

        return CLIPOutput(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=None,
            vision_model_output=None,
        )

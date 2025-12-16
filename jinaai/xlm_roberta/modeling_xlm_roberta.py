# This implementation was adopted from https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/models/bert.py
# Modified to remove flash_attn/triton dependency and use pure PyTorch implementation

# Commit id: abbc1311731867310635f9edc2a9ec18317c8c48
# Copyright (c) 2022, Tri Dao.
# This BERT implementation is based on our MLPerf 2.0 and MLPerf 2.1 BERT implementation.
# https://github.com/mlcommons/training_results_v2.0/blob/main/HazyResearch/benchmarks/bert/implementations/pytorch/modeling.py
# https://github.com/mlcommons/training_results_v2.1/blob/main/Azure-HazyResearch/benchmarks/bert/implementations/ND96amsr_A100_v4/modeling.py

# Inspired by https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py

import logging
import re
from collections import OrderedDict
from collections.abc import Sequence
from functools import partial
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import AutoTokenizer, PretrainedConfig
from transformers.modeling_outputs import MaskedLMOutput, SequenceClassifierOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.models.bert.modeling_bert import (
    BaseModelOutputWithPoolingAndCrossAttentions,
    BertForPreTrainingOutput,
)
from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaLMHead

from .rotary import RotaryEmbedding
from .block import Block
from .configuration_xlm_roberta import XLMRobertaFlashConfig
from .embedding import XLMRobertaEmbeddings
from .mha import MHA
from .mlp import FusedMLP, Mlp
from .xlm_padding import index_first_axis_residual, pad_input, unpad_input

# flash_attn dependencies removed - using pure PyTorch
FusedDense = None
layer_norm_fn = None

try:
    from tqdm.autonotebook import trange
except ImportError:
    trange = None


logger = logging.getLogger(__name__)


def get_use_flash_attn(config: XLMRobertaFlashConfig):
    # flash_attn dependency removed - always use PyTorch native attention
    return False


def create_mixer_cls(config, cross_attn=False, return_residual=False):
    use_flash_attn = get_use_flash_attn(config)
    fused_bias_fc = getattr(config, "fused_bias_fc", False)
    rotary_kwargs = {}
    if config.position_embedding_type == "rotary":
        rotary_kwargs["rotary_emb_dim"] = getattr(
            config, "rotary_emb_dim", config.hidden_size / config.num_attention_heads
        )
        rotary_kwargs["rotary_emb_base"] = config.rotary_emb_base
        rotary_kwargs["rotary_emb_scale_base"] = getattr(
            config, "rotary_emb_scale_base", None
        )
        rotary_kwargs["rotary_emb_interleaved"] = getattr(
            config, "rotary_emb_interleaved", False
        )
    mixer_cls = partial(
        MHA,
        num_heads=config.num_attention_heads,
        cross_attn=cross_attn,
        dropout=config.attention_probs_dropout_prob,
        causal=False,
        fused_bias_fc=fused_bias_fc,
        use_flash_attn=use_flash_attn,
        return_residual=return_residual,
        use_alibi=config.position_embedding_type == "alibi",
        **rotary_kwargs,
    )
    return mixer_cls


def create_mlp_cls(config, layer_idx=None, return_residual=False):
    inner_dim = config.intermediate_size
    fused_mlp = getattr(config, "fused_mlp", False)
    if fused_mlp:
        assert config.hidden_act in ["gelu_new", "gelu_fast", "gelu_pytorch_tanh"], (
            "fused_mlp only " "supports approximate gelu"
        )
    if not fused_mlp:
        approximate = (
            "tanh"
            if config.hidden_act in ["gelu_new", "gelu_fast", "gelu_pytorch_tanh"]
            else "none"
        )
        mlp_cls = partial(
            Mlp,
            hidden_features=inner_dim,
            activation=partial(F.gelu, approximate=approximate),
            return_residual=return_residual,
        )
    else:
        if FusedMLP is None:
            raise ImportError("fused_dense is not installed")
        mlp_checkpoint_lvl = getattr(config, "mlp_checkpoint_lvl", 0)
        # mlp_checkpoint_lvl could be a list, which contains the checkpoint_lvl for each layer
        if isinstance(mlp_checkpoint_lvl, Sequence):
            assert layer_idx is not None
            mlp_checkpoint_lvl = mlp_checkpoint_lvl[layer_idx]
        mlp_cls = partial(
            FusedMLP,
            hidden_features=inner_dim,
            checkpoint_lvl=mlp_checkpoint_lvl,
            return_residual=return_residual,
        )
    return mlp_cls


def create_block(config, layer_idx=None):
    last_layer_subset = getattr(config, "last_layer_subset", False)
    cross_attn = last_layer_subset and layer_idx == config.num_hidden_layers - 1
    # TD [2022-12-19]: For cross attention (last layer), we actually want to return the
    # residual x_kv, not residual x. But it's annoying to change the API (and it only affects
    # one layer) so we just choose not to return residual in this case.
    return_residual = not cross_attn
    mixer_cls = create_mixer_cls(config, cross_attn, return_residual=return_residual)
    mlp_cls = create_mlp_cls(config, layer_idx, return_residual=return_residual)
    norm_cls = partial(nn.LayerNorm, eps=config.layer_norm_eps)
    block = Block(
        config.hidden_size,
        mixer_cls,
        mlp_cls,
        norm_cls=norm_cls,
        prenorm=False,
        resid_dropout1=config.hidden_dropout_prob,
        resid_dropout2=config.hidden_dropout_prob,
        fused_dropout_add_ln=getattr(config, "fused_dropout_add_ln", False),
        return_residual=return_residual,
    )
    return block


# https://github.com/huggingface/transformers/blob/7032e0203262ebb2ebf55da8d2e01f873973e835/src/transformers/models/bert/modeling_bert.py#L748
def _init_weights(module, initializer_range=0.02):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, std=initializer_range)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)
        if module.padding_idx is not None:
            nn.init.zeros_(module.weight[module.padding_idx])


class XLMRobertaEncoder(nn.Module):
    def __init__(self, config: XLMRobertaFlashConfig):
        super().__init__()
        self.use_flash_attn = get_use_flash_attn(config)
        self.use_reentrant = config.use_reentrant
        self.layers = nn.ModuleList(
            [create_block(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )
        self._grad_checkpointing = False

    @property
    def gradient_checkpointing(self):
        return self._grad_checkpointing

    @gradient_checkpointing.setter
    def gradient_checkpointing(self, value):
        self._grad_checkpointing = value

    def forward(
        self,
        hidden_states,
        key_padding_mask=None,
        subset_mask=None,
        adapter_mask=None,
        output_hidden_states: Optional[bool] = None,
    ):
        """If subset_mask is not None, we only want output for the subset of the sequence.
        This means that we only compute the last layer output for these tokens.
        subset_mask: (batch, seqlen), dtype=torch.bool
        """

        all_hidden_states = () if output_hidden_states else None

        if output_hidden_states and subset_mask:
            raise ValueError('output_hidden_states is not supported for subset_masks')

        if key_padding_mask is None or not self.use_flash_attn:
            mixer_kwargs = {"adapter_mask": adapter_mask}
            if key_padding_mask is not None:
                mixer_kwargs["key_padding_mask"] = key_padding_mask.bool()
            for layer in self.layers:
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)
                if self._grad_checkpointing:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        layer,
                        hidden_states,
                        use_reentrant=self.use_reentrant,
                        mixer_kwargs=mixer_kwargs,
                    )
                else:
                    hidden_states = layer(hidden_states, mixer_kwargs=mixer_kwargs)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            if subset_mask is not None:
                hidden_states = hidden_states[subset_mask]
        else:
            batch, seqlen = hidden_states.shape[:2]
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            hidden_states, indices, cu_seqlens, max_seqlen_in_batch, cu_adapter_mask = (
                unpad_input(hidden_states, key_padding_mask, adapter_mask)
            )
            mixer_kwargs = {
                "cu_seqlens": cu_seqlens,
                "max_seqlen": max_seqlen_in_batch,
                "adapter_mask": cu_adapter_mask,
            }

            if subset_mask is None:
                for layer in self.layers:
                    if self._grad_checkpointing:
                        hidden_states = torch.utils.checkpoint.checkpoint(
                            layer,
                            hidden_states,
                            use_reentrant=self.use_reentrant,
                            mixer_kwargs=mixer_kwargs,
                        )
                    else:
                        hidden_states = layer(hidden_states, mixer_kwargs=mixer_kwargs)
                    if output_hidden_states:
                        all_hidden_states = all_hidden_states + (
                            pad_input(hidden_states, indices, batch, seqlen),
                        )
                hidden_states = pad_input(hidden_states, indices, batch, seqlen)
            else:
                for layer in self.layers[:-1]:
                    if self._grad_checkpointing:
                        hidden_states = torch.utils.checkpoint.checkpoint(
                            layer,
                            hidden_states,
                            use_reentrant=self.use_reentrant,
                            mixer_kwargs=mixer_kwargs,
                        )
                    else:
                        hidden_states = layer(hidden_states, mixer_kwargs=mixer_kwargs)
                if key_padding_mask is not None:
                    subset_idx = torch.nonzero(
                        subset_mask[key_padding_mask], as_tuple=False
                    ).flatten()
                    subset_seqlens = (subset_mask & key_padding_mask).sum(
                        dim=-1, dtype=torch.int32
                    )
                    subset_cu_seqlens = F.pad(
                        torch.cumsum(subset_seqlens, dim=0, dtype=torch.torch.int32),
                        (1, 0),
                    )
                else:
                    subset_idx = torch.nonzero(subset_mask, as_tuple=False).flatten()
                    subset_seqlens = subset_mask.sum(dim=-1, dtype=torch.int32)
                    subset_cu_seqlens = F.pad(
                        torch.cumsum(subset_seqlens, dim=0, dtype=torch.torch.int32),
                        (1, 0),
                    )
                hidden_states_subset, hidden_states = index_first_axis_residual(
                    hidden_states, subset_idx
                )
                # It's ok to set max_seqlen_q to be much larger
                mixer_kwargs = {
                    "x_kv": hidden_states,
                    "cu_seqlens": subset_cu_seqlens,
                    "max_seqlen": max_seqlen_in_batch,
                    "cu_seqlens_k": cu_seqlens,
                    "max_seqlen_k": max_seqlen_in_batch,
                }
                if self._grad_checkpointing:
                    torch.utils.checkpoint.checkpoint(
                        self.layers[-1],
                        hidden_states_subset,
                        use_reentrant=self.use_reentrant,
                        mixer_kwargs=mixer_kwargs,
                    )
                else:
                    hidden_states = self.layers[-1](
                        hidden_states_subset, mixer_kwargs=mixer_kwargs
                    )
        return all_hidden_states if output_hidden_states else hidden_states


class XLMRobertaPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Always use standard nn.Linear (fused_bias_fc removed)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states, pool=True, adapter_mask=None):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0] if pool else hidden_states
        if adapter_mask is not None:
            unique_tasks = torch.unique(adapter_mask)
            pool_dtype = next(self.dense.parameters()).dtype
            pooled_output = torch.empty(
                first_token_tensor.shape[0],
                self.dense.out_features,
                dtype=pool_dtype,
                device=first_token_tensor.device,
            )
            for task_id in unique_tasks:
                task_indices = (adapter_mask == task_id).nonzero(as_tuple=True)[0]
                task_first_token_tensor = first_token_tensor[task_indices]
                task_pooled_output = self.dense(
                    task_first_token_tensor, task_id=task_id
                )
                pooled_output[task_indices] = task_pooled_output
        else:
            pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class XLMRobertaPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Always use standard nn.Linear (fused_bias_fc removed)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        approximate = (
            "tanh"
            if config.hidden_act in ["gelu_new", "gelu_fast", "gelu_pytorch_tanh"]
            else "none"
        )
        self.transform_act_fn = nn.GELU(approximate=approximate)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


class XLMRobertaLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = XLMRobertaPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=True)

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class XLMRobertaPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = XLMRobertaLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class XLMRobertaPreTrainedModel(PreTrainedModel):
    """An abstract class to handle weights initialization and
    a simple interface for dowloading and loading pretrained models.
    """

    config_class = XLMRobertaFlashConfig
    base_model_prefix = "roberta"
    supports_gradient_checkpointing = True
    _supports_param_buffer_assignment = False

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, XLMRobertaEncoder):
            module.gradient_checkpointing = value

    @classmethod
    def from_pretrained(
        cls,
        *args,
        **kwargs,
    ):
        if not "torch_dtype" in kwargs:
            kwargs["torch_dtype"] = "auto"
        return super().from_pretrained(*args, **kwargs)


class XLMRobertaModel(XLMRobertaPreTrainedModel):
    def __init__(self, config: XLMRobertaFlashConfig, add_pooling_layer=True):
        super().__init__(config)
        self.pad_vocab_size_multiple = getattr(config, "pad_vocab_size_multiple", 1)
        if config.vocab_size % self.pad_vocab_size_multiple != 0:
            config.vocab_size += self.pad_vocab_size_multiple - (
                config.vocab_size % self.pad_vocab_size_multiple
            )
        # fused_dropout_add_ln removed - always use standard implementation
        self.fused_dropout_add_ln = False
        assert config.hidden_act in [
            "gelu",
            "gelu_new",
            "gelu_fast",
            "gelu_pytorch_tanh",
        ]
        self.embeddings = XLMRobertaEmbeddings(
            config.hidden_size,
            config.vocab_size,
            (
                config.max_position_embeddings
                if config.position_embedding_type == "absolute"
                else -1
            ),
            config.type_vocab_size,
            padding_idx=config.pad_token_id,
        )
        self.emb_drop = nn.Dropout(config.hidden_dropout_prob)
        self.emb_ln = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.encoder = XLMRobertaEncoder(config)
        self.pooler = XLMRobertaPooler(config) if add_pooling_layer else None

        self.apply(partial(_init_weights, initializer_range=config.initializer_range))
        # Only load tokenizer if name_or_path is valid (not empty)
        # Tokenizer is only needed for the encode() method, not for forward()
        if self.name_or_path and self.name_or_path.strip():
            try:
                # Use local_files_only if path is a local directory
                import os
                is_local = os.path.isdir(self.name_or_path)
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.name_or_path,
                    trust_remote_code=True,
                    local_files_only=is_local,
                )
            except Exception as e:
                print(f"[XLMRobertaModel] Could not load tokenizer: {e}")
                self.tokenizer = None
        else:
            self.tokenizer = None
        self._rotary_emb_base = config.rotary_emb_base

    @torch.inference_mode()
    def encode(
        self: "XLMRobertaModel",
        sentences: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: Optional[bool] = None,
        output_value: str = "sentence_embedding",
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        device: Optional[torch.device] = None,
        normalize_embeddings: bool = True,
        truncate_dim: Optional[int] = None,
        adapter_mask: Optional[torch.Tensor] = None,
        task: Optional[str] = None,
        **tokenizer_kwargs,
    ) -> Union[List[torch.Tensor], np.ndarray, torch.Tensor]:
        """
        Computes sentence embeddings
        Args:
            sentences(`str` or `List[str]`):
                Sentence or sentences to be encoded
            batch_size(`int`, *optional*, defaults to 32):
                Batch size for the computation
            show_progress_bar(`bool`, *optional*, defaults to None):
                Show a progress bar when encoding sentences.
                If set to None, progress bar is only shown when
                `logger.level == logging.INFO` or `logger.level == logging.DEBUG`.
            output_value(`str`, *optional*, defaults to 'sentence_embedding'):
                Default sentence_embedding, to get sentence embeddings.
                Can be set to token_embeddings to get wordpiece token embeddings.
                Set to None, to get all output values
            convert_to_numpy(`bool`, *optional*, defaults to True):
                If true, the output is a list of numpy vectors.
                Else, it is a list of pytorch tensors.
            convert_to_tensor(`bool`, *optional*, defaults to False):
                If true, you get one large tensor as return.
                Overwrites any setting from convert_to_numpy
            device(`torch.device`, *optional*, defaults to None):
                Which torch.device to use for the computation
            normalize_embeddings(`bool`, *optional*, defaults to True):
                If set to true, returned vectors will have length 1. In that case, the
                faster dot-product (util.dot_score) instead of cosine similarity can
                be used.
            truncate_dim(`int`, *optional*, defaults to None):
                The dimension to truncate sentence embeddings to. `None` does no truncation.
            tokenizer_kwargs(`Dict[str, Any]`, *optional*, defaults to {}):
                Keyword arguments for the tokenizer
        Returns:
            By default, a list of tensors is returned.
            If convert_to_tensor, a stacked tensor is returned.
            If convert_to_numpy, a numpy matrix is returned.
        """
        is_training = self.training
        self.eval()

        if show_progress_bar is None:
            show_progress_bar = (
                logger.getEffectiveLevel() == logging.INFO
                or logger.getEffectiveLevel() == logging.DEBUG
            )

        if convert_to_tensor:
            convert_to_numpy = False

        if output_value != "sentence_embedding":
            convert_to_tensor = False
            convert_to_numpy = False

        input_was_string = False
        if isinstance(sentences, str) or not hasattr(sentences, "__len__"):
            sentences = [sentences]
            input_was_string = True

        if device is not None:
            self.to(device)

        permutation = np.argsort([-len(i) for i in sentences])
        inverse_permutation = np.argsort(permutation)
        sentences = [sentences[idx] for idx in permutation]

        tokenizer_kwargs["padding"] = tokenizer_kwargs.get("padding", True)
        tokenizer_kwargs["max_length"] = tokenizer_kwargs.get(
            "max_length", self.tokenizer.init_kwargs.get("model_max_length", 8192)
        )
        tokenizer_kwargs["truncation"] = tokenizer_kwargs.get("truncation", True)

        all_embeddings = []

        if trange is not None:
            range_iter = trange(
                0,
                len(sentences),
                batch_size,
                desc="Encoding",
                disable=not show_progress_bar,
            )
        else:
            range_iter = range(0, len(sentences), batch_size)

        for i in range_iter:
            encoded_input = self.tokenizer(
                sentences[i : i + batch_size],
                return_tensors="pt",
                **tokenizer_kwargs,
            ).to(self.device)
            lora_arguments = (
                {"adapter_mask": adapter_mask[i : i + batch_size]}
                if adapter_mask is not None
                else {}
            )
            token_embs = self.forward(**encoded_input, **lora_arguments)[0]

            # Accumulate in fp32 to avoid overflow
            token_embs = token_embs.float()

            if output_value == "token_embeddings":
                raise NotImplementedError
            elif output_value is None:
                raise NotImplementedError
            else:
                if self.config.emb_pooler == "cls":
                    embeddings = self.cls_pooling(
                        token_embs, encoded_input["attention_mask"]
                    )
                else:
                    embeddings = self.mean_pooling(
                        token_embs, encoded_input["attention_mask"]
                    )

            all_embeddings.extend(embeddings)

        all_embeddings = [all_embeddings[idx] for idx in inverse_permutation]

        truncate_dim = truncate_dim or self.config.truncate_dim
        if truncate_dim:
            all_embeddings = self.truncate_embeddings(all_embeddings, truncate_dim)

        if normalize_embeddings:
            all_embeddings = [
                torch.nn.functional.normalize(embedding, p=2, dim=0)
                for embedding in all_embeddings
            ]

        if convert_to_tensor:
            all_embeddings = torch.stack(all_embeddings)
        elif convert_to_numpy:
            all_embeddings = np.asarray([emb.cpu().numpy() for emb in all_embeddings])

        if input_was_string:
            all_embeddings = all_embeddings[0]

        self.train(is_training)
        return all_embeddings

    def truncate_embeddings(self, embeddings, truncate_dim):
        if not self.config.matryoshka_dimensions:
            logger.warning(
                "Matryoshka embeddings are not supported, so dimension truncation will not be performed."
            )
            return embeddings
        elif truncate_dim in self.config.matryoshka_dimensions:
            return [tensor[:truncate_dim] for tensor in embeddings]
        else:
            raise ValueError(
                f"The provided `truncate_dim` value of {truncate_dim} is not supported. "
                f"Supported dimensions are {self.config.matryoshka_dimensions}."
            )

    def mean_pooling(
        self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor
    ):
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def cls_pooling(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor):
        return token_embeddings[:, 0]

    @property
    def rotary_emb_base(self):
        return self._rotary_emb_base

    @rotary_emb_base.setter
    def rotary_emb_base(self, base):
        if not isinstance(base, (int, float)):
            raise TypeError("Base must be an integer or float")
        logger.info(f"Changing RoPE base value to {base}")
        for layer in self.encoder.layers:
            layer.mixer.rotary_emb.base = base
        self._rotary_emb_base = base

    def forward(
        self,
        input_ids,
        position_ids=None,
        token_type_ids=None,
        attention_mask=None,
        masked_tokens_mask=None,
        return_dict=None,
        output_hidden_states=None,
        **kwargs,
    ):
        """If masked_tokens_mask is not None (i.e. last_layer_subset == True in XLMForPreTraining),
        we only want the output for the masked tokens. This means that we only compute the last
        layer output for these tokens.
        masked_tokens_mask: (batch, seqlen), dtype=torch.bool
        """
        adapter_mask = kwargs.pop("adapter_mask", None)
        if kwargs:
            for key, value in kwargs.items():
                if value is not None:
                    logger.warning(
                        "Flash attention implementation does not support kwargs: %s",
                        key,
                    )

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        hidden_states = self.embeddings(
            input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            adapter_mask=adapter_mask,
        )
        # TD [2022-12:18]: Don't need to force residual in fp32
        # BERT puts embedding LayerNorm before embedding dropout.
        hidden_states = self.emb_ln(hidden_states)
        hidden_states = self.emb_drop(hidden_states)

        if masked_tokens_mask is not None:
            batch_size, seqlen = input_ids.shape[:2]
            # We also need the first column for the CLS token
            first_col_mask = torch.zeros(
                batch_size, seqlen, dtype=torch.bool, device=input_ids.device
            )
            first_col_mask[:, 0] = True
            subset_mask = masked_tokens_mask | first_col_mask
        else:
            subset_mask = None

        sequence_output = self.encoder(
            hidden_states,
            key_padding_mask=attention_mask,
            subset_mask=subset_mask,
            adapter_mask=adapter_mask,
            output_hidden_states=output_hidden_states,
        )

        if output_hidden_states:
            all_hidden_states = sequence_output
            sequence_output = sequence_output[-1]
        else:
            all_hidden_states = None

        if masked_tokens_mask is None:
            pooled_output = (
                self.pooler(sequence_output, adapter_mask=adapter_mask)
                if self.pooler is not None
                else None
            )
        else:
            # TD [2022-03-01]: the indexing here is very tricky.
            if attention_mask is not None:
                subset_idx = subset_mask[attention_mask]
                pool_input = sequence_output[first_col_mask[attention_mask][subset_idx]]
                sequence_output = sequence_output[
                    masked_tokens_mask[attention_mask][subset_idx]
                ]
            else:
                pool_input = sequence_output[first_col_mask[subset_mask]]
                sequence_output = sequence_output[masked_tokens_mask[subset_mask]]
            pooled_output = (
                self.pooler(pool_input, pool=False, adapter_mask=adapter_mask)
                if self.pooler is not None
                else None
            )

        if not return_dict:
            return sequence_output, pooled_output

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=all_hidden_states,
        )


class XLMRobertaForMaskedLM(XLMRobertaPreTrainedModel):
    _tied_weights_keys = ["lm_head.decoder.weight", "lm_head.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `XLMRobertaForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.roberta = XLMRobertaModel(config, add_pooling_layer=False)
        self.lm_head = XLMRobertaLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.roberta.embeddings.word_embeddings

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
            Used to hide legacy arguments that have been deprecated.
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(prediction_scores.device)
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size), labels.view(-1)
            )

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return (
                ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
            )

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def remap_state_dict(state_dict, config: PretrainedConfig):
    """
    Map the state_dict of a Huggingface BERT model to be flash_attn compatible.
    """

    # LayerNorm
    def key_mapping_ln_gamma_beta(key):
        key = re.sub(r"LayerNorm.gamma$", "LayerNorm.weight", key)
        key = re.sub(r"LayerNorm.beta$", "LayerNorm.bias", key)
        return key

    state_dict = OrderedDict(
        (key_mapping_ln_gamma_beta(k), v) for k, v in state_dict.items()
    )

    # Layers
    def key_mapping_layers(key):
        return re.sub(r"^bert.encoder.layer.", "bert.encoder.layers.", key)

    state_dict = OrderedDict((key_mapping_layers(k), v) for k, v in state_dict.items())

    # LayerNorm
    def key_mapping_ln(key):
        key = re.sub(r"^bert.embeddings.LayerNorm.", "bert.emb_ln.", key)
        key = re.sub(
            r"^bert.encoder.layers.(\d+).attention.output.LayerNorm.(weight|bias)",
            r"bert.encoder.layers.\1.norm1.\2",
            key,
        )
        key = re.sub(
            r"^bert.encoder.layers.(\d+).output.LayerNorm.(weight|bias)",
            r"bert.encoder.layers.\1.norm2.\2",
            key,
        )
        key = re.sub(
            r"^cls.predictions.transform.LayerNorm.(weight|bias)",
            r"cls.predictions.transform.layer_norm.\1",
            key,
        )
        return key

    state_dict = OrderedDict((key_mapping_ln(k), v) for k, v in state_dict.items())

    # MLP
    def key_mapping_mlp(key):
        key = re.sub(
            r"^bert.encoder.layers.(\d+).intermediate.dense.(weight|bias)",
            r"bert.encoder.layers.\1.mlp.fc1.\2",
            key,
        )
        key = re.sub(
            r"^bert.encoder.layers.(\d+).output.dense.(weight|bias)",
            r"bert.encoder.layers.\1.mlp.fc2.\2",
            key,
        )
        return key

    state_dict = OrderedDict((key_mapping_mlp(k), v) for k, v in state_dict.items())

    # Attention
    last_layer_subset = getattr(config, "last_layer_subset", False)
    for d in range(config.num_hidden_layers):
        Wq = state_dict.pop(f"bert.encoder.layers.{d}.attention.self.query.weight")
        Wk = state_dict.pop(f"bert.encoder.layers.{d}.attention.self.key.weight")
        Wv = state_dict.pop(f"bert.encoder.layers.{d}.attention.self.value.weight")
        bq = state_dict.pop(f"bert.encoder.layers.{d}.attention.self.query.bias")
        bk = state_dict.pop(f"bert.encoder.layers.{d}.attention.self.key.bias")
        bv = state_dict.pop(f"bert.encoder.layers.{d}.attention.self.value.bias")
        if not (last_layer_subset and d == config.num_hidden_layers - 1):
            state_dict[f"bert.encoder.layers.{d}.mixer.Wqkv.weight"] = torch.cat(
                [Wq, Wk, Wv], dim=0
            )
            state_dict[f"bert.encoder.layers.{d}.mixer.Wqkv.bias"] = torch.cat(
                [bq, bk, bv], dim=0
            )
        else:
            state_dict[f"bert.encoder.layers.{d}.mixer.Wq.weight"] = Wq
            state_dict[f"bert.encoder.layers.{d}.mixer.Wkv.weight"] = torch.cat(
                [Wk, Wv], dim=0
            )
            state_dict[f"bert.encoder.layers.{d}.mixer.Wq.bias"] = bq
            state_dict[f"bert.encoder.layers.{d}.mixer.Wkv.bias"] = torch.cat(
                [bk, bv], dim=0
            )

    def key_mapping_attn(key):
        return re.sub(
            r"^bert.encoder.layers.(\d+).attention.output.dense.(weight|bias)",
            r"bert.encoder.layers.\1.mixer.out_proj.\2",
            key,
        )

    state_dict = OrderedDict((key_mapping_attn(k), v) for k, v in state_dict.items())

    def key_mapping_decoder_bias(key):
        return re.sub(r"^cls.predictions.bias", "cls.predictions.decoder.bias", key)

    state_dict = OrderedDict(
        (key_mapping_decoder_bias(k), v) for k, v in state_dict.items()
    )

    # Word embedding
    pad_vocab_size_multiple = getattr(config, "pad_vocab_size_multiple", 1)
    if pad_vocab_size_multiple > 1:
        word_embeddings = state_dict["bert.embeddings.word_embeddings.weight"]
        state_dict["bert.embeddings.word_embeddings.weight"] = F.pad(
            word_embeddings, (0, 0, 0, config.vocab_size - word_embeddings.shape[0])
        )
        decoder_weight = state_dict["cls.predictions.decoder.weight"]
        state_dict["cls.predictions.decoder.weight"] = F.pad(
            decoder_weight, (0, 0, 0, config.vocab_size - decoder_weight.shape[0])
        )
        # If the vocab was padded, we want to set the decoder bias for those padded indices to be
        # strongly negative (i.e. the decoder shouldn't predict those indices).
        # TD [2022-05-09]: I don't think it affects the MLPerf training.
        decoder_bias = state_dict["cls.predictions.decoder.bias"]
        state_dict["cls.predictions.decoder.bias"] = F.pad(
            decoder_bias, (0, config.vocab_size - decoder_bias.shape[0]), value=-100.0
        )

    return state_dict


def inv_remap_state_dict(state_dict, config: PretrainedConfig):
    """
    Map the state_dict of a flash_attn model to be Huggingface BERT compatible.

    This function is meant to be the inverse of remap_state_dict.
    """
    # Word embedding
    pad_vocab_size_multiple = getattr(config, "pad_vocab_size_multiple", 1)
    if pad_vocab_size_multiple > 1:
        word_embeddings = state_dict["bert.embeddings.word_embeddings.weight"]
        decoder_weight = state_dict["cls.predictions.decoder.weight"]
        decoder_bias = state_dict["cls.predictions.decoder.bias"]
        # unpad embeddings
        state_dict["bert.embeddings.word_embeddings.weight"] = word_embeddings[
            : config.orig_vocab_size, :
        ]
        state_dict["cls.predictions.decoder.weight"] = decoder_weight[
            : config.orig_vocab_size, :
        ]
        state_dict["cls.predictions.decoder.bias"] = decoder_bias[
            : config.orig_vocab_size
        ]

    for d in range(config.num_hidden_layers):
        last_layer_subset = getattr(config, "last_layer_subset", False)
        if not last_layer_subset or d != (config.num_hidden_layers - 1):
            Wqkv_weights = state_dict.pop(f"bert.encoder.layers.{d}.mixer.Wqkv.weight")
            Wqkv_biases = state_dict.pop(f"bert.encoder.layers.{d}.mixer.Wqkv.bias")
            state_dict[f"bert.encoder.layers.{d}.attention.self.query.weight"] = (
                Wqkv_weights[: Wqkv_weights.shape[0] // 3, :]
            )
            state_dict[f"bert.encoder.layers.{d}.attention.self.key.weight"] = (
                Wqkv_weights[
                    Wqkv_weights.shape[0] // 3 : 2 * Wqkv_weights.shape[0] // 3, :
                ]
            )
            state_dict[f"bert.encoder.layers.{d}.attention.self.value.weight"] = (
                Wqkv_weights[2 * Wqkv_weights.shape[0] // 3 :, :]
            )
            state_dict[f"bert.encoder.layers.{d}.attention.self.query.bias"] = (
                Wqkv_biases[: Wqkv_biases.shape[0] // 3]
            )
            state_dict[f"bert.encoder.layers.{d}.attention.self.key.bias"] = (
                Wqkv_biases[Wqkv_biases.shape[0] // 3 : 2 * Wqkv_biases.shape[0] // 3]
            )
            state_dict[f"bert.encoder.layers.{d}.attention.self.value.bias"] = (
                Wqkv_biases[2 * Wqkv_biases.shape[0] // 3 :]
            )
        else:
            Wq_weight = state_dict.pop(f"bert.encoder.layers.{d}.mixer.Wq.weight")
            Wkv_weights = state_dict.pop(f"bert.encoder.layers.{d}.mixer.Wkv.weight")
            Wq_bias = state_dict.pop(f"bert.encoder.layers.{d}.mixer.Wq.bias")
            Wkv_biases = state_dict.pop(f"bert.encoder.layers.{d}.mixer.Wkv.bias")
            state_dict[f"bert.encoder.layers.{d}.attention.self.query.weight"] = (
                Wq_weight
            )
            state_dict[f"bert.encoder.layers.{d}.attention.self.key.weight"] = (
                Wkv_weights[: Wkv_weights.shape[0] // 2, :]
            )
            state_dict[f"bert.encoder.layers.{d}.attention.self.value.weight"] = (
                Wkv_weights[Wkv_weights.shape[0] // 2 :, :]
            )
            state_dict[f"bert.encoder.layers.{d}.attention.self.query.bias"] = Wq_bias
            state_dict[f"bert.encoder.layers.{d}.attention.self.key.bias"] = Wkv_biases[
                : Wkv_biases.shape[0] // 2
            ]
            state_dict[f"bert.encoder.layers.{d}.attention.self.value.bias"] = (
                Wkv_biases[Wkv_biases.shape[0] // 2 :]
            )

    def inv_key_mapping_ln(key):
        key = re.sub(r"bert.emb_ln.", "bert.embeddings.LayerNorm.", key)
        key = re.sub(
            r"bert.encoder.layers.(\d+).norm1.(weight|bias)",
            r"bert.encoder.layers.\1.attention.output.LayerNorm.\2",
            key,
        )
        key = re.sub(
            r"bert.encoder.layers.(\d+).norm2.(weight|bias)",
            r"bert.encoder.layers.\1.output.LayerNorm.\2",
            key,
        )
        key = re.sub(
            r"cls.predictions.transform.layer_norm.(weight|bias)",
            r"cls.predictions.transform.LayerNorm.\1",
            key,
        )
        return key

    def inv_key_mapping_ln_gamma_beta(key):
        key = re.sub(r"LayerNorm.weight$", "LayerNorm.gamma", key)
        key = re.sub(r"LayerNorm.bias$", "LayerNorm.beta", key)
        return key

    def inv_key_mapping_layers(key):
        return re.sub(r"bert.encoder.layers.", "bert.encoder.layer.", key)

    def inv_key_mapping_mlp(key):
        key = re.sub(
            r"bert.encoder.layer.(\d+).mlp.fc1.(weight|bias)",
            r"bert.encoder.layer.\1.intermediate.dense.\2",
            key,
        )
        key = re.sub(
            r"bert.encoder.layer.(\d+).mlp.fc2.(weight|bias)",
            r"bert.encoder.layer.\1.output.dense.\2",
            key,
        )
        return key

    def inv_key_mapping_attn(key):
        return re.sub(
            r"bert.encoder.layer.(\d+).mixer.out_proj.(weight|bias)",
            r"bert.encoder.layer.\1.attention.output.dense.\2",
            key,
        )

    def inv_key_mapping_decoder_bias(key):
        return re.sub(r"cls.predictions.decoder.bias", "cls.predictions.bias", key)

    state_dict = OrderedDict(
        (inv_key_mapping_ln(key), value) for key, value in state_dict.items()
    )
    state_dict = OrderedDict(
        (inv_key_mapping_ln_gamma_beta(key), value) for key, value in state_dict.items()
    )
    state_dict = OrderedDict(
        (inv_key_mapping_layers(key), value) for key, value in state_dict.items()
    )
    state_dict = OrderedDict(
        (inv_key_mapping_mlp(key), value) for key, value in state_dict.items()
    )
    state_dict = OrderedDict(
        (inv_key_mapping_attn(key), value) for key, value in state_dict.items()
    )
    state_dict = OrderedDict(
        (inv_key_mapping_decoder_bias(key), value) for key, value in state_dict.items()
    )

    return state_dict


# Copied from transformers.models.roberta.modeling_roberta.RobertaClassificationHead with Roberta->XLMRoberta
class XLMRobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        # Always use standard nn.Linear (fused_bias_fc removed)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


# Copied from transformers.models.roberta.modeling_roberta.RobertaForSequenceClassification with Roberta->XLMRoberta, ROBERTA->XLM_ROBERTA
class XLMRobertaForSequenceClassification(XLMRobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = XLMRobertaModel(config, add_pooling_layer=False)
        self.classifier = XLMRobertaClassificationHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

import math
import os
from functools import partial
from typing import Iterator, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.utils.parametrize as parametrize
from torch import nn
from torch.nn import Parameter
from torch.nn import functional as F
from transformers import PretrainedConfig

from .configuration_xlm_roberta import XLMRobertaFlashConfig
from .modeling_xlm_roberta import (
    XLMRobertaFlashConfig,
    XLMRobertaModel,
    XLMRobertaPreTrainedModel,
)


def initialized_weights(
    shape: Tuple[int], num_adaptations: int, init: str = "kaiming"
) -> torch.Tensor:
    weight_data = []
    for _ in range(num_adaptations):
        new_adaption = torch.zeros(shape)
        if init == "kaiming":
            nn.init.kaiming_uniform_(new_adaption, a=math.sqrt(5))
        elif init == "normal":
            nn.init.normal_(new_adaption)
        else:
            raise NotImplementedError
        weight_data.append(new_adaption)
    return torch.stack(weight_data, dim=0)


class LoRAParametrization(nn.Module):
    """
    This LoRA implementation was inspired by  https://github.com/cccntu/minLoRA
    The MIT License (MIT) Copyright (c) 2020 Andrej Karpathy
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software
    and associated documentation files (the "Software"), to deal in the Software without restriction,
    including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
    and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
    subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all copies or substantial
    portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
    LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
    IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
    WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
    SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
    """

    def __init__(
        self,
        fan_in: int,
        fan_out: int,
        layer_type: str = "linear",
        num_adaptations: int = 1,
        rank: int = 4,
        dropout_p: float = 0.0,
        alpha: float = 1,
    ):
        super().__init__()
        # if weight is stored as (fan_out, fan_in), the memory layout of A & B follows (W + BA)x
        # otherwise, it's x(W + AB). This allows us to tie the weights between linear layers and embeddings
        fan_in_fan_out = layer_type == "embedding"
        self.swap = (lambda x: (x[1], x[0])) if fan_in_fan_out else (lambda x: x)

        if layer_type == "linear":
            self.lora_A = nn.Parameter(
                initialized_weights((rank, fan_in), num_adaptations, init="kaiming")
            )
            self.lora_B = nn.Parameter(torch.zeros((num_adaptations, fan_out, rank)))
        elif layer_type == "embedding":
            self.lora_A = nn.Parameter(torch.zeros((num_adaptations, fan_in, rank)))
            self.lora_B = nn.Parameter(
                initialized_weights(
                    (rank, fan_out), num_adaptations=num_adaptations, init="normal"
                )
            )
        else:
            raise NotImplementedError

        self.lora_alpha, self.rank = alpha, rank
        self.scaling = alpha / rank
        self.lora_dropout = nn.Dropout(p=dropout_p) if dropout_p > 0 else lambda x: x
        self.dropout_fn = self._dropout if dropout_p > 0 else lambda x: x
        self.register_buffer(
            "lora_dropout_mask",
            torch.ones(self.swap((1, fan_in)), dtype=self.lora_A.dtype),
            persistent=False,
        )

    def _dropout(self, A):
        # to mimic the original implementation: A @ dropout(x), we do (A * dropout(ones)) @ x
        return A * self.lora_dropout(self.lora_dropout_mask)

    def lora_forward(self, X, current_task):
        return (
            X
            + torch.matmul(
                *self.swap(
                    (
                        self.lora_B[current_task],
                        self.dropout_fn(self.lora_A[current_task]),
                    )
                )
            ).view(X.shape)
            * self.scaling
        )

    def forward(self, X):
        return X

    @classmethod
    def from_linear(
        cls,
        layer: nn.Module,
        num_adaptations: int,
        rank: int,
        dropout_p: float,
        alpha: float,
    ):
        assert isinstance(layer, nn.Linear)
        fan_out, fan_in = layer.weight.shape
        return cls(
            fan_in,
            fan_out,
            num_adaptations=num_adaptations,
            layer_type="linear",
            rank=rank,
            dropout_p=dropout_p,
            alpha=alpha,
        )

    @classmethod
    def from_embedding(
        cls,
        layer: nn.Module,
        num_adaptations: int,
        rank: int,
        dropout_p: float,
        alpha: float,
    ):
        assert isinstance(layer, nn.Embedding)
        fan_in, fan_out = layer.weight.shape
        return cls(
            fan_in,
            fan_out,
            num_adaptations=num_adaptations,
            layer_type="embedding",
            rank=rank,
            dropout_p=dropout_p,
            alpha=alpha,
        )

    @classmethod
    def add_to_layer(
        cls,
        layer: nn.Module,
        num_adaptations: int,
        rank: int,
        dropout_p: float,
        alpha: float,
    ):
        """
        Registering LoRA adapters to all embedding and linear layers.
        Additionally, we implement a custom forward function for LoRA parametrization.
        This function modifies the layer's forward pass to optionally use task-specific
        parameters. When a `task_id` is provided, it employs a LoRA parametrization
        to modify the original weights according to the specific task. This allows
        the layer to adapt dynamically to different tasks at runtime. If no `task_id`
        is specified, the layer uses its original weights.
        """
        if isinstance(layer, nn.Linear):
            parametrize.register_parametrization(
                layer,
                "weight",
                cls.from_linear(
                    layer,
                    num_adaptations=num_adaptations,
                    rank=rank,
                    dropout_p=dropout_p,
                    alpha=alpha,
                ),
            )

            def new_forward(self, input, task_id=None, residual=False):
                if task_id is not None:
                    weights = self.parametrizations.weight[0].lora_forward(
                        self.weight, current_task=task_id
                    )
                else:
                    weights = self.weight

                out = F.linear(input, weights, self.bias)

                if residual:
                    return out, input
                return out

            layer.forward = new_forward.__get__(layer, layer.__class__)

        elif isinstance(layer, nn.Embedding):
            parametrize.register_parametrization(
                layer,
                "weight",
                cls.from_embedding(
                    layer,
                    num_adaptations=num_adaptations,
                    rank=rank,
                    dropout_p=dropout_p,
                    alpha=alpha,
                ),
            )

            def new_forward(self, input, task_id=None):
                if task_id is not None:
                    weights = self.parametrizations.weight[0].lora_forward(
                        self.weight, current_task=task_id
                    )
                else:
                    weights = self.weight

                out = F.embedding(
                    input,
                    weights,
                    self.padding_idx,
                    self.max_norm,
                    self.norm_type,
                    self.scale_grad_by_freq,
                    self.sparse,
                )

                return out

            layer.forward = new_forward.__get__(layer, layer.__class__)


class XLMRobertaLoRA(XLMRobertaPreTrainedModel):
    """
    A wrapper class around the Jina XLM-RoBERTa model that integrates LoRA (Low-Rank Adaptation) adapters.
    """

    def __init__(
        self,
        config: XLMRobertaFlashConfig,
        roberta: Optional[XLMRobertaModel] = None,
        add_pooling_layer: bool = True,
    ):
        super().__init__(config)
        if roberta is None:
            self.roberta = XLMRobertaModel(config, add_pooling_layer=add_pooling_layer)
        else:
            self.roberta = roberta

        self._lora_adaptations = config.lora_adaptations
        if (
            not isinstance(self._lora_adaptations, list)
            or len(self._lora_adaptations) < 1
        ):
            raise ValueError(
                f"`lora_adaptations` must be a list and contain at least one element"
            )
        self._task_instructions = config.task_instructions
        if (
            not isinstance(self._task_instructions, dict)
            or len(self._task_instructions) != len(self._lora_adaptations)
            or not all(
                [v in self._lora_adaptations for v in self._task_instructions.keys()]
            )
        ):
            raise ValueError(
                f"`task_instructions` must be a dict and contain the same number of elements "
                f"as `lora_adaptations` with all keys in `task_instructions` present in `lora_adaptations`."
            )
        self._adaptation_map = {
            name: idx for idx, name in enumerate(self._lora_adaptations)
        }
        self._rank = config.lora_rank
        self._dropout_p = config.lora_dropout_p
        self._alpha = config.lora_alpha
        self._register_lora(
            num_adaptations=len(self._lora_adaptations),
            rank=self._rank,
            dropout_p=self._dropout_p,
            alpha=self._alpha,
        )
        self.main_params_trainable = config.lora_main_params_trainable

    @property
    def rotary_emb_base(self):
        return self.roberta.rotary_emb_base

    @rotary_emb_base.setter
    def rotary_emb_base(self, base):
        self.roberta.rotary_emb_base = base

    @property
    def main_params_trainable(self):
        return self._main_params_trainable

    @main_params_trainable.setter
    def main_params_trainable(self, val: bool):
        """Whether the main parameters (i.e. those that are not LoRA) should be trainable.
        This method sets the `requires_grad_` attribute of the main weights
        and controls which parameters are returned in `self.parameters()`.
        :param val: Whether or not to make the parameters trainable.
        :return: None
        """
        self._main_params_trainable = val
        for name, param in super().named_parameters():
            if "lora" not in name:
                param.requires_grad_(val)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        use_safetensors: bool = None,
        **kwargs,
    ):
        for key in list(kwargs.keys()):
            if key in config.to_dict():
                config.update({key: kwargs.pop(key)})
        if config.load_trained_adapters:  # checkpoint already contains LoRA adapters
            return super().from_pretrained(
                pretrained_model_name_or_path,
                *model_args,
                config=config,
                cache_dir=cache_dir,
                ignore_mismatched_sizes=ignore_mismatched_sizes,
                force_download=force_download,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                use_safetensors=use_safetensors,
                **kwargs,
            )
        else:  # initializing new adapters
            roberta = XLMRobertaModel.from_pretrained(
                pretrained_model_name_or_path,
                *model_args,
                use_flash_attn=config.use_flash_attn,
                **kwargs,
            )
            return cls(config, roberta=roberta)

    def _register_lora(self, num_adaptations, rank, dropout_p, alpha):
        self.apply(
            partial(
                LoRAParametrization.add_to_layer,
                num_adaptations=num_adaptations,
                rank=rank,
                dropout_p=dropout_p,
                alpha=alpha,
            )
        )

    def forward(self, *args, **kwargs):
        return self.roberta(*args, **kwargs)

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        for _, param in self.named_parameters(recurse=recurse):
            yield param

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, Parameter]]:
        for name, param in super().named_parameters(
            prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate
        ):
            if "lora" in name or self.main_params_trainable:
                yield name, param

    @torch.inference_mode()
    def encode(
        self,
        sentences: Union[str, List[str]],
        *args,
        task: Optional[str] = None,
        **kwargs,
    ) -> Union[List[torch.Tensor], np.ndarray, torch.Tensor]:
        """
        Computes sentence embeddings.
        sentences(`str` or `List[str]`):
            Sentence or sentences to be encoded
        task(`str`, *optional*, defaults to `None`):
            Specifies the task for which the encoding is intended. If `task` is not provided,
            all LoRA adapters are disabled, and the model reverts to its original,
            general-purpose weights.
        """
        if task and task not in self._lora_adaptations:
            raise ValueError(
                f"Unsupported task '{task}'. "
                f"Supported tasks are: {', '.join(self.config.lora_adaptations)}."
                f"Alternatively, don't pass the `task` argument to disable LoRA."
            )
        adapter_mask = None
        if task:
            task_id = self._adaptation_map[task]
            num_examples = 1 if isinstance(sentences, str) else len(sentences)
            adapter_mask = torch.full(
                (num_examples,), task_id, dtype=torch.int32, device=self.device
            )
            if isinstance(sentences, str):
                sentences = self._task_instructions[task] + sentences
            else:
                sentences = [
                    self._task_instructions[task] + sentence for sentence in sentences
                ]
        return self.roberta.encode(
            sentences, *args, adapter_mask=adapter_mask, **kwargs
        )


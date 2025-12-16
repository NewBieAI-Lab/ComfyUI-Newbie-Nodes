from typing import Any, Dict, List, Optional, Union

import torch
from transformers import PretrainedConfig


class XLMRobertaFlashConfig(PretrainedConfig):

    model_type = "xlm-roberta"

    def __init__(
        self,
        vocab_size: int = 250002,
        hidden_size: int = 1024,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 16,
        intermediate_size: int = 4096,
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 8194,
        type_vocab_size: int = 1,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-05,
        pad_token_id: int = 1,
        bos_token_id: int = 0,
        eos_token_id: int = 2,
        position_embedding_type: str = "rotary",
        rotary_emb_base: float = 10000.0,
        use_cache: bool = True,
        use_reentrant: bool = False,
        classifier_dropout: Optional[float] = None,
        lora_adaptations: Optional[List[str]] = None,
        task_instructions: Optional[Dict[str, str]] = None,
        lora_rank: int = 4,
        lora_dropout_p: float = 0.0,
        lora_alpha: int = 1,
        lora_main_params_trainable: bool = False,
        load_trained_adapters: bool = False,
        use_flash_attn: bool = True,
        torch_dtype: Optional[Union[str, torch.dtype]] = None,
        emb_pooler: Optional[str] = None,
        matryoshka_dimensions: Optional[List[int]] = None,
        truncate_dim: Optional[int] = None,
        **kwargs: Dict[str, Any],
    ):
        """
        Initialize the XLMRobertaFlashConfig configuration.

        Args:
            vocab_size (int): Size of the vocabulary.
            hidden_size (int): Dimensionality of the encoder layers and the pooler layer.
            num_hidden_layers (int): Number of hidden layers in the Transformer encoder.
            num_attention_heads (int): Number of attention heads for each attention layer in the Transformer encoder.
            intermediate_size (int): Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer.
            hidden_act (str): The activation function to use.
            hidden_dropout_prob (float): The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob (float): The dropout ratio for the attention probabilities.
            max_position_embeddings (int): The maximum length of the position embeddings.
            type_vocab_size (int): The vocabulary size of the token type ids.
            initializer_range (float): The standard deviation for initializing all weight matrices.
            layer_norm_eps (float): The epsilon used by the layer normalization layers.
            pad_token_id (int): The ID of the padding token.
            bos_token_id (int): The ID of the beginning-of-sequence token.
            eos_token_id (int): The ID of the end-of-sequence token.
            position_embedding_type (str): Type of position embeddings. Options are 'absolute', 'alibi', or 'rotary'.
            rotary_emb_base (float): Base for rotary embeddings.
            use_cache (bool): Whether or not the model should return the last key/values attentions (not used by all models).
            use_reentrant (bool): Whether or not the model should enable the 'use_reentrant' flag in gradient checkpointing.
            classifier_dropout (Optional[float]): The dropout ratio for the classification head.
            lora_adaptations (Optional[List[str]]): LoRA adaptations configuration.
            lora_prompts (Optional[Dict[str, str]]): LoRA prompts configuration.
            lora_rank (int): Rank for LoRA adaptations.
            lora_dropout_p (float): Dropout probability for LoRA adaptations.
            lora_alpha (int): Alpha parameter for LoRA.
            lora_main_params_trainable (bool): Whether to make the main model parameters trainable when using LoRA.
            load_trained_adapters (bool): Whether to load trained adapters.
            use_flash_attn (bool): Whether to use FlashAttention.
            torch_dtype (Optional[Union[str, torch.dtype]]): Data type for the tensors.
            emb_pooler (Optional[str]): Pooling layer configuration.
            matryoshka_dimensions (Optional[List[int]]): Configuration for matryoshka dimension reduction.
            truncate_dim (Optional[int]): Dimension to truncate embeddings to, if any.
            **kwargs (Dict[str, Any]): Additional keyword arguments passed to the configuration.
        """

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.rotary_emb_base = rotary_emb_base
        self.use_cache = use_cache
        self.use_reentrant = use_reentrant
        self.classifier_dropout = classifier_dropout
        self.load_trained_adapters = load_trained_adapters
        self.lora_adaptations = lora_adaptations
        self.task_instructions = task_instructions
        self.lora_rank = lora_rank
        self.lora_dropout_p = lora_dropout_p
        self.lora_alpha = lora_alpha
        self.lora_main_params_trainable = lora_main_params_trainable
        self.use_flash_attn = use_flash_attn
        self.emb_pooler = emb_pooler
        self.matryoshka_dimensions = matryoshka_dimensions
        self.truncate_dim = truncate_dim
        if (
            torch_dtype
            and hasattr(torch, torch_dtype)
            and type(getattr(torch, torch_dtype)) is torch.dtype
        ):
            self.torch_dtype = getattr(torch, torch_dtype)
        else:
            self.torch_dtype = torch_dtype
        if not self.use_flash_attn or not torch.cuda.is_available():
            self.torch_dtype = torch.float32
# This implementation was adapted from https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/modules/embedding.py
# Commit id: f1a73d074002226c42ce65a1df170ecff9f022c0

# Copyright (c) 2022, Tri Dao.

import torch
import torch.nn as nn
from transformers.models.xlm_roberta.modeling_xlm_roberta import \
    create_position_ids_from_input_ids


class XLMRobertaEmbeddings(nn.Module):
    def __init__(
        self,
        embed_dim,
        vocab_size,
        max_position_embeddings,
        type_vocab_size,
        padding_idx=None,
        device=None,
        dtype=None,
    ):
        """
        If max_position_embeddings <= 0, there's no position embeddings
        If type_vocab_size <= 0, there's no token type embeddings
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.word_embeddings = nn.Embedding(
            vocab_size, embed_dim, padding_idx=padding_idx, **factory_kwargs
        )
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        if self.max_position_embeddings > 0:
            self.position_embeddings = nn.Embedding(
                max_position_embeddings, embed_dim, **factory_kwargs
            )
        if self.type_vocab_size > 0:
            self.token_type_embeddings = nn.Embedding(
                type_vocab_size, embed_dim, **factory_kwargs
            )

    def forward(
        self, input_ids, position_ids=None, token_type_ids=None, adapter_mask=None
    ):
        """
        input_ids: (batch, seqlen)
        position_ids: (batch, seqlen)
        token_type_ids: (batch, seqlen)
        adapter_mask: (batch, 1)
        """
        batch_size, seqlen = input_ids.shape
        if adapter_mask is not None:
            unique_tasks = torch.unique(adapter_mask)
            embedding_dtype = next(self.word_embeddings.parameters()).dtype
            embeddings = torch.empty(
                *input_ids.shape,
                self.word_embeddings.embedding_dim,
                dtype=embedding_dtype,
                device=input_ids.device
            )
            for task_id in unique_tasks:
                task_indices = (adapter_mask == task_id).nonzero(as_tuple=True)[0]
                task_input_ids = input_ids[task_indices]
                task_embeddings = self.word_embeddings(task_input_ids, task_id=task_id)
                embeddings[task_indices] = task_embeddings
        else:
            embeddings = self.word_embeddings(input_ids)
        if self.max_position_embeddings > 0:
            if position_ids is None:
                position_ids = create_position_ids_from_input_ids(
                    input_ids, padding_idx=self.word_embeddings.padding_idx
                ).to(input_ids.device)
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = embeddings + position_embeddings
        if self.type_vocab_size > 0:
            if token_type_ids is None:
                token_type_ids = torch.zeros(
                    seqlen, dtype=torch.long, device=input_ids.device
                )

            if adapter_mask is not None:
                unique_tasks = torch.unique(adapter_mask)
                for task_id in unique_tasks:
                    task_token_type_embeddings = self.token_type_embeddings(
                        token_type_ids, task_id=task_id
                    )
                    task_indices = (adapter_mask == task_id).nonzero(as_tuple=True)[0]
                    embeddings[task_indices] = (
                        embeddings[task_indices] + task_token_type_embeddings
                    )
            else:
                token_type_embeddings = self.token_type_embeddings(token_type_ids)
                embeddings = embeddings + token_type_embeddings
        return embeddings

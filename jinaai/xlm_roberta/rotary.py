# This implementation was adapted from https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/layers/rotary.py
# Modified to remove flash_attn/triton dependency and use pure PyTorch implementation

# Commit id: 3566596ad867ee415dd3c12616dd50c610176f6c
# Rotary varlen support from https://github.com/Dao-AILab/flash-attention/pull/556

# Copyright (c) 2023, Tri Dao.

from typing import Optional, Tuple, Union

import torch
from einops import rearrange, repeat


def rotate_half(x, interleaved=False):
    if not interleaved:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    else:
        x1, x2 = x[..., ::2], x[..., 1::2]
        return rearrange(
            torch.stack((-x2, x1), dim=-1), "... d two -> ... (d two)", two=2
        )


def apply_rotary_emb_torch(x, cos, sin, interleaved=False):
    """
    x: (batch_size, seqlen, nheads, headdim)
    cos, sin: (seqlen, rotary_dim / 2) or (batch_size, seqlen, rotary_dim / 2)
    """
    ro_dim = cos.shape[-1] * 2
    assert ro_dim <= x.shape[-1]
    cos, sin = (
        cos[: x.shape[1]],
        sin[: x.shape[1]],
    )
    cos = repeat(
        cos, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)"
    )
    sin = repeat(
        sin, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)"
    )
    return torch.cat(
        [
            x[..., :ro_dim] * cos + rotate_half(x[..., :ro_dim], interleaved) * sin,
            x[..., ro_dim:],
        ],
        dim=-1,
    )


def apply_rotary_emb_torch_with_offset(x, cos, sin, seqlen_offset=0, interleaved=False):
    """
    Apply rotary embedding with sequence offset support.
    x: (batch_size, seqlen, nheads, headdim)
    cos, sin: (seqlen_rotary, rotary_dim / 2)
    seqlen_offset: int or (batch_size,) tensor
    """
    seqlen = x.shape[1]
    ro_dim = cos.shape[-1] * 2
    assert ro_dim <= x.shape[-1]

    if isinstance(seqlen_offset, int):
        cos = cos[seqlen_offset : seqlen_offset + seqlen]
        sin = sin[seqlen_offset : seqlen_offset + seqlen]
    else:
        # Per-batch offset - need to handle each batch differently
        # For simplicity, use the first offset (most common case in inference)
        offset = seqlen_offset[0].item() if hasattr(seqlen_offset, '__len__') else seqlen_offset
        cos = cos[offset : offset + seqlen]
        sin = sin[offset : offset + seqlen]

    cos = repeat(
        cos, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)"
    )
    sin = repeat(
        sin, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)"
    )
    return torch.cat(
        [
            x[..., :ro_dim] * cos + rotate_half(x[..., :ro_dim], interleaved) * sin,
            x[..., ro_dim:],
        ],
        dim=-1,
    )


class ApplyRotaryEmb(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        cos,
        sin,
        interleaved=False,
        inplace=False,
        seqlen_offsets: Union[int, torch.Tensor] = 0,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
    ):
        # Use pure PyTorch implementation
        out = apply_rotary_emb_torch_with_offset(
            x, cos, sin, seqlen_offset=seqlen_offsets, interleaved=interleaved
        )

        if isinstance(seqlen_offsets, int):
            ctx.save_for_backward(cos, sin)
            ctx.seqlen_offsets = seqlen_offsets
        else:
            ctx.save_for_backward(cos, sin, seqlen_offsets)
            ctx.seqlen_offsets = None
        ctx.interleaved = interleaved
        ctx.inplace = inplace
        ctx.max_seqlen = max_seqlen
        return out

    @staticmethod
    def backward(ctx, do):
        seqlen_offsets = ctx.seqlen_offsets
        if seqlen_offsets is None:
            cos, sin, seqlen_offsets = ctx.saved_tensors
        else:
            cos, sin = ctx.saved_tensors

        # Backward pass: rotate with conjugate (negate sin)
        dx = apply_rotary_emb_torch_with_offset(
            do, cos, -sin, seqlen_offset=seqlen_offsets, interleaved=ctx.interleaved
        )
        return dx, None, None, None, None, None, None, None


def apply_rotary_emb(
    x,
    cos,
    sin,
    interleaved=False,
    inplace=False,
    seqlen_offsets: Union[int, torch.Tensor] = 0,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
):
    """
    Arguments:
        x: (batch_size, seqlen, nheads, headdim) if cu_seqlens is None
            else (total_seqlen, nheads, headdim)
        cos, sin: (seqlen_rotary, rotary_dim / 2)
        interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead
            of 1st half and 2nd half (GPT-NeoX style).
        inplace: if True, apply rotary embedding in-place.
        seqlen_offsets: (batch_size,) or int. Each sequence in x is shifted by this amount.
            Most commonly used in inference when we have KV cache.
        cu_seqlens: (batch + 1,) or None
        max_seqlen: int
    Return:
        out: (batch_size, seqlen, nheads, headdim) if cu_seqlens is None
            else (total_seqlen, nheads, headdim)
    rotary_dim must be <= headdim
    Apply rotary embedding to the first rotary_dim of x.
    """
    return ApplyRotaryEmb.apply(
        x, cos, sin, interleaved, inplace, seqlen_offsets, cu_seqlens, max_seqlen
    )


# For backward compatibility
apply_rotary_emb_func = apply_rotary_emb


class ApplyRotaryEmbQKV_(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        qkv,
        cos,
        sin,
        cos_k=None,
        sin_k=None,
        interleaved=False,
        seqlen_offsets: Union[int, torch.Tensor] = 0,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        use_flash_attn: bool = False,  # Ignored - always use PyTorch
    ):
        # batch, seqlen, three, nheads, headdim = qkv.shape
        assert qkv.shape[-3] == 3

        # Use pure PyTorch implementation
        q_rot = apply_rotary_emb_torch_with_offset(
            qkv[..., 0, :, :],
            cos,
            sin,
            seqlen_offset=seqlen_offsets,
            interleaved=interleaved,
        )

        cos_k = cos if cos_k is None else cos_k
        sin_k = sin if sin_k is None else sin_k
        k_rot = apply_rotary_emb_torch_with_offset(
            qkv[..., 1, :, :],
            cos_k,
            sin_k,
            seqlen_offset=seqlen_offsets,
            interleaved=interleaved,
        )

        qkv_out = torch.stack((q_rot, k_rot, qkv[..., 2, :, :]), dim=-3)

        if isinstance(seqlen_offsets, int):
            ctx.save_for_backward(cos, sin, cos_k, sin_k)
            ctx.seqlen_offsets = seqlen_offsets
        else:
            ctx.save_for_backward(cos, sin, cos_k, sin_k, seqlen_offsets)
            ctx.seqlen_offsets = None
        ctx.max_seqlen = max_seqlen
        ctx.interleaved = interleaved
        return qkv_out

    @staticmethod
    def backward(ctx, dqkv):
        seqlen_offsets = ctx.seqlen_offsets
        if seqlen_offsets is None:
            cos, sin, cos_k, sin_k, seqlen_offsets = ctx.saved_tensors
        else:
            cos, sin, cos_k, sin_k = ctx.saved_tensors

        cos_k = cos if cos_k is None else cos_k
        sin_k = sin if sin_k is None else sin_k

        # Backward: rotate with conjugate (negate sin)
        dq = apply_rotary_emb_torch_with_offset(
            dqkv[..., 0, :, :], cos, -sin, seqlen_offset=seqlen_offsets, interleaved=ctx.interleaved
        )
        dk = apply_rotary_emb_torch_with_offset(
            dqkv[..., 1, :, :], cos_k, -sin_k, seqlen_offset=seqlen_offsets, interleaved=ctx.interleaved
        )

        dqkv_out = torch.stack((dq, dk, dqkv[..., 2, :, :]), dim=-3)
        return dqkv_out, None, None, None, None, None, None, None, None, None


def apply_rotary_emb_qkv_(
    qkv,
    cos,
    sin,
    cos_k=None,
    sin_k=None,
    interleaved=False,
    seqlen_offsets: Union[int, torch.Tensor] = 0,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
    use_flash_attn=False,  # Ignored - always use PyTorch
):
    """
    Arguments:
        qkv: (batch_size, seqlen, 3, nheads, headdim) if cu_seqlens is None
            else (total_seqlen, 3, nheads, headdim)
        cos, sin: (seqlen, rotary_dim / 2)
        cos_k, sin_k: (seqlen, rotary_dim / 2), optional
        interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead of
            1st half and 2nd half (GPT-NeoX style).
        seqlen_offsets: (batch_size,) or int. Each sequence in Q and K is shifted by this amount.
            Most commonly used in inference when we have KV cache.
            cu_seqlens: (batch + 1,) or None
        max_seqlen: int
    Return:
        qkv: (batch_size, seqlen, 3, nheads, headdim) if cu_seqlens is None
            else (total_seqlen, 3, nheads, headdim)
    rotary_dim must be <= headdim
    Apply rotary embedding to Q and K.
    """
    return ApplyRotaryEmbQKV_.apply(
        qkv, cos, sin, cos_k, sin_k, interleaved, seqlen_offsets, cu_seqlens, max_seqlen, use_flash_attn,
    )


class ApplyRotaryEmbKV_(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        kv,
        cos,
        sin,
        interleaved=False,
        seqlen_offsets: Union[int, torch.Tensor] = 0,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
    ):
        # batch, seqlen, two, nheads, headdim = kv.shape
        assert kv.shape[-3] == 2

        # Apply rotary to K only
        k_rot = apply_rotary_emb_torch_with_offset(
            kv[..., 0, :, :],
            cos,
            sin,
            seqlen_offset=seqlen_offsets,
            interleaved=interleaved,
        )

        kv_out = torch.stack((k_rot, kv[..., 1, :, :]), dim=-3)

        if isinstance(seqlen_offsets, int):
            ctx.save_for_backward(cos, sin)
            ctx.seqlen_offsets = seqlen_offsets
        else:
            ctx.save_for_backward(cos, sin, seqlen_offsets)
            ctx.seqlen_offsets = None
        ctx.max_seqlen = max_seqlen
        ctx.interleaved = interleaved
        return kv_out

    @staticmethod
    def backward(ctx, dkv):
        seqlen_offsets = ctx.seqlen_offsets
        if seqlen_offsets is None:
            cos, sin, seqlen_offsets = ctx.saved_tensors
        else:
            cos, sin = ctx.saved_tensors

        # Backward: rotate K with conjugate
        dk = apply_rotary_emb_torch_with_offset(
            dkv[..., 0, :, :], cos, -sin, seqlen_offset=seqlen_offsets, interleaved=ctx.interleaved
        )

        dkv_out = torch.stack((dk, dkv[..., 1, :, :]), dim=-3)
        return dkv_out, None, None, None, None, None, None


def apply_rotary_emb_kv_(
    kv,
    cos,
    sin,
    interleaved=False,
    seqlen_offsets: Union[int, torch.Tensor] = 0,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
):
    """
    Arguments:
        kv: (batch_size, seqlen, 2, nheads, headdim) if cu_seqlens is None
            else (total_seqlen, 2, nheads, headdim)
        cos, sin: (seqlen, rotary_dim / 2)
        interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead of
            1st half and 2nd half (GPT-NeoX style).
        seqlen_offsets: (batch_size,) or int. Each sequence in Q and K is shifted by this amount.
            Most commonly used in inference when we have KV cache.
        cu_seqlens: (batch + 1,) or None
        max_seqlen: int
    Return:
        kv: (batch_size, seqlen, 2, nheads, headdim) if cu_seqlens is None
            else (total_seqlen, 2, nheads, headdim)
    rotary_dim must be <= headdim
    Apply rotary embedding to K.
    """
    return ApplyRotaryEmbKV_.apply(
        kv, cos, sin, interleaved, seqlen_offsets, cu_seqlens, max_seqlen
    )


class RotaryEmbedding(torch.nn.Module):
    """
    The rotary position embeddings from RoFormer_ (Su et. al).
    A crucial insight from the method is that the query and keys are
    transformed by rotation matrices which depend on the relative positions.

    Other implementations are available in the Rotary Transformer repo_ and in
    GPT-NeoX_, GPT-NeoX was an inspiration

    .. _RoFormer: https://arxiv.org/abs/2104.09864
    .. _repo: https://github.com/ZhuiyiTechnology/roformer
    .. _GPT-NeoX: https://github.com/EleutherAI/gpt-neox

    If scale_base is not None, this implements XPos (Sun et al., https://arxiv.org/abs/2212.10554).
    A recommended value for scale_base is 512: https://github.com/HazyResearch/flash-attention/issues/96
    Reference: https://github.com/sunyt32/torchscale/blob/main/torchscale/component/xpos_relative_position.py
    """

    def __init__(
        self,
        dim: int,
        base=10000.0,
        interleaved=False,
        scale_base=None,
        pos_idx_in_fp32=True,
        device=None,
        use_flash_attn=False,  # Ignored - always use PyTorch
    ):
        """
        interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead
            of 1st half and 2nd half (GPT-NeoX style).
        pos_idx_in_fp32: if True, the position indices [0.0, ..., seqlen - 1] are in fp32,
            otherwise they might be in lower precision.
            This option was added because previously (before 2023-07-02), when we construct
            the position indices, we use the dtype of self.inv_freq. In most cases this would
            be fp32, but if the model is trained in pure bf16 (not mixed precision), then
            self.inv_freq would be bf16, and the position indices are also in bf16.
            Because of the limited precision of bf16 (e.g. 1995.0 is rounded to 2000.0), the
            embeddings for some positions will coincide.
            To maintain compatibility with models previously trained in pure bf16,
            we add this option.
        """
        super().__init__()
        self.dim = dim
        self._base = float(base)
        self.pos_idx_in_fp32 = pos_idx_in_fp32
        self.use_flash_attn = False  # Always use PyTorch implementation
        # Generate and save the inverse frequency buffer (non trainable)
        inv_freq = self._compute_inv_freq(device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.interleaved = interleaved
        self.scale_base = scale_base
        scale = (
            (torch.arange(0, dim, 2, device=device, dtype=torch.float32) + 0.4 * dim)
            / (1.4 * dim)
            if scale_base is not None
            else None
        )
        self.register_buffer("scale", scale, persistent=False)

        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
        self._cos_k_cached = None
        self._sin_k_cached = None

    @property
    def base(self):
        return self._base

    @base.setter
    def base(self, new_base):
        new_base = float(new_base)
        if new_base > 0:
            if self._base != new_base:  # only update if the base value has changed
                self._base = new_base
                self._update_cos_sin_cache(
                    self._seq_len_cached,
                    device=self.inv_freq.device,
                    dtype=self._cos_cached.dtype if self._cos_cached is not None else None,
                    rotary_base_changed=True,
                )
        else:
            raise ValueError("Rotary base value must be positive")

    def _compute_inv_freq(self, device=None):
        return 1.0 / (
            self.base
            ** (
                torch.arange(0, self.dim, 2, device=device, dtype=torch.float32)
                / self.dim
            )
        )

    def _update_cos_sin_cache(
        self, seqlen, device=None, dtype=None, rotary_base_changed=False
    ):
        # Reset the tables if the sequence length has changed,
        # if we're on a new device (possibly due to tracing for instance),
        # or if we're switching from inference mode to training
        # or if the rotary base value was changed
        if (
            seqlen > self._seq_len_cached
            or self._cos_cached is None
            or self._cos_cached.device != device
            or self._cos_cached.dtype != dtype
            or (self.training and self._cos_cached.is_inference())
            or rotary_base_changed
        ):
            self._seq_len_cached = seqlen
            # We want fp32 here, not self.inv_freq.dtype, since the model could be loaded in bf16
            # And the output of arange can be quite large, so bf16 would lose a lot of precision.
            # However, for compatibility reason, we add an option to use the dtype of self.inv_freq.
            if rotary_base_changed:
                self.inv_freq = self._compute_inv_freq(device=device)
            if self.pos_idx_in_fp32:
                t = torch.arange(seqlen, device=device, dtype=torch.float32)
                # We want fp32 here as well since inv_freq will be multiplied with t, and the output
                # will be large. Having it in bf16 will lose a lot of precision and cause the
                # cos & sin output to change significantly.
                # We want to recompute self.inv_freq if it was not loaded in fp32
                if self.inv_freq.dtype != torch.float32:
                    inv_freq = self._compute_inv_freq(device=device)
                else:
                    inv_freq = self.inv_freq
            else:
                t = torch.arange(seqlen, device=device, dtype=self.inv_freq.dtype)
                inv_freq = self.inv_freq

            # Don't do einsum, it converts fp32 to fp16 under AMP
            # freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            freqs = torch.outer(t, inv_freq)
            if self.scale is None:
                self._cos_cached = torch.cos(freqs).to(dtype)
                self._sin_cached = torch.sin(freqs).to(dtype)
            else:
                power = (
                    torch.arange(
                        seqlen, dtype=self.scale.dtype, device=self.scale.device
                    )
                    - seqlen // 2
                ) / self.scale_base
                scale = self.scale.to(device=power.device) ** rearrange(
                    power, "s -> s 1"
                )
                # We want the multiplication by scale to happen in fp32
                self._cos_cached = (torch.cos(freqs) * scale).to(dtype)
                self._sin_cached = (torch.sin(freqs) * scale).to(dtype)
                self._cos_k_cached = (torch.cos(freqs) / scale).to(dtype)
                self._sin_k_cached = (torch.sin(freqs) / scale).to(dtype)

    def forward(
        self,
        qkv: torch.Tensor,
        kv: Optional[torch.Tensor] = None,
        seqlen_offset: Union[int, torch.Tensor] = 0,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        qkv: (batch, seqlen, 3, nheads, headdim) if kv is none,
             else it's just q of shape (batch, seqlen, nheads, headdim)
        kv: (batch, seqlen, 2, nheads, headdim)
        seqlen_offset: (batch_size,) or int. Each sequence in x is shifted by this amount.
            Most commonly used in inference when we have KV cache.
            If it's a tensor of shape (batch_size,), then to update the cos / sin cache, one
            should pass in max_seqlen, which will update the cos / sin cache up to that length.
        Apply rotary embedding to qkv and / or kv.
        """
        if cu_seqlens is not None:
            assert max_seqlen is not None
        seqlen = qkv.shape[1] if max_seqlen is None else max_seqlen
        if max_seqlen is not None:
            self._update_cos_sin_cache(max_seqlen, device=qkv.device, dtype=qkv.dtype)
        elif isinstance(seqlen_offset, int):
            self._update_cos_sin_cache(
                seqlen + seqlen_offset, device=qkv.device, dtype=qkv.dtype
            )
        if kv is None:
            if self.scale is None:
                return apply_rotary_emb_qkv_(
                    qkv,
                    self._cos_cached,
                    self._sin_cached,
                    interleaved=self.interleaved,
                    seqlen_offsets=seqlen_offset,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                    use_flash_attn=False,  # Always use PyTorch
                )
            else:
                return apply_rotary_emb_qkv_(
                    qkv,
                    self._cos_cached,
                    self._sin_cached,
                    self._cos_k_cached,
                    self._sin_k_cached,
                    interleaved=self.interleaved,
                    seqlen_offsets=seqlen_offset,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                    use_flash_attn=False,  # Always use PyTorch
                )
        else:
            q = qkv
            q = apply_rotary_emb_func(
                q,
                self._cos_cached,
                self._sin_cached,
                interleaved=self.interleaved,
                inplace=False,  # Can't use inplace with PyTorch implementation
                seqlen_offsets=seqlen_offset,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
            if self.scale is None:
                kv = apply_rotary_emb_kv_(
                    kv,
                    self._cos_cached,
                    self._sin_cached,
                    interleaved=self.interleaved,
                    seqlen_offsets=seqlen_offset,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                )
            else:
                kv = apply_rotary_emb_kv_(
                    kv,
                    self._cos_k_cached,
                    self._sin_k_cached,
                    interleaved=self.interleaved,
                    seqlen_offsets=seqlen_offset,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                )
            return q, kv

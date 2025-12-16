import torch
import math

import comfy.model_patcher
import comfy.latent_formats
import comfy.model_sampling
import comfy.conds


class NewBieModelConfig:
    def __init__(self):
        self.unet_config = {
            "image_model": "newbie",
            "in_channels": 16,
            "dim": 2304,
            "cap_feat_dim": 2560,
            "n_layers": 36,
            "n_heads": 24,
            "n_kv_heads": 8,
        }
        self.latent_format = comfy.latent_formats.Flux()
        self.manual_cast_dtype = None
        self.sampling_settings = {"shift": 6.0, "multiplier": 1.0}
        self.memory_usage_factor = 1.2
        self.supported_inference_dtypes = [torch.bfloat16, torch.float16, torch.float32]


class NewBieModelSampling(comfy.model_sampling.ModelSamplingDiscreteFlow, comfy.model_sampling.CONST):
    pass


class NewBieBaseModel(torch.nn.Module):
    def __init__(self, diffusion_model, model_config, device=None):
        super().__init__()
        self.diffusion_model = diffusion_model
        self.model_config = model_config
        self.latent_format = model_config.latent_format
        self.manual_cast_dtype = model_config.manual_cast_dtype
        self.device = device
        self.model_type = None
        self.memory_usage_factor = model_config.memory_usage_factor
        self.model_sampling = NewBieModelSampling(model_config)
        self.adm_channels = 0
        self.inpaint_model = False
        self.concat_keys = ()
        self.memory_usage_factor_conds = ()
        self.current_patcher = None

    def get_dtype(self):
        for param in self.diffusion_model.parameters():
            return param.dtype
        return torch.float32

    def memory_required(self, input_shape, cond_shapes={}):
        area = input_shape[0] * math.prod(input_shape[2:])
        return (area * 0.15 * self.memory_usage_factor) * (1024 * 1024)

    def extra_conds_shapes(self, **kwargs):
        return {}

    def encode_adm(self, **kwargs):
        return None

    def concat_cond(self, **kwargs):
        return None

    def process_latent_in(self, latent):
        return self.latent_format.process_in(latent)

    def process_latent_out(self, latent):
        return self.latent_format.process_out(latent)

    def process_timestep(self, timestep, **kwargs):
        return timestep

    def extra_conds(self, **kwargs):
        out = {}
        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            out['c_crossattn'] = comfy.conds.CONDCrossAttn(cross_attn)

        attention_mask = kwargs.get("attention_mask", None)
        if attention_mask is not None:
            out['attention_mask'] = comfy.conds.CONDRegular(attention_mask)

        cap_feats = kwargs.get("cap_feats", None)
        if cap_feats is not None:
            out['cap_feats'] = comfy.conds.CONDRegular(cap_feats)

        cap_mask = kwargs.get("cap_mask", None)
        if cap_mask is not None:
            out['cap_mask'] = comfy.conds.CONDRegular(cap_mask)

        clip_text_pooled = kwargs.get("clip_text_pooled", None)
        if clip_text_pooled is not None:
            out['clip_text_pooled'] = comfy.conds.CONDRegular(clip_text_pooled)

        clip_img_pooled = kwargs.get("clip_img_pooled", None)
        if clip_img_pooled is not None:
            out['clip_img_pooled'] = comfy.conds.CONDRegular(clip_img_pooled)

        return out

    def apply_model(self, x, t, c_concat=None, c_crossattn=None, control=None, transformer_options={}, **kwargs):
        sigma = t
        xc = self.model_sampling.calculate_input(sigma, x)
        if c_concat is not None:
            xc = torch.cat([xc] + [c_concat], dim=1)

        dtype = self.get_dtype()
        if self.manual_cast_dtype is not None:
            dtype = self.manual_cast_dtype

        device = xc.device
        xc = xc.to(dtype)

        t_val = (1.0 - sigma).float().to(device)

        cap_feats = kwargs.get('cap_feats', c_crossattn)
        cap_mask = kwargs.get('cap_mask', kwargs.get('attention_mask'))
        clip_text_pooled = kwargs.get('clip_text_pooled')
        clip_img_pooled = kwargs.get('clip_img_pooled')

        if cap_feats is not None:
            cap_feats = cap_feats.to(device=device, dtype=dtype)
        if cap_mask is None and cap_feats is not None:
            cap_mask = torch.ones(cap_feats.shape[:2], dtype=torch.long, device=device)
        elif cap_mask is not None:
            cap_mask = cap_mask.to(device)

        model_kwargs = {}
        if clip_text_pooled is not None:
            model_kwargs['clip_text_pooled'] = clip_text_pooled.to(device=device, dtype=dtype)
        if clip_img_pooled is not None:
            model_kwargs['clip_img_pooled'] = clip_img_pooled.to(device=device, dtype=dtype)

        # Ensure diffusion_model is on the correct device before forward pass
        self.diffusion_model.to(device)

        model_output = self.diffusion_model(xc, t_val, cap_feats, cap_mask, **model_kwargs).float()
        model_output = -model_output

        return self.model_sampling.calculate_denoised(sigma, model_output, x)

    def forward(self, *args, **kwargs):
        return self.apply_model(*args, **kwargs)

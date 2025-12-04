import torch
import math
import os
from safetensors.torch import load_file

import folder_paths
import comfy.model_management as model_management
import comfy.model_patcher
import comfy.latent_formats
import comfy.model_sampling
import comfy.conds
import comfy.sd

from transformers import AutoModel, AutoConfig, AutoTokenizer

from .comfy_newbie_clip_loader import NewBieCLIP


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
        xc = xc.to(dtype)

        t_val = (1.0 - sigma).float()

        cap_feats = kwargs.get('cap_feats', c_crossattn)
        cap_mask = kwargs.get('cap_mask', kwargs.get('attention_mask'))
        clip_text_pooled = kwargs.get('clip_text_pooled')
        clip_img_pooled = kwargs.get('clip_img_pooled')

        if cap_feats is not None:
            cap_feats = cap_feats.to(dtype)
        if cap_mask is None and cap_feats is not None:
            cap_mask = torch.ones(cap_feats.shape[:2], dtype=torch.long, device=cap_feats.device)

        model_kwargs = {}
        if clip_text_pooled is not None:
            model_kwargs['clip_text_pooled'] = clip_text_pooled.to(dtype)
        if clip_img_pooled is not None:
            model_kwargs['clip_img_pooled'] = clip_img_pooled.to(dtype)

        model_output = self.diffusion_model(xc, t_val, cap_feats, cap_mask, **model_kwargs).float()
        model_output = -model_output

        return self.model_sampling.calculate_denoised(sigma, model_output, x)

    def forward(self, *args, **kwargs):
        return self.apply_model(*args, **kwargs)


class NewBieCheckpointLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
                "dtype": (["default", "fp16", "bf16", "fp8_e4m3fn", "fp8_e5m2"], {"default": "bf16"}),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_checkpoint"
    CATEGORY = "loaders"
    TITLE = "NewBie Checkpoint Loader"

    def load_checkpoint(self, ckpt_name: str, dtype: str = "bf16"):
        ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)

        dtype_map = {
            "default": torch.bfloat16,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp8_e4m3fn": torch.float8_e4m3fn,
            "fp8_e5m2": torch.float8_e5m2,
        }
        torch_dtype = dtype_map.get(dtype, torch.bfloat16)
        load_device = model_management.get_torch_device()
        offload_device = model_management.unet_offload_device()

        print(f"Loading NewBie checkpoint: {ckpt_path}")
        state_dict = load_file(ckpt_path, device="cpu")

        dit_sd = {}
        gemma_sd = {}
        jina_sd = {}
        vae_sd = {}

        for k, v in state_dict.items():
            if k.startswith("model.diffusion_model."):
                new_key = k.replace("model.diffusion_model.", "")
                dit_sd[new_key] = v
            elif k.startswith("text_encoders.gemma3.transformer."):
                new_key = k.replace("text_encoders.gemma3.transformer.", "")
                gemma_sd[new_key] = v
            elif k.startswith("text_encoders.jina_clip."):
                new_key = k.replace("text_encoders.jina_clip.", "")
                jina_sd[new_key] = v
            elif k.startswith("vae."):
                new_key = k.replace("vae.", "")
                vae_sd[new_key] = v

        model = self._load_dit(dit_sd, torch_dtype, load_device, offload_device)
        clip = self._load_clip(gemma_sd, jina_sd, torch_dtype, str(load_device))
        vae = self._load_vae(vae_sd)

        return (model, clip, vae)

    def _load_dit(self, state_dict, dtype, load_device, offload_device):
        from .models.model import NextDiT_3B_GQA_patch2_Adaln_Refiner_WHIT_CLIP

        print("Loading DiT model...")
        cap_feat_dim = state_dict["cap_embedder.0.weight"].shape[0]

        diffusion_model = NextDiT_3B_GQA_patch2_Adaln_Refiner_WHIT_CLIP(
            in_channels=16,
            cap_feat_dim=cap_feat_dim,
            qk_norm=True,
            clip_text_dim=1024,
            clip_img_dim=1024,
        )

        missing, unexpected = diffusion_model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"DiT missing keys: {len(missing)}")
        if unexpected:
            print(f"DiT unexpected keys: {len(unexpected)}")

        diffusion_model = diffusion_model.to(dtype=dtype, device=offload_device)
        diffusion_model.eval()

        model_config = NewBieModelConfig()
        model = NewBieBaseModel(diffusion_model, model_config, device=offload_device)

        patcher = comfy.model_patcher.ModelPatcher(model, load_device=load_device, offload_device=offload_device)
        return patcher

    def _load_clip(self, gemma_sd, jina_sd, dtype, device):
        print("Loading Gemma model...")

        gemma_config = AutoConfig.from_pretrained("unsloth/gemma-3-4b-it", trust_remote_code=True)
        with torch.device("meta"):
            gemma_model = AutoModel.from_config(gemma_config, trust_remote_code=True)
        gemma_model.to_empty(device="cpu")
        missing, unexpected = gemma_model.load_state_dict(gemma_sd, strict=False, assign=True)
        print(f"  Gemma: missing={len(missing)}, unexpected={len(unexpected)}")
        if missing:
            print(f"  Missing sample: {missing[:3]}")
        gemma_model = gemma_model.to(device=device, dtype=dtype)
        gemma_model.eval()
        gemma_tokenizer = AutoTokenizer.from_pretrained("unsloth/gemma-3-4b-it", trust_remote_code=True)
        gemma_tokenizer.padding_side = "right"

        print("Loading Jina CLIP model...")
        jina_config = AutoConfig.from_pretrained("jinaai/jina-clip-v2", trust_remote_code=True)
        jina_config.use_flash_attn = False
        if hasattr(jina_config, 'text_config') and hasattr(jina_config.text_config, 'hf_model_config_kwargs'):
            jina_config.text_config.hf_model_config_kwargs['use_flash_attn'] = False
        jina_model = AutoModel.from_pretrained("jinaai/jina-clip-v2", config=jina_config, trust_remote_code=True)
        self._load_jina_with_lora(jina_model, jina_sd)
        jina_model = jina_model.to(device=device, dtype=dtype)
        jina_model.eval()
        jina_tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-clip-v2", trust_remote_code=True)

        self._install_hidden_state_hook(jina_model)

        clip = NewBieCLIP(
            text_encoder=gemma_model,
            tokenizer=gemma_tokenizer,
            clip_model=jina_model,
            clip_tokenizer=jina_tokenizer,
            device=device,
            cpu_offload=False,
            processor=None,
            enable_jina_weights=True,
            weight_baseline_mode="mean",
            weight_strength=1.0,
            mask_normalization=True
        )
        return clip

    def _install_hidden_state_hook(self, clip_model):
        clip_model._last_hidden_states = None
        def hook_fn(module, input, output):
            if hasattr(output, 'last_hidden_state'):
                clip_model._last_hidden_states = output.last_hidden_state
            elif isinstance(output, tuple):
                for out in output:
                    if isinstance(out, torch.Tensor) and out.dim() == 3:
                        clip_model._last_hidden_states = out
                        break
        if hasattr(clip_model, 'text_model'):
            if hasattr(clip_model.text_model, 'transformer'):
                clip_model.text_model.transformer.register_forward_hook(hook_fn)
            elif hasattr(clip_model.text_model, 'encoder'):
                clip_model.text_model.encoder.register_forward_hook(hook_fn)
            else:
                clip_model.text_model.register_forward_hook(hook_fn)

    def _load_jina_with_lora(self, model, state_dict):
        model_sd = model.state_dict()
        new_sd = {}
        for k, v in state_dict.items():
            if k in model_sd:
                new_sd[k] = v
            elif "parametrizations.weight.original" in k:
                base_key = k.replace(".parametrizations.weight.original", ".weight")
                if base_key in model_sd:
                    lora_a_key = k.replace(".original", ".0.lora_A")
                    lora_b_key = k.replace(".original", ".0.lora_B")
                    if lora_a_key in state_dict and lora_b_key in state_dict:
                        lora_a = state_dict[lora_a_key]
                        lora_b = state_dict[lora_b_key]
                        merged = v + (lora_b @ lora_a)
                        new_sd[base_key] = merged
                    else:
                        new_sd[base_key] = v
        model.load_state_dict(new_sd, strict=False)

    def _load_vae(self, state_dict):
        if not state_dict:
            print("WARNING: No VAE weights detected.")
            return None
        try:
            vae = comfy.sd.VAE(sd=state_dict)
            return vae
        except Exception as e:
            print(f"VAE load error: {e}")
            return None


NODE_CLASS_MAPPINGS = {
    "NewBieCheckpointLoader": NewBieCheckpointLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NewBieCheckpointLoader": "NewBie Checkpoint Loader",
}

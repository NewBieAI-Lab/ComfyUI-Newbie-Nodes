import torch
import os
from safetensors.torch import load_file

import folder_paths
import comfy.model_management as model_management
import comfy.model_patcher
import comfy.sd

from transformers import AutoModel, AutoConfig, AutoTokenizer

from .comfy_newbie_clip_loader import NewBieCLIP
from .newbie_model_support import NewBieModelConfig, NewBieBaseModel


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
            weight_baseline_mode="compel",
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

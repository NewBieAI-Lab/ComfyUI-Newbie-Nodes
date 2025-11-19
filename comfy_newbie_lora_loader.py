import os
import typing
import math
import json

import torch
from safetensors import safe_open

import lora as comfy_lora
import comfy.utils as comfy_utils
import comfy.model_patcher
import folder_paths


def _get_model_state_dict(model: typing.Any) -> dict:
    if hasattr(model, "model_state_dict"):
        try:
            return model.model_state_dict()
        except TypeError:
            return model.model_state_dict(None)
    return model.state_dict()


def build_newbie_lora_key_map(model) -> dict:
    sd = _get_model_state_dict(model)
    key_map = {}
    for full_key in sd.keys():
        if not full_key.endswith(".weight"):
            continue
        base = full_key[:-len(".weight")]
        variants = set()
        variants.add(base)
        variants.add("base_model.model." + base)
        variants.add("transformer." + base)
        if base.startswith("diffusion_model."):
            short = base[len("diffusion_model."):]
            variants.add(short)
            variants.add("base_model.model." + short)
            variants.add("transformer." + short)
            variants.add("unet.base_model.model." + short)
        for v in variants:
            if v not in key_map:
                key_map[v] = full_key
    return key_map


def estimate_lora_rank(lora_sd: dict) -> typing.Optional[float]:
    ranks = []
    for k, v in lora_sd.items():
        if not isinstance(v, torch.Tensor):
            continue
        name = k.lower()
        if any(x in name for x in ["lora_up.weight", "lora_down.weight", "lora_a.weight", "lora_b.weight"]):
            if v.ndim >= 2:
                r = min(v.shape[0], v.shape[1])
                if r > 0:
                    ranks.append(r)
        elif "lokr" in name and v.ndim >= 2:
            r = min(v.shape[0], v.shape[1])
            if r > 0:
                ranks.append(r)
    if not ranks:
        return None
    return float(sum(ranks) / len(ranks))


def load_newbie_lora_state_dict(lora_name: str) -> tuple:
    if not lora_name:
        raise ValueError("LoRA name is empty.")
    lora_path = folder_paths.get_full_path("loras", lora_name)
    if lora_path is None:
        raise FileNotFoundError(f"LoRA '{lora_name}' not found in models/loras folder.")
    if os.path.isdir(lora_path):
        raise ValueError(f"'{lora_path}' is a directory. Please select a LoRA file instead of a folder.")

    metadata = {}
    if lora_path.endswith('.safetensors'):
        with safe_open(lora_path, framework="pt", device="cpu") as f:
            metadata = f.metadata() or {}

    sd = comfy_utils.load_torch_file(lora_path)
    if not isinstance(sd, dict):
        raise ValueError(f"Loaded LoRA '{lora_name}' does not contain a valid state dict.")
    return sd, metadata


def apply_newbie_lora_to_model(
    model,
    lora_name: str,
    strength: float,
) -> comfy.model_patcher.ModelPatcher:
    if strength == 0.0:
        return model
    lora_sd, metadata = load_newbie_lora_state_dict(lora_name)

    scale = 1.0
    if metadata:
        lora_rank = float(metadata.get("lora_rank", 0))
        lora_alpha = float(metadata.get("lora_alpha", lora_rank))
        if lora_rank > 0:
            scale = lora_alpha / lora_rank

    final_strength = strength * scale
    to_load = build_newbie_lora_key_map(model)
    patches = comfy_lora.load_lora(lora_sd, to_load, log_missing=False)
    if not patches:
        return model
    if hasattr(model, "clone"):
        patched = model.clone()
    else:
        patched = model
    if hasattr(patched, "add_patches"):
        patched.add_patches(patches, strength_patch=float(final_strength), strength_model=1.0)
    return patched


class NewBieLoraModelOnly:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.01}),
                "enabled": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "NewBie/LoRA"

    def apply(self, model, lora_name, strength, enabled=True):
        if not enabled or not lora_name:
            return (model,)
        patched = apply_newbie_lora_to_model(model, lora_name, strength)
        return (patched,)


class NewBieLoraLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.01}),
                "enabled": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "apply"
    CATEGORY = "NewBie/LoRA"

    def apply(self, model, clip, lora_name, strength, enabled=True):
        if not enabled or not lora_name:
            return (model, clip)
        patched = apply_newbie_lora_to_model(model, lora_name, strength)
        return (patched, clip)


class NewBieLoraLoaderMulti:
    MAX_LORAS = 5

    @classmethod
    def INPUT_TYPES(cls):
        lora_list = folder_paths.get_filename_list("loras")
        required = {
            "model": ("MODEL",),
            "clip": ("CLIP",),
        }
        for i in range(1, cls.MAX_LORAS + 1):
            required[f"lora_name_{i}"] = (lora_list,)
            required[f"strength_{i}"] = ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.01})
            required[f"enabled_{i}"] = ("BOOLEAN", {"default": False})
        return {"required": required}

    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "apply"
    CATEGORY = "NewBie/LoRA"

    def apply(self, model, clip, **kwargs):
        patched = model
        for i in range(1, self.MAX_LORAS + 1):
            name_key = f"lora_name_{i}"
            strength_key = f"strength_{i}"
            enabled_key = f"enabled_{i}"
            lora_name = kwargs.get(name_key, None)
            strength = kwargs.get(strength_key, 1.0)
            enabled = kwargs.get(enabled_key, False)
            if not enabled:
                continue
            if not lora_name:
                continue
            patched = apply_newbie_lora_to_model(patched, lora_name, strength)
        return (patched, clip)


NODE_CLASS_MAPPINGS = {
    "NewBieLoraModelOnly": NewBieLoraModelOnly,
    "NewBieLoraLoader": NewBieLoraLoader,
    "NewBieLoraLoaderMulti": NewBieLoraLoaderMulti,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NewBieLoraModelOnly": "NewBie LoRa Loader (Model Only)",
    "NewBieLoraLoader": "NewBie LoRa Loader",
    "NewBieLoraLoaderMulti": "NewBie LoRa Loader (Multi)",
}

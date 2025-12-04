#!/usr/bin/env python3
"""
NewBie All-in-One 模型打包脚本
将 DiT, Gemma3, Jina CLIP, VAE 打包成单个 safetensors 文件
"""

import argparse
import os
import torch
from safetensors.torch import load_file, save_file
from pathlib import Path
from typing import Dict, Optional
import glob


def load_dit_weights(dit_path: str) -> Dict[str, torch.Tensor]:
    """加载DiT权重"""
    print(f"Loading DiT from: {dit_path}")

    if dit_path.endswith('.safetensors'):
        state_dict = load_file(dit_path, device='cpu')
    else:
        ckpt = torch.load(dit_path, map_location='cpu')
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        elif 'model' in ckpt:
            state_dict = ckpt['model']
        else:
            state_dict = ckpt

    result = {}
    for k, v in state_dict.items():
        new_key = f"model.diffusion_model.{k}"
        result[new_key] = v

    print(f"  DiT: {len(result)} keys")
    return result


def load_gemma_weights(gemma_path: str) -> Dict[str, torch.Tensor]:
    """加载Gemma3权重 (支持分片safetensors和HuggingFace repo)"""
    print(f"Loading Gemma from: {gemma_path}")

    state_dict = {}

    if os.path.isdir(gemma_path):
        safetensors_files = glob.glob(os.path.join(gemma_path, "*.safetensors"))
        safetensors_files = [f for f in safetensors_files if 'model' in os.path.basename(f).lower()]

        if not safetensors_files:
            safetensors_files = glob.glob(os.path.join(gemma_path, "*.safetensors"))

        for sf in sorted(safetensors_files):
            print(f"  Loading shard: {os.path.basename(sf)}")
            shard = load_file(sf, device='cpu')
            state_dict.update(shard)
    elif os.path.isfile(gemma_path):
        state_dict = load_file(gemma_path, device='cpu')
    else:
        print(f"  Downloading from HuggingFace: {gemma_path}")
        from huggingface_hub import snapshot_download
        local_dir = snapshot_download(gemma_path, local_files_only=False)
        safetensors_files = glob.glob(os.path.join(local_dir, "*.safetensors"))
        safetensors_files = [f for f in safetensors_files if 'model' in os.path.basename(f).lower()]
        if not safetensors_files:
            safetensors_files = glob.glob(os.path.join(local_dir, "*.safetensors"))
        for sf in sorted(safetensors_files):
            print(f"  Loading shard: {os.path.basename(sf)}")
            shard = load_file(sf, device='cpu')
            state_dict.update(shard)

    result = {}
    for k, v in state_dict.items():
        new_key = f"text_encoders.gemma3.transformer.{k}"
        result[new_key] = v

    print(f"  Gemma: {len(result)} keys")
    return result


def load_jina_weights(jina_path: str) -> Dict[str, torch.Tensor]:
    """加载Jina CLIP权重"""
    print(f"Loading Jina CLIP from: {jina_path}")

    state_dict = {}

    if os.path.isdir(jina_path):
        safetensors_files = glob.glob(os.path.join(jina_path, "*.safetensors"))
        safetensors_files = [f for f in safetensors_files if 'model' in os.path.basename(f).lower()]

        if not safetensors_files:
            safetensors_files = glob.glob(os.path.join(jina_path, "*.safetensors"))

        for sf in sorted(safetensors_files):
            print(f"  Loading shard: {os.path.basename(sf)}")
            shard = load_file(sf, device='cpu')
            state_dict.update(shard)
    else:
        state_dict = load_file(jina_path, device='cpu')

    result = {}
    for k, v in state_dict.items():
        new_key = f"text_encoders.jina_clip.{k}"
        result[new_key] = v

    print(f"  Jina CLIP: {len(result)} keys")
    return result


def load_vae_weights(vae_path: str) -> Dict[str, torch.Tensor]:
    """加载VAE权重"""
    print(f"Loading VAE from: {vae_path}")

    if vae_path.endswith('.safetensors'):
        state_dict = load_file(vae_path, device='cpu')
    else:
        ckpt = torch.load(vae_path, map_location='cpu')
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt

    result = {}
    for k, v in state_dict.items():
        if not k.startswith('vae.'):
            new_key = f"vae.{k}"
        else:
            new_key = k
        result[new_key] = v

    print(f"  VAE: {len(result)} keys")
    return result


def convert_dtype(state_dict: Dict[str, torch.Tensor], target_dtype: str) -> Dict[str, torch.Tensor]:
    """转换数据类型"""
    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }

    torch_dtype = dtype_map.get(target_dtype, torch.bfloat16)

    result = {}
    for k, v in state_dict.items():
        if v.dtype in [torch.float32, torch.float16, torch.bfloat16]:
            result[k] = v.to(torch_dtype)
        else:
            result[k] = v

    return result


def create_allinone(
    dit_path: str,
    gemma_path: str,
    jina_path: str,
    vae_path: str,
    output_path: str,
    dtype: str = "bf16",
):
    """创建All-in-One模型"""
    print("=" * 60)
    print("Creating NewBie All-in-One Model")
    print("=" * 60)

    all_weights = {}

    dit_weights = load_dit_weights(dit_path)
    all_weights.update(dit_weights)

    gemma_weights = load_gemma_weights(gemma_path)
    all_weights.update(gemma_weights)

    jina_weights = load_jina_weights(jina_path)
    all_weights.update(jina_weights)

    vae_weights = load_vae_weights(vae_path)
    all_weights.update(vae_weights)

    print(f"\nConverting to {dtype}...")
    all_weights = convert_dtype(all_weights, dtype)

    print(f"\nSaving to: {output_path}")
    save_file(all_weights, output_path)

    total_params = sum(v.numel() for v in all_weights.values())
    file_size = os.path.getsize(output_path) / (1024**3)

    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Total keys: {len(all_weights)}")
    print(f"  Total params: {total_params:,}")
    print(f"  File size: {file_size:.2f} GB")
    print(f"  Output: {output_path}")
    print("=" * 60)

    print("\nKey prefixes:")
    prefixes = {}
    for k in all_weights.keys():
        prefix = k.split('.')[0]
        if prefix not in prefixes:
            prefixes[prefix] = 0
        prefixes[prefix] += 1
    for p, c in sorted(prefixes.items()):
        print(f"  {p}: {c} keys")


def main():
    parser = argparse.ArgumentParser(description="Create NewBie All-in-One model")

    parser.add_argument("--dit", type=str, required=True, help="Path to DiT checkpoint")
    parser.add_argument("--gemma", type=str, required=True, help="Path to Gemma3 model directory or safetensors")
    parser.add_argument("--jina", type=str, required=True, help="Path to Jina CLIP model directory or safetensors")
    parser.add_argument("--vae", type=str, required=True, help="Path to VAE checkpoint")
    parser.add_argument("--output", type=str, required=True, help="Output safetensors path")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"], help="Target dtype")

    args = parser.parse_args()

    create_allinone(
        dit_path=args.dit,
        gemma_path=args.gemma,
        jina_path=args.jina,
        vae_path=args.vae,
        output_path=args.output,
        dtype=args.dtype,
    )


if __name__ == "__main__":
    main()

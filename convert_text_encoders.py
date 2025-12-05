#!/usr/bin/env python3
"""
Convert Gemma3-4B-IT and Jina-CLIP-v2 models to single safetensors files.
Only outputs 2 safetensors files - config and tokenizer info embedded in metadata.

Usage:
    # Convert Jina CLIP only
    python convert_text_encoders.py --jina-repo jinaai/jina-clip-v2 --output-dir ./models/text_encoders

    # Convert Gemma only
    python convert_text_encoders.py --gemma-repo google/gemma-3-4b-it --output-dir ./models/text_encoders

    # Convert both
    python convert_text_encoders.py --gemma-repo google/gemma-3-4b-it --jina-repo jinaai/jina-clip-v2 --output-dir ./models/text_encoders

    # Verify converted models
    python convert_text_encoders.py --verify --output-dir ./models/text_encoders
"""

import argparse
import os
import json
from typing import Dict, Any, Tuple

import torch
from safetensors.torch import save_file, load_file

try:
    from transformers import AutoModel, AutoTokenizer, AutoConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Error: transformers library is required. Install with: pip install transformers")
    exit(1)


def convert_gemma_to_safetensors(
    repo_path: str,
    output_dir: str,
    dtype: str = "bf16",
    model_name: str = "gemma3_4b_text_encoder"
) -> str:
    """
    Convert Gemma3-4B-IT model to a single safetensors file.
    Config and tokenizer info stored in metadata.
    """
    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32
    }
    target_dtype = dtype_map[dtype]

    print(f"\n{'='*60}")
    print(f"Converting Gemma3-4B-IT to safetensors")
    print(f"{'='*60}")
    print(f"Source: {repo_path}")
    print(f"Output: {output_dir}")
    print(f"Dtype: {dtype}")

    # Load model directly to GPU
    print("\n[1/4] Loading Gemma model from HuggingFace...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(
        repo_path,
        torch_dtype=target_dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map=device
    )
    model.eval()

    # Load tokenizer to get its config
    print("[2/4] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(repo_path, trust_remote_code=True)

    # Get state dict (includes persistent buffers)
    print("[3/4] Extracting and converting state dict...")
    state_dict = model.state_dict()

    # Also add non-persistent buffers (like inv_freq, position_ids, etc.)
    print("   Adding non-persistent buffers...")
    buffer_count = 0
    for name, buf in model.named_buffers():
        if name not in state_dict:
            state_dict[name] = buf
            buffer_count += 1
            print(f"   + Added buffer: {name} shape={buf.shape}")
    print(f"   Added {buffer_count} non-persistent buffers")

    # Debug: show what we're saving
    print(f"   Total tensors: {len(state_dict)}")
    sample_keys = list(state_dict.keys())[:5]
    print(f"   Sample keys: {sample_keys}")

    # Check if vision_tower exists
    has_vision = any('vision' in k for k in state_dict.keys())
    has_language = any('language_model' in k for k in state_dict.keys())
    print(f"   Has vision_tower: {has_vision}")
    print(f"   Has language_model: {has_language}")

    # Convert all tensors to target dtype
    # BUT keep certain buffers in fp32 to preserve precision (inv_freq, embed_scale)
    converted_state_dict = {}
    keep_fp32_patterns = ['inv_freq', 'embed_scale']
    for key, tensor in state_dict.items():
        if tensor.is_floating_point():
            # Keep precision-sensitive buffers in fp32
            if any(pattern in key for pattern in keep_fp32_patterns):
                converted_state_dict[key] = tensor.to(torch.float32).contiguous()
                print(f"   Keeping {key} in fp32 for precision")
            else:
                converted_state_dict[key] = tensor.to(target_dtype).contiguous()
        else:
            converted_state_dict[key] = tensor.contiguous()

    # Create metadata with config and tokenizer info
    config_dict = model.config.to_dict()
    config_dict["torch_dtype"] = dtype
    config_dict["model_type_newbie"] = "gemma3_text_encoder"
    config_dict["_name_or_path"] = repo_path

    # Serialize full tokenizer to JSON (all files as base64)
    print("   Serializing tokenizer...")
    import tempfile
    import base64
    with tempfile.TemporaryDirectory() as tmp_dir:
        tokenizer.save_pretrained(tmp_dir)
        tokenizer_files = {}
        for filename in os.listdir(tmp_dir):
            filepath = os.path.join(tmp_dir, filename)
            if os.path.isfile(filepath):
                with open(filepath, 'rb') as f:
                    tokenizer_files[filename] = base64.b64encode(f.read()).decode('ascii')

    metadata = {
        "format": "newbie_text_encoder",
        "model_type": "gemma3_text_encoder",
        "original_repo": repo_path,
        "torch_dtype": dtype,
        "config": json.dumps(config_dict),
        "tokenizer_files": json.dumps(tokenizer_files),
    }

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save safetensors with metadata
    safetensors_path = os.path.join(output_dir, f"{model_name}.safetensors")
    print(f"[4/4] Saving to {safetensors_path}...")
    save_file(converted_state_dict, safetensors_path, metadata=metadata)

    # Print stats
    total_params = sum(p.numel() for p in converted_state_dict.values())
    total_size_mb = os.path.getsize(safetensors_path) / (1024**2)

    print(f"\n✅ Gemma conversion complete!")
    print(f"   Parameters: {total_params:,}")
    print(f"   File size: {total_size_mb:.1f} MB")
    print(f"   Output: {safetensors_path}")

    # Cleanup
    del model, state_dict, converted_state_dict
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return safetensors_path


def convert_jina_clip_to_safetensors(
    repo_path: str,
    output_dir: str,
    dtype: str = "bf16",
    model_name: str = "jina_clip_v2"
) -> str:
    """
    Convert Jina-CLIP-v2 model to a single safetensors file.
    Config and tokenizer info stored in metadata.
    """
    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32
    }
    target_dtype = dtype_map[dtype]

    print(f"\n{'='*60}")
    print(f"Converting Jina-CLIP-v2 to safetensors")
    print(f"{'='*60}")
    print(f"Source: {repo_path}")
    print(f"Output: {output_dir}")
    print(f"Dtype: {dtype}")

    # Load model with config modification to disable flash attention
    print("\n[1/4] Loading Jina CLIP model from HuggingFace...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = AutoConfig.from_pretrained(repo_path, trust_remote_code=True)
    config.use_flash_attn = False

    model = AutoModel.from_pretrained(
        repo_path,
        config=config,
        torch_dtype=target_dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map=device
    )
    model.eval()

    # Load tokenizer
    print("[2/4] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(repo_path, trust_remote_code=True)

    # Get state dict
    print("[3/4] Extracting and converting state dict...")
    state_dict = model.state_dict()

    # Convert all tensors to target dtype
    converted_state_dict = {}
    for key, tensor in state_dict.items():
        if tensor.is_floating_point():
            converted_state_dict[key] = tensor.to(target_dtype).contiguous()
        else:
            converted_state_dict[key] = tensor.contiguous()

    # Create metadata with config and tokenizer info
    config_dict = config.to_dict()

    # Recursively convert all non-serializable types (torch.dtype, etc.) to strings
    def make_json_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_json_serializable(v) for v in obj]
        elif isinstance(obj, torch.dtype):
            return str(obj).replace("torch.", "")
        elif hasattr(obj, 'dtype') and isinstance(obj.dtype, torch.dtype):
            # Handle tensor-like objects
            return str(obj)
        elif not isinstance(obj, (str, int, float, bool, type(None))):
            # Convert any other non-serializable type to string
            return str(obj)
        return obj

    config_dict = make_json_serializable(config_dict)
    config_dict["torch_dtype"] = dtype
    config_dict["model_type_newbie"] = "jina_clip_v2"
    config_dict["use_flash_attn"] = False
    config_dict["_name_or_path"] = repo_path

    # Serialize full tokenizer to JSON (all files as base64)
    print("   Serializing tokenizer...")
    import tempfile
    import base64
    with tempfile.TemporaryDirectory() as tmp_dir:
        tokenizer.save_pretrained(tmp_dir)
        tokenizer_files = {}
        for filename in os.listdir(tmp_dir):
            filepath = os.path.join(tmp_dir, filename)
            if os.path.isfile(filepath):
                with open(filepath, 'rb') as f:
                    tokenizer_files[filename] = base64.b64encode(f.read()).decode('ascii')

    metadata = {
        "format": "newbie_text_encoder",
        "model_type": "jina_clip_v2",
        "original_repo": repo_path,
        "torch_dtype": dtype,
        "config": json.dumps(config_dict),
        "tokenizer_files": json.dumps(tokenizer_files),
    }

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save safetensors with metadata
    safetensors_path = os.path.join(output_dir, f"{model_name}.safetensors")
    print(f"[4/4] Saving to {safetensors_path}...")
    save_file(converted_state_dict, safetensors_path, metadata=metadata)

    # Print stats
    total_params = sum(p.numel() for p in converted_state_dict.values())
    total_size_mb = os.path.getsize(safetensors_path) / (1024**2)

    print(f"\n✅ Jina CLIP conversion complete!")
    print(f"   Parameters: {total_params:,}")
    print(f"   File size: {total_size_mb:.1f} MB")
    print(f"   Output: {safetensors_path}")

    # Cleanup
    del model, state_dict, converted_state_dict
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return safetensors_path


def verify_gemma_conversion(
    output_dir: str,
    model_name: str = "gemma3_4b_text_encoder",
    device: str = "cuda"
) -> bool:
    """
    Verify that the converted Gemma model loads correctly.
    """
    from safetensors import safe_open

    print(f"\n{'='*60}")
    print(f"Verifying Gemma3 conversion")
    print(f"{'='*60}")

    safetensors_path = os.path.join(output_dir, f"{model_name}.safetensors")

    if not os.path.exists(safetensors_path):
        print(f"❌ File not found: {safetensors_path}")
        return False
    print(f"✓ Found: {safetensors_path}")

    # Load metadata
    print("\n[1/3] Loading metadata...")
    with safe_open(safetensors_path, framework="pt") as f:
        metadata = f.metadata()

    if not metadata:
        print("❌ No metadata found in safetensors file")
        return False

    config_dict = json.loads(metadata.get("config", "{}"))
    original_repo = metadata.get("original_repo", "google/gemma-3-4b-it")
    torch_dtype = metadata.get("torch_dtype", "bf16")

    print(f"   Original repo: {original_repo}")
    print(f"   Dtype: {torch_dtype}")

    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32
    }
    target_dtype = dtype_map.get(torch_dtype, torch.bfloat16)

    # Load tokenizer from original repo (needed for inference)
    print("[2/3] Loading tokenizer from original repo...")
    tokenizer = AutoTokenizer.from_pretrained(original_repo, trust_remote_code=True)
    tokenizer.padding_side = "right"

    # Simple verification - just check weights can be loaded
    print("[3/3] Verifying weights...")
    print(f"   Loading safetensors to {device}...")
    state_dict = load_file(safetensors_path, device=device)
    print(f"   Loaded {len(state_dict)} tensors")

    # Check a few tensors
    sample_keys = list(state_dict.keys())[:3]
    for key in sample_keys:
        tensor = state_dict[key]
        print(f"   {key}: shape={tensor.shape}, dtype={tensor.dtype}")

    # Quick sanity check - verify tensors are valid
    total_params = sum(t.numel() for t in state_dict.values())
    print(f"   Total parameters: {total_params:,}")

    del state_dict
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n✅ Gemma verification passed!")
    return True


def verify_jina_conversion(
    output_dir: str,
    model_name: str = "jina_clip_v2",
    device: str = "cuda"
) -> bool:
    """
    Verify that the converted Jina CLIP model loads correctly.
    """
    from safetensors import safe_open

    print(f"\n{'='*60}")
    print(f"Verifying Jina CLIP conversion")
    print(f"{'='*60}")

    safetensors_path = os.path.join(output_dir, f"{model_name}.safetensors")

    if not os.path.exists(safetensors_path):
        print(f"❌ File not found: {safetensors_path}")
        return False
    print(f"✓ Found: {safetensors_path}")

    # Load metadata
    print("\n[1/3] Loading metadata...")
    with safe_open(safetensors_path, framework="pt") as f:
        metadata = f.metadata()

    if not metadata:
        print("❌ No metadata found in safetensors file")
        return False

    original_repo = metadata.get("original_repo", "jinaai/jina-clip-v2")
    torch_dtype = metadata.get("torch_dtype", "bf16")

    print(f"   Original repo: {original_repo}")
    print(f"   Dtype: {torch_dtype}")

    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32
    }
    target_dtype = dtype_map.get(torch_dtype, torch.bfloat16)

    # Load tokenizer from original repo
    print("[2/3] Loading tokenizer from original repo...")
    tokenizer = AutoTokenizer.from_pretrained(original_repo, trust_remote_code=True)

    # Simple verification - just check weights can be loaded
    print("[3/3] Verifying weights...")
    print(f"   Loading safetensors to {device}...")
    state_dict = load_file(safetensors_path, device=device)
    print(f"   Loaded {len(state_dict)} tensors")

    # Check a few tensors
    sample_keys = list(state_dict.keys())[:3]
    for key in sample_keys:
        tensor = state_dict[key]
        print(f"   {key}: shape={tensor.shape}, dtype={tensor.dtype}")

    # Quick sanity check - verify tensors are valid
    total_params = sum(t.numel() for t in state_dict.values())
    print(f"   Total parameters: {total_params:,}")

    del state_dict
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n✅ Jina CLIP verification passed!")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Convert Gemma3-4B-IT and Jina-CLIP-v2 to safetensors format (2 files only)"
    )

    parser.add_argument(
        "--gemma-repo",
        type=str,
        default=None,
        help="Gemma model repo (e.g., 'google/gemma-3-4b-it' or local path)"
    )

    parser.add_argument(
        "--jina-repo",
        type=str,
        default=None,
        help="Jina CLIP model repo (e.g., 'jinaai/jina-clip-v2' or local path)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for converted models"
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="Target data type (default: bf16)"
    )

    parser.add_argument(
        "--gemma-name",
        type=str,
        default="gemma3_4b_text_encoder",
        help="Output name for Gemma safetensors file"
    )

    parser.add_argument(
        "--jina-name",
        type=str,
        default="jina_clip_v2",
        help="Output name for Jina CLIP safetensors file"
    )

    parser.add_argument(
        "--verify",
        action="store_true",
        help="Only verify existing converted models (skip conversion)"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for verification (default: cuda)"
    )

    args = parser.parse_args()

    # Verify mode
    if args.verify:
        print("Running verification only...")
        gemma_ok = verify_gemma_conversion(args.output_dir, args.gemma_name, device=args.device)
        jina_ok = verify_jina_conversion(args.output_dir, args.jina_name, device=args.device)

        if gemma_ok and jina_ok:
            print("\n✅ All verifications passed!")
        else:
            print("\n❌ Some verifications failed!")
            exit(1)
        return

    if not args.gemma_repo and not args.jina_repo:
        parser.error("At least one of --gemma-repo or --jina-repo must be specified (or use --verify)")

    results = {}

    # Convert Gemma
    if args.gemma_repo:
        try:
            results["gemma"] = convert_gemma_to_safetensors(
                repo_path=args.gemma_repo,
                output_dir=args.output_dir,
                dtype=args.dtype,
                model_name=args.gemma_name
            )
            verify_gemma_conversion(args.output_dir, args.gemma_name, args.device)
        except Exception as e:
            print(f"❌ Gemma conversion failed: {e}")
            import traceback
            traceback.print_exc()

    # Convert Jina CLIP
    if args.jina_repo:
        try:
            results["jina"] = convert_jina_clip_to_safetensors(
                repo_path=args.jina_repo,
                output_dir=args.output_dir,
                dtype=args.dtype,
                model_name=args.jina_name
            )
            verify_jina_conversion(args.output_dir, args.jina_name, args.device)
        except Exception as e:
            print(f"❌ Jina CLIP conversion failed: {e}")
            import traceback
            traceback.print_exc()

    # Print summary
    if results:
        print(f"\n{'='*60}")
        print("Conversion Summary - Only 2 safetensors files!")
        print(f"{'='*60}")
        for model_type, path in results.items():
            print(f"  {model_type.upper()}: {path}")


if __name__ == "__main__":
    main()

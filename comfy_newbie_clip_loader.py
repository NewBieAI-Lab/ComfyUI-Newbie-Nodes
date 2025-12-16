import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Any, Dict, Optional
import os
import sys
from safetensors.torch import load_file

try:
    from transformers import AutoModel, AutoTokenizer, AutoConfig, AutoProcessor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available")


class NewBieCLIP:

    def __init__(self, text_encoder, tokenizer, clip_model, clip_tokenizer, device="cuda", cpu_offload=False, processor=None, enable_jina_weights=True, weight_baseline_mode="compel", weight_strength=1.0, mask_normalization=True):
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.clip_model = clip_model
        self.clip_tokenizer = clip_tokenizer
        self.processor = processor
        self.device = device
        self.cpu_offload = cpu_offload
        self.original_device = device
        self.enable_jina_weights = enable_jina_weights
        self.weight_baseline_mode = weight_baseline_mode  # "mean", "empty", "compel", or "attn_mask"
        self.weight_strength = weight_strength  # Global weight strength multiplier
        self.mask_normalization = mask_normalization  # Whether to normalize mask after weight application
        
        self.text_encoder.eval()
        self.clip_model.eval()
        
        if self.cpu_offload:
            self.text_encoder = self.text_encoder.to("cpu")
            self.clip_model = self.clip_model.to("cpu")
            torch.cuda.empty_cache()
    
    def encode_from_tokens(self, tokens, return_pooled=False):
        if isinstance(tokens, list) and tokens and isinstance(tokens[0], tuple):
            return self.encode_token_weights([tokens])
        
        if self.cpu_offload:
            self.text_encoder = self.text_encoder.to(self.original_device)
            self.clip_model = self.clip_model.to(self.original_device)
        
        if isinstance(tokens, str):
            tokens = self.tokenize(tokens)
        
        with torch.no_grad():
            if hasattr(tokens, 'input_ids'):
                input_ids = tokens.input_ids
                attention_mask = tokens.attention_mask
            else:
                # 如果直接传入token ids
                input_ids = tokens
                attention_mask = torch.ones_like(input_ids)
            
            # 1. Gemma编码 - 主要的交叉注意力特征
            gemma_outputs = self.text_encoder(
                input_ids=input_ids.to(self.device),
                attention_mask=attention_mask.to(self.device),
                output_hidden_states=True,
            )
            # 使用倒数第二层作为主要特征 (cap_feats)
            cap_feats = gemma_outputs.hidden_states[-2]
            
            # 2. CLIP编码 - 用于时间嵌入的池化特征
            # 重新tokenize用于CLIP（因为tokenizer可能不同）
            batch_size = input_ids.shape[0]
            # 简化处理：使用相同的文本
            if hasattr(self, '_last_text'):
                clip_text = self._last_text
            else:
                # 从tokens恢复文本（简化版本）
                clip_text = [""] * batch_size
            
            clip_inputs = self.clip_tokenizer(
                clip_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=8000
            ).to(self.device)
            
            # Get pooled features - hook will capture hidden states if needed
            clip_text_embeddings = None  # Will be set by hook if available
            clip_text_pooled = self.clip_model.get_text_features(input_ids=clip_inputs.input_ids)
            
            # 3. 创建NextDiT兼容的条件字典，符合ComfyUI格式
            extra_conds = {
                "cap_feats": cap_feats,
                "cap_mask": attention_mask.to(self.device),
                "clip_text_pooled": clip_text_pooled,
            }
            
            if return_pooled:
                result = ([[cap_feats, extra_conds]], clip_text_pooled)
                if self.cpu_offload:
                    self.text_encoder = self.text_encoder.to("cpu")
                    self.clip_model = self.clip_model.to("cpu")
                    torch.cuda.empty_cache()
                return result
            
            # ComfyUI标准格式: [[features, extra_dict]]
            result = [[cap_feats, extra_conds]]
        
        if self.cpu_offload:
            self.text_encoder = self.text_encoder.to("cpu")
            self.clip_model = self.clip_model.to("cpu")
            torch.cuda.empty_cache()
        
        return result
    
    def encode_text(self, text):
        self._last_text = text if isinstance(text, list) else [text]
        tokens = self.tokenize(text)
        return self.encode_from_tokens(tokens)
    
    def encode_from_tokens_scheduled(self, tokens, return_pooled=False):
        return self.encode_from_tokens(tokens, return_pooled)
    
    def encode_token_weights(self, token_weight_pairs):
        to_encode = []
        token_weights_list = []
        char_weight_spans = []  # Store character-level weight spans
        max_token_len = 0
        has_weights = False

        print(f"\n[NewbieCLIP] ===== 开始权重处理 =====")

        # Parse weight spans from token_weight_pairs
        for x in token_weight_pairs:
            tokens = [a[0] for a in x]
            weights = [a[1] for a in x]

            # Reconstruct text and build character-level weight mapping
            text = " ".join(str(t) for t in tokens if str(t))
            text = text.replace('\x00\x02', '(').replace('\x00\x01', ')')

            # Build character spans with weights
            char_weights = []
            current_pos = 0
            for i, (token, weight) in enumerate(x):
                token_str = str(token).replace('\x00\x02', '(').replace('\x00\x01', ')')
                if token_str:  # Skip empty tokens
                    end_pos = current_pos + len(token_str)
                    char_weights.append((current_pos, end_pos, weight))
                    current_pos = end_pos
                    # Add space only if not last token and next token is not empty
                    if i < len(x) - 1:
                        next_token = str(x[i+1][0]).replace('\x00\x02', '(').replace('\x00\x01', ')')
                        if next_token:
                            current_pos += 1  # +1 for space

            char_weight_spans.append((text, char_weights))

            section_weights = [a for a in x if a[1] != 1.0]
            if section_weights:
                print(f"[NewbieCLIP] 检测到权重: {len(section_weights)}个token带权重")
                for token, weight in section_weights[:3]:
                    clean_token = str(token).replace('\x00\x02', '(').replace('\x00\x01', ')')
                    print(f"  - {clean_token}: {weight:.2f}")
                if len(section_weights) > 3:
                    print(f"  ... 还有{len(section_weights)-3}个token带权重")

            has_weights = has_weights or bool(section_weights)
            max_token_len = max(len(tokens), max_token_len)
            to_encode.append(tokens)
            token_weights_list.append(weights)

        # Count real sections before adding empty
        real_sections = len(to_encode)

        # Add empty section for weight interpolation if needed
        if has_weights or real_sections == 0:
            to_encode.append([""] * max_token_len)
            token_weights_list.append([1.0] * max_token_len)
            char_weight_spans.append(("", []))

        # Total sections including empty
        total_sections = len(to_encode)

        all_gemma_outputs = []
        all_clip_embeddings = []
        all_pooled = []
        all_attention_masks = []
        all_gemma_token_weights = []  # Store aligned token weights for Gemma
        all_clip_token_weights = []   # Store aligned token weights for Jina

        original_prompt = getattr(self, '_original_prompt', None)

        for i, (text, char_weights) in enumerate(char_weight_spans):
            if i == 0 and original_prompt:
                result = self._encode_text_direct_weighted_with_offsets(
                    text, char_weights, original_text=original_prompt
                )
            else:
                result = self._encode_text_direct_weighted_with_offsets(
                    text, char_weights
                )

            if isinstance(result, dict):
                cap_feats = result['cap_feats']
                attention_mask = result['cap_mask']
                pooled = result['clip_text_pooled']
                clip_embeddings = result.get('clip_text_embeddings')
                gemma_token_weights = result.get('gemma_token_weights')
                clip_token_weights = result.get('clip_token_weights')
            else:
                # Fallback for old format
                if isinstance(result, list) and result:
                    cap_feats = result[0][0]
                    extra_dict = result[0][1]
                    pooled = extra_dict.get("clip_text_pooled")
                    clip_embeddings = extra_dict.get("clip_text_embeddings")
                    attention_mask = extra_dict.get("cap_mask")
                else:
                    cap_feats = result
                    pooled = None
                    clip_embeddings = None
                    attention_mask = None
                gemma_token_weights = None
                clip_token_weights = None

            all_gemma_outputs.append(cap_feats)
            all_clip_embeddings.append(clip_embeddings)
            all_pooled.append(pooled)
            all_attention_masks.append(attention_mask)
            all_gemma_token_weights.append(gemma_token_weights)
            all_clip_token_weights.append(clip_token_weights)

        import torch

        if has_weights:
            weighted_gemma_outputs = []
            weighted_clip_outputs = []

            # Choose baseline mode for Gemma
            if self.weight_baseline_mode == "attn_mask":
                # Attention mask mode: modify cap_mask for DiT instead of changing embeddings
                print(f"[NewbieCLIP] 使用注意力掩码模式：保持embeddings不变，通过cap_mask控制DiT的注意力权重")

                # Keep original embeddings unchanged
                for k in range(real_sections):
                    weighted_gemma_outputs.append(all_gemma_outputs[k])

                # The mask modification will be done after concatenation

            elif self.weight_baseline_mode == "compel":
                # Compel mode: use mask to hide tokens instead of removing them
                print(f"[NewbieCLIP] Using Compel mode with masked attention for aligned interpolation")

                empty_gemma = all_gemma_outputs[-1]  # Empty baseline for weight > 1

                # Process with mask-based weight handling
                for k in range(real_sections):
                    gemma_emb = all_gemma_outputs[k].clone()
                    batch_size = gemma_emb.shape[0]
                    seq_len = gemma_emb.shape[1]

                    if all_gemma_token_weights[k] is not None:
                        token_weights = all_gemma_token_weights[k]

                        # Find tokens to mask (weight < 1)
                        tokens_to_mask = []
                        for j, w in enumerate(token_weights):
                            if w < 1.0 and w > 0.0:
                                tokens_to_mask.append(j)

                        # Generate masked version if needed
                        if tokens_to_mask:
                            text, _ = char_weight_spans[k]

                            # Create masked attention mask
                            original_mask = all_attention_masks[k] if k < len(all_attention_masks) else torch.ones(batch_size, seq_len, dtype=torch.long, device=self.device)
                            masked_attention = original_mask.clone()

                            # Set mask to 0 for low-weight tokens
                            for j in tokens_to_mask:
                                if j < masked_attention.shape[1]:
                                    masked_attention[:, j] = 0

                            print(f"  Section {k}: Masking {len(tokens_to_mask)} tokens for weight < 1")

                            # Re-encode with masked attention
                            with torch.no_grad():
                                tokens = self.tokenizer(
                                    [text],
                                    return_tensors="pt",
                                    padding=True,
                                    truncation=True,
                                    max_length=8000
                                )

                                input_ids = tokens.input_ids.to(self.device)

                                if self.cpu_offload:
                                    self.text_encoder = self.text_encoder.to(self.original_device)

                                # Encode with masked attention
                                gemma_outputs = self.text_encoder(
                                    input_ids=input_ids,
                                    attention_mask=masked_attention,
                                    output_hidden_states=True
                                )
                                gemma_without = gemma_outputs.hidden_states[-2]

                                if self.cpu_offload:
                                    self.text_encoder = self.text_encoder.to("cpu")
                                    torch.cuda.empty_cache()

                            # Verify alignment
                            if gemma_without.shape[1] != seq_len:
                                print(f"  Warning: Shape mismatch, padding/truncating")
                                if gemma_without.shape[1] < seq_len:
                                    pad_size = seq_len - gemma_without.shape[1]
                                    gemma_without = torch.cat([gemma_without, gemma_without[:, -1:].expand(-1, pad_size, -1)], dim=1)
                                else:
                                    gemma_without = gemma_without[:, :seq_len]

                            # Apply interpolation with aligned positions
                            for i in range(batch_size):
                                for j in range(min(seq_len, len(token_weights))):
                                    weight = token_weights[j]
                                    if weight != 1.0:
                                        if weight < 1.0 and weight > 0:
                                            # Positions are aligned!
                                            alpha = weight ** 0.7  # Smoother curve
                                            gemma_emb[i, j] = gemma_emb[i, j] * alpha + gemma_without[i, j] * (1 - alpha)
                                        elif weight <= 0:
                                            gemma_emb[i, j] = gemma_without[i, j]
                                        else:
                                            # weight > 1
                                            empty_token = empty_gemma[i, 0] if empty_gemma.shape[1] > 0 else torch.zeros_like(gemma_emb[i, j])
                                            scale = weight ** 0.8
                                            gemma_emb[i, j] = (gemma_emb[i, j] - empty_token) * scale + empty_token
                        else:
                            # No low-weight tokens, only handle weight > 1
                            for i in range(batch_size):
                                for j in range(min(seq_len, len(token_weights))):
                                    weight = token_weights[j]
                                    if weight > 1.0:
                                        empty_token = empty_gemma[i, 0] if empty_gemma.shape[1] > 0 else torch.zeros_like(gemma_emb[i, j])
                                        gemma_emb[i, j] = (gemma_emb[i, j] - empty_token) * weight + empty_token
                    else:
                        # Fallback when no aligned weights available
                        for i in range(batch_size):
                            for j in range(min(seq_len, len(token_weights_list[k]))):
                                weight = token_weights_list[k][j]
                                if weight != 1.0:
                                    if weight > 1.0:
                                        empty_token = empty_gemma[i, 0] if empty_gemma.shape[1] > 0 else torch.zeros_like(gemma_emb[i, j])
                                        gemma_emb[i, j] = (gemma_emb[i, j] - empty_token) * weight + empty_token
                                    # For weight < 1 in fallback, just scale down
                                    else:
                                        gemma_emb[i, j] = gemma_emb[i, j] * weight

                    weighted_gemma_outputs.append(gemma_emb)

            elif self.weight_baseline_mode == "mean":
                # Mean baseline mode: generate unweighted baseline
                print(f"[NewbieCLIP] Gemma使用均值baseline模式")
                unweighted_gemma_outputs = []

                # Generate unweighted versions (all weights = 1.0)
                for i, (text, _) in enumerate(char_weight_spans[:real_sections]):
                    # Use the same text but without weights
                    result = self._encode_text_direct_weighted_with_offsets(text, [])

                    if isinstance(result, dict):
                        unweighted_gemma = result['cap_feats']
                    else:
                        # Fallback
                        unweighted_gemma = all_gemma_outputs[i]

                    unweighted_gemma_outputs.append(unweighted_gemma)

                # Process with mean baseline
                for k in range(real_sections):
                    gemma_emb = all_gemma_outputs[k].clone()
                    unweighted_gemma = unweighted_gemma_outputs[k]

                    batch_size = gemma_emb.shape[0]
                    seq_len = gemma_emb.shape[1]

                    # Calculate context mean μ for this section
                    context_mean = unweighted_gemma.mean(dim=1, keepdim=True)  # [batch_size, 1, hidden_dim]

                    # Apply mean-relative scaling for Gemma
                    # emb' = (emb_ref - μ) * w + μ
                    if all_gemma_token_weights[k] is not None:
                        token_weights = all_gemma_token_weights[k]
                        for i in range(batch_size):
                            for j in range(min(seq_len, len(token_weights))):
                                weight = token_weights[j]
                                if weight != 1.0:
                                    # Mean-relative scaling
                                    gemma_emb[i, j] = (unweighted_gemma[i, j] - context_mean[i, 0]) * weight + context_mean[i, 0]
                    else:
                        # Fallback to old method with token_weights_list
                        for i in range(batch_size):
                            for j in range(min(seq_len, len(token_weights_list[k]))):
                                weight = token_weights_list[k][j]
                                if weight != 1.0:
                                    # Mean-relative scaling
                                    gemma_emb[i, j] = (unweighted_gemma[i, j] - context_mean[i, 0]) * weight + context_mean[i, 0]

                    weighted_gemma_outputs.append(gemma_emb)

            else:  # empty baseline mode
                print(f"[NewbieCLIP] Gemma使用空字符串baseline模式")
                empty_gemma = all_gemma_outputs[-1]  # Last one is empty

                # Process with empty baseline
                for k in range(real_sections):
                    gemma_emb = all_gemma_outputs[k].clone()

                    batch_size = gemma_emb.shape[0]
                    seq_len = gemma_emb.shape[1]

                    # Apply empty baseline interpolation
                    if all_gemma_token_weights[k] is not None:
                        token_weights = all_gemma_token_weights[k]
                        for i in range(batch_size):
                            for j in range(min(seq_len, len(token_weights))):
                                weight = token_weights[j]
                                if weight != 1.0:
                                    empty_token = empty_gemma[i, 0] if empty_gemma.shape[1] > 0 else torch.zeros_like(gemma_emb[i, j])
                                    gemma_emb[i, j] = (gemma_emb[i, j] - empty_token) * weight + empty_token
                    else:
                        # Fallback to old method with token_weights_list
                        for i in range(batch_size):
                            for j in range(min(seq_len, len(token_weights_list[k]))):
                                weight = token_weights_list[k][j]
                                if weight != 1.0:
                                    empty_token = empty_gemma[i, 0] if empty_gemma.shape[1] > 0 else torch.zeros_like(gemma_emb[i, j])
                                    gemma_emb[i, j] = (gemma_emb[i, j] - empty_token) * weight + empty_token

                    weighted_gemma_outputs.append(gemma_emb)

                if self.enable_jina_weights and all_clip_embeddings[k] is not None:
                    clip_emb = all_clip_embeddings[k].clone()
                    empty_clip = all_clip_embeddings[-1] if all_clip_embeddings[-1] is not None else torch.zeros_like(clip_emb)

                    # Use aligned token weights if available
                    if all_clip_token_weights[k] is not None:
                        token_weights = all_clip_token_weights[k]
                        for i in range(clip_emb.shape[0]):
                            for j in range(min(clip_emb.shape[1], len(token_weights))):
                                weight = token_weights[j]
                                if weight != 1.0:
                                    empty_token = empty_clip[i, 0] if empty_clip.shape[1] > 0 else torch.zeros_like(clip_emb[i, j])
                                    clip_emb[i, j] = (clip_emb[i, j] - empty_token) * weight + empty_token
                    else:
                        # Fallback to old method
                        for i in range(clip_emb.shape[0]):
                            for j in range(min(clip_emb.shape[1], len(token_weights_list[k]))):
                                weight = token_weights_list[k][j]
                                if weight != 1.0:
                                    empty_token = empty_clip[i, 0] if empty_clip.shape[1] > 0 else torch.zeros_like(clip_emb[i, j])
                                    clip_emb[i, j] = (clip_emb[i, j] - empty_token) * weight + empty_token
                    weighted_clip_outputs.append(clip_emb)
                else:
                    weighted_clip_outputs.append(all_clip_embeddings[k])

            # Concatenate all real weighted outputs (no need to exclude anything)
            if len(weighted_gemma_outputs) > 1:
                final_gemma = torch.cat(weighted_gemma_outputs, dim=1)
            else:
                final_gemma = weighted_gemma_outputs[0] if weighted_gemma_outputs else all_gemma_outputs[0]

            mask_list = []
            # Build mask for real sections only
            for k in range(real_sections):
                if k < len(all_attention_masks) and all_attention_masks[k] is not None:
                    mask_list.append(all_attention_masks[k])
                elif k < len(weighted_gemma_outputs):
                    batch_size = weighted_gemma_outputs[k].shape[0]
                    seq_len = weighted_gemma_outputs[k].shape[1]
                    mask_list.append(torch.ones(batch_size, seq_len, dtype=torch.long, device=self.device))

            final_attention_mask = torch.cat(mask_list, dim=1) if len(mask_list) > 1 else mask_list[0] if mask_list else None

            if weighted_clip_outputs and weighted_clip_outputs[0] is not None:
                # Concatenate all real weighted clip outputs
                clip_to_concat = [c for c in weighted_clip_outputs if c is not None]
                weighted_clip_concat = torch.cat(clip_to_concat, dim=1) if len(clip_to_concat) > 1 else clip_to_concat[0] if clip_to_concat else None

                if weighted_clip_concat is not None:
                    # Use aligned clip token weights if available
                    if any(w is not None for w in all_clip_token_weights[:real_sections]):
                        weight_list = []
                        for k in range(real_sections):
                            if k < len(all_clip_token_weights) and all_clip_token_weights[k] is not None:
                                weight_list.extend(all_clip_token_weights[k])
                            elif k < len(token_weights_list):
                                # Fallback to original weights
                                weight_list.extend(token_weights_list[k][:len(token_weights_list[k])])
                    else:
                        # Fallback to original method
                        weight_list = []
                        for k in range(real_sections):
                            if k < len(token_weights_list):
                                weight_list.extend(token_weights_list[k][:len(token_weights_list[k])])

                    weight_tensor = torch.tensor(
                        weight_list[:weighted_clip_concat.shape[1]],  # Ensure correct length
                        dtype=weighted_clip_concat.dtype,
                        device=weighted_clip_concat.device
                    )

                    # Pad if necessary
                    if len(weight_tensor) < weighted_clip_concat.shape[1]:
                        padding = torch.ones(
                            weighted_clip_concat.shape[1] - len(weight_tensor),
                            dtype=weight_tensor.dtype,
                            device=weight_tensor.device
                        )
                        weight_tensor = torch.cat([weight_tensor, padding])

                    weight_tensor = weight_tensor.unsqueeze(0).unsqueeze(-1)
                    weight_tensor = weight_tensor.expand_as(weighted_clip_concat)

                    weighted_sum = (weighted_clip_concat * weight_tensor).sum(dim=1)
                    weight_sum = weight_tensor.sum(dim=1).squeeze(-1)
                    clip_pooled_final = weighted_sum / (weight_sum + 1e-8)
                else:
                    clip_pooled_final = all_pooled[0] if all_pooled[0] is not None else None
                print(f"[NewbieCLIP] Jina CLIP: 使用加权池化")
            else:
                clip_pooled_final = all_pooled[0] if all_pooled[0] is not None else None
        else:
            final_gemma = all_gemma_outputs[0] if all_gemma_outputs else None
            final_attention_mask = all_attention_masks[0] if all_attention_masks else None
            clip_pooled_final = all_pooled[0] if all_pooled else None


        print(f"[NewbieCLIP] 权重处理完成")
        if has_weights:
            if self.weight_baseline_mode == "mean":
                print(f"  - Gemma: 使用均值相对缩放 (emb_ref - μ) * w + μ")
            elif self.weight_baseline_mode == "compel":
                print(f"  - Gemma: Compel模式 (w<1: tan插值到无权重, w>1: 空baseline放大)")
            else:
                print(f"  - Gemma: 使用空字符串baseline插值")
            print(f"  - CLIP: 使用空字符串baseline插值")
        print(f"  - Gemma输出: shape={final_gemma.shape if final_gemma is not None else 'None'}")
        print(f"  - Attention mask: shape={final_attention_mask.shape if final_attention_mask is not None else 'None'}")
        print(f"  - Jina pooled: shape={clip_pooled_final.shape if clip_pooled_final is not None else 'None'}")
        print(f"[NewbieCLIP] ===== 权重处理结束 =====\n")

        extra_conds = {
            "cap_feats": final_gemma,
            "cap_mask": final_attention_mask,
            "clip_text_pooled": clip_pooled_final,
        }

        return [[final_gemma, extra_conds]]
    
    def tokenize(self, text):
        import re
        if re.search(r'[()\[\]{}]', text):
            return self._parse_weights_extended(text)
        
        if isinstance(text, str):
            text = [text]
        
        return self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=8000)
    
    def _parse_weights_extended(self, text):
        from comfy.sd1_clip import escape_important, token_weights
        import re
        
        text = text.replace('\\[', '<<<ESC_LBRACKET>>>')
        text = text.replace('\\]', '<<<ESC_RBRACKET>>>')
        text = text.replace('\\{', '<<<ESC_LBRACE>>>')
        text = text.replace('\\}', '<<<ESC_RBRACE>>>')
        text = text.replace('\\(', '<<<ESC_LPAREN>>>')
        text = text.replace('\\)', '<<<ESC_RPAREN>>>')
        def process_brackets(text):
            pattern = r'\[+([^\[\]]*(?:<<<ESC_[LR]BRACKET>>>[^\[\]]*)*)\]+'
            
            def replace_brackets(match):
                full_match = match.group(0)
                content = match.group(1)
                
                left_count = 0
                i = 0
                while i < len(full_match) and full_match[i] == '[':
                    left_count += 1
                    i += 1
                
                weight = 0.9 ** left_count
                return f"({content}:{weight:.3f})"
            
            while re.search(pattern, text):
                text = re.sub(pattern, replace_brackets, text)
            return text
        
        def process_braces(text):
            pattern = r'\{+([^{}]*(?:<<<ESC_[LR]BRACE>>>[^{}]*)*)\}+'
            
            def replace_braces(match):
                full_match = match.group(0)
                content = match.group(1)
                
                left_count = 0
                i = 0
                while i < len(full_match) and full_match[i] == '{':
                    left_count += 1
                    i += 1
                
                weight = 1.2 ** left_count
                return f"({content}:{weight:.3f})"
            
            while re.search(pattern, text):
                text = re.sub(pattern, replace_braces, text)
            return text
        
        text = process_brackets(text)
        text = process_braces(text)
        text = text.replace('<<<ESC_LBRACKET>>>', '\\[')
        text = text.replace('<<<ESC_RBRACKET>>>', '\\]')
        text = text.replace('<<<ESC_LBRACE>>>', '\\{')
        text = text.replace('<<<ESC_RBRACE>>>', '\\}')
        text = text.replace('<<<ESC_LPAREN>>>', '\\(')
        text = text.replace('<<<ESC_RPAREN>>>', '\\)')
        
        return token_weights(escape_important(text), 1.0)
    
    def _encode_text_direct(self, text, original_text=None):
        if isinstance(text, str):
            text = [text]

        self._last_text = text

        if self.cpu_offload:
            self.text_encoder = self.text_encoder.to(self.original_device)
            self.clip_model = self.clip_model.to(self.original_device)

        with torch.no_grad():
            # Use original_text for CLIP if provided, otherwise use text
            clip_text = original_text if original_text is not None else text
            if isinstance(clip_text, str):
                clip_text = [clip_text]

            # Gemma processes full text with system prompt
            tokens = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=8000
            )

            input_ids = tokens.input_ids.to(self.device)
            attention_mask = tokens.attention_mask.to(self.device)

            gemma_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            cap_feats = gemma_outputs.hidden_states[-2]

            # CLIP processes only the original user prompt
            clip_inputs = self.clip_tokenizer(clip_text, return_tensors="pt", padding=True, truncation=True, max_length=8000).to(self.device)
            # Get pooled features - hook will capture hidden states if needed
            clip_text_embeddings = None  # Will be set by hook if available
            clip_text_pooled = self.clip_model.get_text_features(input_ids=clip_inputs.input_ids)

            extra_conds = {
                "cap_feats": cap_feats,
                "cap_mask": attention_mask,
                "clip_text_pooled": clip_text_pooled,
            }

            result = [[cap_feats, extra_conds]]

        if self.cpu_offload:
            self.text_encoder = self.text_encoder.to("cpu")
            self.clip_model = self.clip_model.to("cpu")
            torch.cuda.empty_cache()

        return result

    def _encode_text_direct_weighted(self, text, original_text=None):
        """Enhanced version that returns CLIP embeddings for weight processing"""
        if isinstance(text, str):
            text = [text]

        self._last_text = text

        if self.cpu_offload:
            self.text_encoder = self.text_encoder.to(self.original_device)
            self.clip_model = self.clip_model.to(self.original_device)

        with torch.no_grad():
            clip_text = original_text if original_text is not None else text
            if isinstance(clip_text, str):
                clip_text = [clip_text]

            tokens = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=8000
            )

            input_ids = tokens.input_ids.to(self.device)
            attention_mask = tokens.attention_mask.to(self.device)

            gemma_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            cap_feats = gemma_outputs.hidden_states[-2]

            clip_inputs = self.clip_tokenizer(clip_text, return_tensors="pt", padding=True, truncation=True, max_length=8000).to(self.device)

            clip_text_embeddings = None

            if self.enable_jina_weights:
                if hasattr(self.clip_model, '_last_hidden_states'):
                    self.clip_model._last_hidden_states = None

            clip_text_pooled = self.clip_model.get_text_features(input_ids=clip_inputs.input_ids)

            if self.enable_jina_weights and hasattr(self.clip_model, '_last_hidden_states') and self.clip_model._last_hidden_states is not None:
                clip_text_embeddings = self.clip_model._last_hidden_states
                print(f"[NewbieCLIP] Jina CLIP: Using hook to capture token embeddings, shape={clip_text_embeddings.shape}")
            elif not self.enable_jina_weights:
                print(f"[NewbieCLIP] Jina权重处理已禁用")

            extra_conds = {
                "cap_feats": cap_feats,
                "cap_mask": attention_mask,
                "clip_text_pooled": clip_text_pooled,
                "clip_text_embeddings": clip_text_embeddings,
            }

            result = [[cap_feats, extra_conds]]

        if self.cpu_offload:
            self.text_encoder = self.text_encoder.to("cpu")
            self.clip_model = self.clip_model.to("cpu")
            torch.cuda.empty_cache()

        return result

    def _encode_text_direct_weighted_with_offsets(self, text, char_weights, original_text=None):
        """Enhanced version with offset mapping for precise weight alignment"""
        if isinstance(text, str):
            text = [text]

        self._last_text = text

        if self.cpu_offload:
            self.text_encoder = self.text_encoder.to(self.original_device)
            self.clip_model = self.clip_model.to(self.original_device)

        with torch.no_grad():
            clip_text = original_text if original_text is not None else text
            if isinstance(clip_text, str):
                clip_text = [clip_text]

            # Tokenize with offset mapping for Gemma
            tokens = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=8000,
                return_offsets_mapping=True
            )

            input_ids = tokens.input_ids.to(self.device)
            attention_mask = tokens.attention_mask.to(self.device)

            # Align weights using offset mapping for Gemma
            gemma_token_weights = None
            if hasattr(tokens, 'offset_mapping') and tokens.offset_mapping is not None:
                gemma_token_weights = self._align_weights_with_offsets(
                    tokens.offset_mapping[0], char_weights, text[0]
                )
                print(f"[NewbieCLIP] Gemma: 对齐了{len(gemma_token_weights)}个token权重")

            # Process Gemma
            gemma_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            cap_feats = gemma_outputs.hidden_states[-2]

            # Tokenize with offset mapping for CLIP
            clip_inputs = self.clip_tokenizer(
                clip_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=8000,
                return_offsets_mapping=True
            ).to(self.device)

            # Align weights for CLIP
            clip_token_weights = None
            if hasattr(clip_inputs, 'offset_mapping') and clip_inputs.offset_mapping is not None:
                clip_token_weights = self._align_weights_with_offsets(
                    clip_inputs.offset_mapping[0], char_weights, clip_text[0]
                )
                print(f"[NewbieCLIP] Jina CLIP: 对齐了{len(clip_token_weights)}个token权重")

            clip_text_embeddings = None

            if self.enable_jina_weights:
                if hasattr(self.clip_model, '_last_hidden_states'):
                    self.clip_model._last_hidden_states = None

            clip_text_pooled = self.clip_model.get_text_features(input_ids=clip_inputs.input_ids)

            if self.enable_jina_weights and hasattr(self.clip_model, '_last_hidden_states') and self.clip_model._last_hidden_states is not None:
                clip_text_embeddings = self.clip_model._last_hidden_states
                print(f"[NewbieCLIP] Jina CLIP: Using hook to capture token embeddings, shape={clip_text_embeddings.shape}")

            result = {
                "cap_feats": cap_feats,
                "cap_mask": attention_mask,
                "clip_text_pooled": clip_text_pooled,
                "clip_text_embeddings": clip_text_embeddings,
                "gemma_token_weights": gemma_token_weights,
                "clip_token_weights": clip_token_weights
            }

        if self.cpu_offload:
            self.text_encoder = self.text_encoder.to("cpu")
            self.clip_model = self.clip_model.to("cpu")
            torch.cuda.empty_cache()

        return result

    def _align_weights_with_offsets(self, offset_mapping, char_weights, text):
        """Align character-level weights to token-level using offset mapping"""
        token_weights = []

        for token_start, token_end in offset_mapping:
            if token_start == token_end:  # Special tokens like [PAD]
                token_weights.append(1.0)
                continue

            # Find overlapping weight spans
            weight_sum = 0.0
            overlap_sum = 0

            for char_start, char_end, weight in char_weights:
                # Calculate overlap between token span and weight span
                overlap_start = max(token_start, char_start)
                overlap_end = min(token_end, char_end)

                if overlap_start < overlap_end:
                    overlap_len = overlap_end - overlap_start
                    weight_sum += weight * overlap_len
                    overlap_sum += overlap_len

            # Average weight for the token based on character overlap
            if overlap_sum > 0:
                token_weights.append(weight_sum / overlap_sum)
            else:
                token_weights.append(1.0)  # Default weight

        return token_weights

    def encode_with_image(self, user_text, image=None, system_text=""):
        if self.cpu_offload:
            self.text_encoder = self.text_encoder.to(self.original_device)
            self.clip_model = self.clip_model.to(self.original_device)
        
        if not hasattr(self, 'processor') or self.processor is None:
            return self.encode_text(user_text)
        
        with torch.no_grad():
            if image is not None:
                # 构建正确的消息格式
                messages = []
                
                if system_text and system_text.strip():
                    messages.append({
                        "role": "system",
                        "content": [{"type": "text", "text": system_text.strip()}]
                    })
                
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": user_text}
                    ]
                })
                
                inputs = self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt"
                ).to(self.device)
                
                gemma_outputs = self.text_encoder(
                    input_ids=inputs.input_ids,
                    pixel_values=inputs.pixel_values if hasattr(inputs, 'pixel_values') else None,
                    attention_mask=inputs.attention_mask,
                    output_hidden_states=True,
                )
                
                cap_feats = gemma_outputs.hidden_states[-2]
                attention_mask = inputs.attention_mask
            else:
                text = [user_text] if isinstance(user_text, str) else user_text
                tokens = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=8000).to(self.device)
                gemma_outputs = self.text_encoder(
                    input_ids=tokens.input_ids,
                    attention_mask=tokens.attention_mask,
                    output_hidden_states=True,
                )
                cap_feats = gemma_outputs.hidden_states[-2]
                attention_mask = tokens.attention_mask
            
            text_for_clip = [user_text] if isinstance(user_text, str) else user_text
            clip_inputs = self.clip_tokenizer(text_for_clip, return_tensors="pt", padding=True, truncation=True, max_length=8000).to(self.device)
            # Get pooled features - hook will capture hidden states if needed
            clip_text_embeddings = None  # Will be set by hook if available
            clip_text_pooled = self.clip_model.get_text_features(input_ids=clip_inputs.input_ids)
            
            extra_conds = {
                "cap_feats": cap_feats,
                "cap_mask": attention_mask,
                "clip_text_pooled": clip_text_pooled,
            }
            
            result = [[cap_feats, extra_conds]]
        
        if self.cpu_offload:
            self.text_encoder = self.text_encoder.to("cpu")
            self.clip_model = self.clip_model.to("cpu")
            torch.cuda.empty_cache()
        
        return result

    def get_clip_features(self, text):
        if self.cpu_offload:
            self.clip_model = self.clip_model.to(self.original_device)
        
        if isinstance(text, str):
            text = [text]
            
        with torch.no_grad():
            clip_inputs = self.clip_tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=8000
            ).to(self.device)
            
            # Get pooled features - hook will capture hidden states if needed
            clip_text_embeddings = None  # Will be set by hook if available
            clip_text_pooled = self.clip_model.get_text_features(input_ids=clip_inputs.input_ids)
        
        if self.cpu_offload:
            self.clip_model = self.clip_model.to("cpu")
            torch.cuda.empty_cache()
            
        return clip_text_pooled


class NewBieCLIPLoader:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gemma_model_path": ("STRING", {
                    "default": "",
                    "description": "Path to Gemma3-4B-IT safetensors directory"
                }),
                "jina_clip_path": ("STRING", {
                    "default": "jinaai/jina-clip-v2",
                    "description": "Path to Jina CLIP model (HuggingFace name or local path)"
                }),
            },
            "optional": {
                "device": (["cuda", "cpu"], {
                    "default": "cuda"
                }),
                "dtype": (["bf16", "fp16", "fp32"], {
                    "default": "bf16"
                }),
                "cpu_offload": ("BOOLEAN", {
                    "default": False,
                    "description": "Offload models to CPU memory after inference"
                }),
                "enable_jina_weights": ("BOOLEAN", {
                    "default": True,
                    "description": "Enable weight processing for Jina CLIP (requires more memory)"
                }),
                "weight_baseline_mode": (["compel", "mean", "empty", "attn_bias", "hybrid"], {
                    "default": "compel",
                    "description": "Weight baseline mode: 'compel' for advanced, 'mean' for context-aware, 'empty' for traditional, 'attn_bias' for attention-based, 'hybrid' for combined approach"
                }),
                "weight_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "description": "Global weight strength multiplier (0=no effect, 1=normal, 2=double strength)"
                }),
                "mask_normalization": ("BOOLEAN", {
                    "default": True,
                    "description": "Normalize attention mask after weight application to maintain distribution"
                }),
            }
        }
    
    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"
    CATEGORY = "loaders"
    TITLE = "NewBie CLIP Loader"

    def load_gemma_model(self, model_path: str, device: str, dtype: torch.dtype):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library is required")
        
        if not os.path.exists(model_path):
            try:
                print(f"Loading Gemma model from HuggingFace: {model_path}")
                text_encoder = AutoModel.from_pretrained(
                    model_path,
                    torch_dtype=dtype,
                    device_map=device,
                    trust_remote_code=True
                )
                tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=True
                )
                tokenizer.padding_side = "right"
                try:
                    processor = AutoProcessor.from_pretrained(
                        model_path,
                        trust_remote_code=True
                    )
                except:
                    processor = None
                return text_encoder, tokenizer, processor
            except Exception as e:
                raise FileNotFoundError(f"Cannot load Gemma model from {model_path}: {e}")
        
        print(f"Loading Gemma model from local path: {model_path}")
        
        try:
            text_encoder = AutoModel.from_pretrained(
                model_path,
                torch_dtype=dtype,
                device_map=device,
                trust_remote_code=True,
                local_files_only=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=True,
            )
            tokenizer.padding_side = "right"
            try:
                processor = AutoProcessor.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    local_files_only=True,
                )
            except:
                processor = None
            return text_encoder, tokenizer, processor
            
        except Exception as e:
            print(f"Direct loading failed: {e}")
            print("Falling back to manual loading...")
            
            config_path = os.path.join(model_path, "config.json")
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"config.json not found in {model_path}")
            
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            tokenizer.padding_side = "right"
            try:
                processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            except:
                processor = None
            
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            text_encoder = AutoModel.from_config(config, torch_dtype=torch.float32)
            
            safetensors_path = os.path.join(model_path, "model.safetensors")
            if os.path.exists(safetensors_path):
                state_dict = load_file(safetensors_path, device="cpu")
            else:
                import glob
                shard_files = glob.glob(os.path.join(model_path, "model-*.safetensors"))
                if shard_files:
                    state_dict = {}
                    for shard_file in sorted(shard_files):
                        shard_dict = load_file(shard_file, device="cpu")
                        state_dict.update(shard_dict)
                else:
                    raise FileNotFoundError(f"No safetensors files found in {model_path}")
            
            if any(key.startswith("language_model.model.") for key in state_dict.keys()):
                state_dict = {
                    key.replace("language_model.model.", "language_model.") if key.startswith("language_model.model.") else key: value
                    for key, value in state_dict.items()
                }
            
            missing_keys, unexpected_keys = text_encoder.load_state_dict(state_dict, strict=False)
            if missing_keys:
                print(f"Missing keys: {len(missing_keys)}")
            if unexpected_keys:
                print(f"Unexpected keys: {len(unexpected_keys)}")
            
            text_encoder = text_encoder.to(device=device, dtype=dtype)
            
            return text_encoder, tokenizer, processor

    def load_jina_clip(self, model_path: str, device: str, dtype: torch.dtype, enable_jina_weights: bool = True):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library is required")

        print(f"Loading Jina CLIP model: {model_path}")
        print(f"Device: {device}, Dtype: {dtype}")

        try:
            # Standard loading with trust_remote_code
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            config.use_flash_attn = False
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            clip_model = AutoModel.from_pretrained(
                model_path,
                config=config,
                torch_dtype=dtype,
                device_map=device,
                trust_remote_code=True
            )

            # Add hook to capture hidden states if weight processing is enabled
            if enable_jina_weights and hasattr(clip_model, 'text_model'):
                print(f"[NewbieCLIP] Installing hook to capture hidden states")
                self._install_hidden_state_hook(clip_model, enable_jina_weights)

            return clip_model, tokenizer

        except Exception as e:
            print(f"Error in load_jina_clip: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _install_hidden_state_hook(self, clip_model, enable_jina_weights=True):
        """Install forward hook to capture hidden states from text_model"""
        if not enable_jina_weights:
            return

        clip_model._last_hidden_states = None

        def hook_fn(module, input, output):
            if hasattr(output, 'last_hidden_state'):
                clip_model._last_hidden_states = output.last_hidden_state
                print(f"[NewbieCLIP] Hook: Captured hidden states shape={output.last_hidden_state.shape}")
            elif isinstance(output, tuple):
                for out in output:
                    if isinstance(out, torch.Tensor) and out.dim() == 3:
                        clip_model._last_hidden_states = out
                        print(f"[NewbieCLIP] Hook: Captured hidden states from tuple, shape={out.shape}")
                        break

        if hasattr(clip_model.text_model, 'transformer'):
            handle = clip_model.text_model.transformer.register_forward_hook(hook_fn)
            clip_model._hook_handle = handle
            print(f"[NewbieCLIP] Hook registered on text_model.transformer")
        elif hasattr(clip_model.text_model, 'encoder'):
            handle = clip_model.text_model.encoder.register_forward_hook(hook_fn)
            clip_model._hook_handle = handle
            print(f"[NewbieCLIP] Hook registered on text_model.encoder")
        else:
            handle = clip_model.text_model.register_forward_hook(hook_fn)
            clip_model._hook_handle = handle
            print(f"[NewbieCLIP] Hook registered on text_model directly")

    def load_clip(
        self,
        gemma_model_path: str,
        jina_clip_path: str,
        device: str = "cuda",
        dtype: str = "bf16",
        cpu_offload: bool = False,
        enable_jina_weights: bool = True,
        weight_baseline_mode: str = "mean",
        weight_strength: float = 1.0,
        mask_normalization: bool = True
    ) -> Tuple[Any,]:
        dtype_map = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32
        }
        torch_dtype = dtype_map[dtype]
        
        print(f"Loading NewBie CLIP models...")
        print(f"Gemma path: {gemma_model_path}")
        print(f"Jina CLIP path: {jina_clip_path}")
        print(f"Device: {device}, Dtype: {dtype}")
        
        text_encoder, tokenizer, processor = self.load_gemma_model(
            gemma_model_path, device, torch_dtype
        )
        
        clip_model, clip_tokenizer = self.load_jina_clip(
            jina_clip_path, device, torch_dtype, enable_jina_weights
        )
        
        newbie_clip = NewBieCLIP(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            clip_model=clip_model,
            clip_tokenizer=clip_tokenizer,
            device=device,
            cpu_offload=cpu_offload,
            processor=processor,
            enable_jina_weights=enable_jina_weights,
            weight_baseline_mode=weight_baseline_mode,
            weight_strength=weight_strength,
            mask_normalization=mask_normalization
        )
        
        return (newbie_clip,)


NODE_CLASS_MAPPINGS = {
    "NewBieCLIPLoader": NewBieCLIPLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NewBieCLIPLoader": "NewBie CLIP Loader",
}
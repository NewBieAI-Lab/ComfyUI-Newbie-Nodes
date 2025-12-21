"""
NewBie Gemma Chat Node
Enables chat/text generation using the Gemma model loaded in NewBie CLIP.
"""

import torch
from typing import Tuple, Any, Optional

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available for Gemma Chat")


# Cache for the generation model to avoid reloading
_gemma_gen_cache = {
    "model": None,
    "tokenizer": None,
    "model_path": None,
    "device": None,
    "dtype": None,
}


def _clear_gemma_cache():
    """Clear the cached Gemma generation model to free VRAM"""
    global _gemma_gen_cache
    if _gemma_gen_cache["model"] is not None:
        print("[NewBie Gemma Chat] Unloading generation model to free VRAM...")
        del _gemma_gen_cache["model"]
        del _gemma_gen_cache["tokenizer"]
        _gemma_gen_cache["model"] = None
        _gemma_gen_cache["tokenizer"] = None
        _gemma_gen_cache["model_path"] = None
        _gemma_gen_cache["device"] = None
        _gemma_gen_cache["dtype"] = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print("[NewBie Gemma Chat] VRAM cleared")


class NewBieGemmaChat:
    """
    Chat with the Gemma model that's used in NewBie CLIP.
    This node loads a separate instance of Gemma for text generation.
    """
    
    default_sys_prompt = """
You are a Danbooru-to-XML prompt converter for AI image generation.

INPUT: Comma-separated Danbooru tags
OUTPUT: XML prompt in the exact format below

STRICT RULES:
1. PRESERVE tags exactly as given - do not modify, translate, or rephrase
2. ESCAPE parentheses with backslash: ( becomes \( and ) becomes \)
3. Character names go in <n> tags exactly as written (e.g., hatsune_miku, rem_\(re:zero\))
4. If no character name is specified, keep <n></n> empty but do not delete it
5. For single character, omit <character_2> entirely
6. Artist tags (artist:name or by_artist) go in <artists> as: artist:exact_name
7. Delete <clothing> tag entirely if no clothing specified
8. The <caption> section uses natural language WITHOUT underscores - convert blue_hair to "blue hair" etc.

CLASSIFICATION GUIDE:
- Gender: 1girl, 1boy, 2girls, multiple_boys, etc.
- Appearance: hair color, eye color, body features (blue_hair, red_eyes, long_hair, etc.)
- Clothing: outfit tags (dress, school_uniform, hat, etc.)
- Expression: emotions (smile, blush, crying, angry, etc.)
- Action: poses/activities (sitting, standing, holding, looking_at_viewer, etc.)
- Background: setting tags (outdoors, indoors, classroom, forest, etc.)
- Atmosphere: mood tags (dark, bright, romantic, etc.)

XML FORMAT:
<character_1>
  <n></n>
  <gender></gender>
  <appearance></appearance>
  <clothing></clothing>
  <expression></expression>
  <action></action>
  <interaction></interaction>
  <position></position>
</character_1>

<general_tags>
  <count></count>
  <artists></artists>
  <style>anime style</style>
  <background></background>
  <environment></environment>
  <perspective></perspective>
  <atmosphere></atmosphere>
  <lighting></lighting>
  <resolution>max_high_resolution</resolution>
  <quality>very aesthetic, masterpiece, no text</quality>
  <objects></objects>
  <other></other>
</general_tags>

<caption>Natural language description of the scene with lighting/shadow details. Escape parentheses with backslash here too.</caption>
"""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP", {
                    "tooltip": "NewBie CLIP model - used to get the Gemma model path"
                }),
                "user_message": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Your message to Gemma"
                }),
            },
            "optional": {
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "You are a helpful AI assistant.",
                    "tooltip": "System prompt to set Gemma's behavior"
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "Sampling temperature (0=deterministic, higher=more creative)"
                }),
                "max_new_tokens": ("INT", {
                    "default": 512,
                    "min": 1,
                    "max": 4096,
                    "step": 1,
                    "tooltip": "Maximum number of tokens to generate"
                }),
                "top_p": ("FLOAT", {
                    "default": 0.9,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Top-p (nucleus) sampling threshold"
                }),
                "top_k": ("INT", {
                    "default": 50,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Top-k sampling (0=disabled)"
                }),
                "conversation_history": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Previous conversation history (optional, for multi-turn chat)"
                }),
                "unload_after": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Unload the generation model after response to free VRAM"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("response", "full_conversation",)
    OUTPUT_TOOLTIPS = ("Gemma's response", "Full conversation history for chaining",)
    FUNCTION = "chat"
    CATEGORY = "NewBie/LLM"
    TITLE = "NewBie Gemma Chat"
    DESCRIPTION = "Chat with the Gemma model using the same model loaded in NewBie CLIP. Supports system prompt, temperature, and multi-turn conversations."

    def _get_gemma_model_info(self, clip):
        """Extract Gemma model path and settings from the CLIP object"""
        # The tokenizer in NewBieCLIP contains info about where it was loaded from
        if hasattr(clip, 'tokenizer') and hasattr(clip.tokenizer, 'name_or_path'):
            model_path = clip.tokenizer.name_or_path
        elif hasattr(clip, 'text_encoder') and hasattr(clip.text_encoder, 'name_or_path'):
            model_path = clip.text_encoder.name_or_path
        elif hasattr(clip, 'text_encoder') and hasattr(clip.text_encoder.config, '_name_or_path'):
            model_path = clip.text_encoder.config._name_or_path
        else:
            raise ValueError("Cannot determine Gemma model path from CLIP. Make sure you're using NewBie CLIP Loader.")
        
        device = clip.device if hasattr(clip, 'device') else "cuda"
        
        # Get dtype from text_encoder
        if hasattr(clip, 'text_encoder'):
            dtype = next(clip.text_encoder.parameters()).dtype
        else:
            dtype = torch.bfloat16
            
        return model_path, device, dtype

    def _load_generation_model(self, model_path: str, device: str, dtype: torch.dtype):
        """Load or retrieve cached Gemma model for generation"""
        global _gemma_gen_cache
        
        # Check if we can reuse cached model
        if (_gemma_gen_cache["model"] is not None and 
            _gemma_gen_cache["model_path"] == model_path and
            _gemma_gen_cache["device"] == device and
            _gemma_gen_cache["dtype"] == dtype):
            print(f"[NewBie Gemma Chat] Using cached generation model")
            return _gemma_gen_cache["model"], _gemma_gen_cache["tokenizer"]
        
        # Clear old cache
        if _gemma_gen_cache["model"] is not None:
            del _gemma_gen_cache["model"]
            del _gemma_gen_cache["tokenizer"]
            torch.cuda.empty_cache()
        
        print(f"[NewBie Gemma Chat] Loading Gemma generation model from: {model_path}")
        print(f"[NewBie Gemma Chat] Device: {device}, Dtype: {dtype}")
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library is required for Gemma Chat")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        
        # Load model for causal LM (generation)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=True,
        )
        model.eval()
        
        # Cache for reuse
        _gemma_gen_cache["model"] = model
        _gemma_gen_cache["tokenizer"] = tokenizer
        _gemma_gen_cache["model_path"] = model_path
        _gemma_gen_cache["device"] = device
        _gemma_gen_cache["dtype"] = dtype
        
        print(f"[NewBie Gemma Chat] Model loaded successfully")
        return model, tokenizer

    def _format_messages(self, system_prompt: str, user_message: str, conversation_history: str = "") -> list:
        """Format messages for Gemma chat template"""
        messages = []
        
        # Add system prompt if provided
        if system_prompt and system_prompt.strip():
            messages.append({
                "role": "user",
                "content": f"[System Instructions]\n{system_prompt.strip()}\n\n[End System Instructions]\n\nAcknowledge that you understand these instructions."
            })
            messages.append({
                "role": "assistant", 
                "content": "I understand and will follow these instructions."
            })
        
        # Parse and add conversation history if provided
        if conversation_history and conversation_history.strip():
            # Parse the history format: USER: ... ASSISTANT: ...
            history = conversation_history.strip()
            parts = []
            current_role = None
            current_content = []
            
            for line in history.split('\n'):
                if line.startswith('USER:'):
                    if current_role and current_content:
                        parts.append((current_role, '\n'.join(current_content)))
                    current_role = 'user'
                    current_content = [line[5:].strip()]
                elif line.startswith('ASSISTANT:'):
                    if current_role and current_content:
                        parts.append((current_role, '\n'.join(current_content)))
                    current_role = 'assistant'
                    current_content = [line[10:].strip()]
                elif current_role:
                    current_content.append(line)
            
            if current_role and current_content:
                parts.append((current_role, '\n'.join(current_content)))
            
            for role, content in parts:
                if content.strip():
                    messages.append({"role": role, "content": content.strip()})
        
        # Add current user message
        if user_message and user_message.strip():
            messages.append({
                "role": "user",
                "content": user_message.strip()
            })
        
        return messages

    def _apply_chat_template(self, tokenizer, messages: list) -> str:
        """Apply Gemma3 chat template to messages"""
        # Try to use tokenizer's built-in chat template if available
        if hasattr(tokenizer, 'apply_chat_template'):
            try:
                return tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception as e:
                print(f"[NewBie Gemma Chat] Tokenizer chat template failed: {e}, using manual format")
        
        # Manual Gemma3 chat template
        result = ""
        for message in messages:
            role = message['role']
            content = message['content']
            
            if role == 'user':
                result += f"<start_of_turn>user\n{content}<end_of_turn>\n"
            elif role == 'assistant':
                result += f"<start_of_turn>model\n{content}<end_of_turn>\n"
        
        result += "<start_of_turn>model\n"
        return result

    def chat(
        self,
        clip,
        user_message: str,
        system_prompt: str = "You are a helpful AI assistant.",
        temperature: float = 0.7,
        max_new_tokens: int = 512,
        top_p: float = 0.9,
        top_k: int = 50,
        conversation_history: str = "",
        unload_after: bool = True,
    ) -> Tuple[str, str]:
        """Generate a chat response from Gemma"""
        
        if not hasattr(clip, 'text_encoder') or not hasattr(clip, 'tokenizer'):
            raise ValueError("This node requires a NewBie CLIP model loaded with NewBie CLIP Loader")
        
        if not user_message or not user_message.strip():
            return ("Please provide a message.", conversation_history)
        
        # Get model info and load generation model
        model_path, device, dtype = self._get_gemma_model_info(clip)
        model, tokenizer = self._load_generation_model(model_path, device, dtype)
        
        # Format messages
        messages = self._format_messages(system_prompt, user_message, conversation_history)
        
        # Apply chat template
        prompt = self._apply_chat_template(tokenizer, messages)
        
        print(f"[NewBie Gemma Chat] Generating response...")
        print(f"[NewBie Gemma Chat] Temperature: {temperature}, Max tokens: {max_new_tokens}")
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Generate
        with torch.no_grad():
            generation_config = {
                "max_new_tokens": max_new_tokens,
                "do_sample": temperature > 0,
                "pad_token_id": tokenizer.eos_token_id,
            }
            
            if temperature > 0:
                generation_config["temperature"] = temperature
                generation_config["top_p"] = top_p
                if top_k > 0:
                    generation_config["top_k"] = top_k
            
            outputs = model.generate(
                **inputs,
                **generation_config
            )
        
        # Decode response
        generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Clean up response (remove any trailing turn markers)
        response = response.replace("<end_of_turn>", "").strip()
        
        # Build full conversation history for chaining
        if conversation_history and conversation_history.strip():
            full_conversation = f"{conversation_history.strip()}\nUSER: {user_message.strip()}\nASSISTANT: {response}"
        else:
            full_conversation = f"USER: {user_message.strip()}\nASSISTANT: {response}"
        
        print(f"[NewBie Gemma Chat] Response generated ({len(response)} chars)")
        
        # Clear VRAM if requested
        if unload_after:
            _clear_gemma_cache()
        
        return (response, full_conversation)


class NewBieGemmaChatAdvanced(NewBieGemmaChat):
    """
    Advanced Gemma Chat with more generation parameters and optional model override.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        base_inputs = super().INPUT_TYPES()
        
        # Add advanced options
        base_inputs["optional"]["repetition_penalty"] = ("FLOAT", {
            "default": 1.0,
            "min": 1.0,
            "max": 2.0,
            "step": 0.05,
            "tooltip": "Penalty for repeating tokens (1.0=no penalty)"
        })
        base_inputs["optional"]["do_sample"] = ("BOOLEAN", {
            "default": True,
            "tooltip": "Enable sampling (False=greedy decoding)"
        })
        base_inputs["optional"]["seed"] = ("INT", {
            "default": -1,
            "min": -1,
            "max": 2**31-1,
            "tooltip": "Random seed for reproducibility (-1=random)"
        })
        
        return base_inputs
    
    TITLE = "NewBie Gemma Chat (Advanced)"
    DESCRIPTION = "Advanced Gemma chat with additional generation parameters."

    def chat(
        self,
        clip,
        user_message: str,
        system_prompt: str = "You are a helpful AI assistant.",
        temperature: float = 0.7,
        max_new_tokens: int = 512,
        top_p: float = 0.9,
        top_k: int = 50,
        conversation_history: str = "",
        unload_after: bool = True,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
        seed: int = -1,
    ) -> Tuple[str, str]:
        """Generate a chat response from Gemma with advanced options"""
        
        if not hasattr(clip, 'text_encoder') or not hasattr(clip, 'tokenizer'):
            raise ValueError("This node requires a NewBie CLIP model loaded with NewBie CLIP Loader")
        
        if not user_message or not user_message.strip():
            return ("Please provide a message.", conversation_history)
        
        # Set seed if specified
        if seed >= 0:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        
        # Get model info and load generation model
        model_path, device, dtype = self._get_gemma_model_info(clip)
        model, tokenizer = self._load_generation_model(model_path, device, dtype)
        
        # Format messages
        messages = self._format_messages(system_prompt, user_message, conversation_history)
        
        # Apply chat template
        prompt = self._apply_chat_template(tokenizer, messages)
        
        print(f"[NewBie Gemma Chat Advanced] Generating response...")
        print(f"[NewBie Gemma Chat Advanced] Temperature: {temperature}, Max tokens: {max_new_tokens}, Seed: {seed}")
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Generate with advanced options
        with torch.no_grad():
            generation_config = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample and temperature > 0,
                "pad_token_id": tokenizer.eos_token_id,
                "repetition_penalty": repetition_penalty,
            }
            
            if do_sample and temperature > 0:
                generation_config["temperature"] = temperature
                generation_config["top_p"] = top_p
                if top_k > 0:
                    generation_config["top_k"] = top_k
            
            outputs = model.generate(
                **inputs,
                **generation_config
            )
        
        # Decode response
        generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Clean up response
        response = response.replace("<end_of_turn>", "").strip()
        
        # Build full conversation history
        if conversation_history and conversation_history.strip():
            full_conversation = f"{conversation_history.strip()}\nUSER: {user_message.strip()}\nASSISTANT: {response}"
        else:
            full_conversation = f"USER: {user_message.strip()}\nASSISTANT: {response}"
        
        print(f"[NewBie Gemma Chat Advanced] Response generated ({len(response)} chars)")
        
        # Clear VRAM if requested
        if unload_after:
            _clear_gemma_cache()
        
        return (response, full_conversation)


NODE_CLASS_MAPPINGS = {
    "NewBieGemmaChat": NewBieGemmaChat,
    "NewBieGemmaChatAdvanced": NewBieGemmaChatAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NewBieGemmaChat": "NewBie Gemma Chat",
    "NewBieGemmaChatAdvanced": "NewBie Gemma Chat (Advanced)",
}

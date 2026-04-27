# ============================================================
# CONFIGURATION
# ============================================================

PROMPT = """You are a distinguished historian specializing in the Scientific Revolution and the life of Sir Isaac Newton. Please provide a comprehensive and detailed response to the following questions about Isaac Newton's contributions to science and mathematics.

Background Context:
Isaac Newton (1642-1727) was an English mathematician, physicist, astronomer, and author who is widely recognized as one of the most influential scientists of all time. He was a key figure in the Scientific Revolution and his work laid the foundations for classical mechanics. Newton made seminal contributions to optics, developed the laws of motion and universal gravitation, and shares credit with Gottfried Wilhelm Leibniz for developing infinitesimal calculus.

Questions to Address:
1. What were Newton's three laws of motion and how did they revolutionize our understanding of physics?
2. How did Newton's work on universal gravitation explain both terrestrial and celestial phenomena?
3. What were Newton's major contributions to the field of optics, particularly regarding the nature of light and color?
4. How did Newton contribute to the development of calculus, and what was the nature of the priority dispute with Leibniz?

Instructions for Your Response:
- Begin with a brief introduction summarizing Newton's overall significance to science
- Address each question in a separate paragraph with specific examples and historical context
- Include at least one famous quote or anecdote related to Newton for each major topic
- Explain the practical applications and lasting impact of each contribution
- Use precise scientific terminology while remaining accessible to a general audience
- Conclude with a reflection on how Newton's work influenced subsequent generations of scientists
- Your response should demonstrate deep knowledge and analytical thinking

Please provide your detailed analysis:
"""

import argparse
_parser = argparse.ArgumentParser()
_parser.add_argument("--max-tokens", type=int, default=1000)
_args = _parser.parse_args()

MAX_NEW_TOKENS = _args.max_tokens  # Number of tokens to generate
BATCH_SIZE = 4        # Number of parallel sequences

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MODEL_REVISION = "fe8a4ea1ffedaf415f4da2f062534de366a451e6"
print(f"Model: {MODEL_NAME}")

# ============================================================
# SETUP
# ============================================================
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.get_device_name(0)}")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load model in FP16 with standard attention
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    revision=MODEL_REVISION,
    dtype=torch.float16,
    device_map="cuda",
    attn_implementation="eager"  # Use standard attention (not FlashAttention)
)

num_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {num_params / 1e9:.2f}B")


# ============================================================
# PREFILL
# ============================================================
# Tokenize and create batch
inputs = tokenizer([PROMPT] * BATCH_SIZE, return_tensors="pt", padding=True).to("cuda")
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

print(f"Batch size: {BATCH_SIZE}")
print(f"Input tokens per sequence: {input_ids.shape[1]}")
print(f"Building KV-Cache...")

# PREFILL: Process input and build KV-cache
import nvtx
with nvtx.annotate("Prefill"):
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True
        )

past_key_values = outputs.past_key_values
next_token_logits = outputs.logits[:, -1, :]
generated_ids = input_ids.clone()

# Show KV-cache info
if hasattr(past_key_values, 'layers'):
    num_layers = len(past_key_values.layers)
    first_layer = past_key_values.layers[0]
    key_shape = first_layer.keys.shape   # plural: .keys
elif hasattr(past_key_values, 'key_cache'):
    num_layers = len(past_key_values.key_cache)
    key_shape = past_key_values.key_cache[0].shape
else:
    num_layers = len(past_key_values)
    key_shape = past_key_values[0][0].shape

print(f"\nKV-Cache after prefill:")
print(f"  Layers: {num_layers}")
print(f"  Key shape: {list(key_shape)} (batch, heads, seq_len, head_dim)")


# ============================================================
# DECODE
# ============================================================
# Each iteration:
#   1. Sample next token from logits
#   2. Forward pass with only the new token
#   3. Read ENTIRE KV-cache (memory-bound, causes Long Scoreboard stalls)
#   4. Update KV-cache with new K,V
# ============================================================
import time

decode_times = []

for step in range(MAX_NEW_TOKENS):
    with nvtx.annotate(f"DecodeStep{step}"):
        # Sample next token
        next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
    
        # Check for EOS
        if (next_tokens == tokenizer.eos_token_id).all():
            print(f"  [Step {step+1}] All sequences finished (EOS)")
            break
    
        generated_ids = torch.cat([generated_ids, next_tokens], dim=-1)
        attention_mask = torch.cat([
            attention_mask,
            torch.ones((BATCH_SIZE, 1), device="cuda", dtype=attention_mask.dtype)
        ], dim=-1)
    
        # Forward pass with KV-cache (only the new token)
        step_start = time.perf_counter()
        with torch.no_grad():
            outputs = model(
                input_ids=next_tokens,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True
            )
        step_time = time.perf_counter() - step_start
        decode_times.append(step_time)
    
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]
    
        if (step + 1) % 50 == 0:
            if hasattr(past_key_values, 'layers'):
                kv_cache_size_mb = sum(
                    layer.keys.numel() + layer.values.numel() for layer in past_key_values.layers
                ) * 2 / 1024**2
                seq_len = past_key_values.layers[0].keys.shape[2]
            elif hasattr(past_key_values, 'key_cache'):
                kv_cache_size_mb = sum(
                    k.numel() + v.numel() for k, v in zip(past_key_values.key_cache, past_key_values.value_cache)
                ) * 2 / 1024**2
                seq_len = past_key_values.key_cache[0].shape[2]
            else:
                kv_cache_size_mb = sum(
                    layer[0].numel() + layer[1].numel() for layer in past_key_values
                ) * 2 / 1024**2
                seq_len = past_key_values[0][0].shape[2]
            print(f"  [Step {step+1}] Generated {step+1} tokens/seq, "
                  f"KV-cache seq_len: {seq_len}, "
                  f"KV-cache size: {kv_cache_size_mb:.1f} MB, "
                  f"step time: {step_time*1000:.2f} ms")

torch.cuda.synchronize()
print(f"\nGenerated {generated_ids.shape[1] - input_ids.shape[1]} new tokens per sequence")
if decode_times:
    print(f"Average step time: {sum(decode_times)/len(decode_times)*1000:.2f} ms")


# ============================================================
# OUTPUT
# ============================================================
# Decode and display generated text
generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print("Generated text:")
print("-" * 60)
print(generated_text)
print("-" * 60)

# Final KV-cache size
if hasattr(past_key_values, 'layers'):
    total_elements = sum(layer.keys.numel() + layer.values.numel() for layer in past_key_values.layers)
    seq_len = past_key_values.layers[0].keys.shape[2]
elif hasattr(past_key_values, 'key_cache'):
    total_elements = sum(k.numel() + v.numel() for k, v in zip(past_key_values.key_cache, past_key_values.value_cache))
    seq_len = past_key_values.key_cache[0].shape[2]
else:
    total_elements = sum(layer[0].numel() + layer[1].numel() for layer in past_key_values)
    seq_len = past_key_values[0][0].shape[2]

kv_size_mb = total_elements * 2 / 1024**2  # FP16 = 2 bytes
print(f"\nFinal KV-cache:")
print(f"  Sequence length: {seq_len}")
print(f"  Total size: {kv_size_mb:.1f} MB")

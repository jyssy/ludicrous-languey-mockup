import torch
from transformers import AutoTokenizer
from llmcompressor.transformers import SparseAutoModelForCausalLM
from llmcompressor.transformers import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
import os

# Set up environment for CPU usage
torch.set_num_threads(8)  # Optimize for 8 CPU cores

# Model configuration
model_id = "meta-llama/Llama-3.3-70B-Instruct"

# Load model with CPU-only settings
model = SparseAutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cpu",
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
    offload_folder="offload_folder"
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Configure quantization for CPU
recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_DYNAMIC",
    ignore=["lm_head"],
    chunk_size=1024
)

# Apply quantization with memory optimization
oneshot(
    model=model,
    recipe=recipe,
    batch_size=1
)

# Save paths
SAVE_DIR = model_id.split("/")[1] + "-FP8-Dynamic"

# Save with memory optimization
model.save_pretrained(
    SAVE_DIR,
    max_shard_size="2GB",
    safe_serialization=True
)
tokenizer.save_pretrained(SAVE_DIR)

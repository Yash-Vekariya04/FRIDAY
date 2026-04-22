# Modern HuggingFace Transformers code
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# 1. Define the Quantization Rules
# We tell the system: "Load this model using only 4 bits per parameter"
quant_config = BitsAndBytesConfig(load_in_4bit=True)

# 2. Load the Brain
# This would normally crash a 16GB laptop, but with 4-bit quantization, it loads easily!
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B", 
    quantization_config=quant_config, 
    device_map="auto"
)

print("F.R.I.D.A.Y. is loaded and quantized.")
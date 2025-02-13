# ollama_entrypoint.py
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Model configuration (same as before)
base_model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
adapter_path = "./deepseek-r1-1.5B-qwen-distill-finetuned"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
base_model = AutoModelForCausalLM.from_pretrained(base_model_id, trust_remote_code=True)
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

def main():
    # Read prompt from STDIN
    prompt = sys.stdin.read().strip()
    if not prompt:
        print("No prompt provided.", flush=True)
        return

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7
    )
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(response_text, flush=True)

if __name__ == "__main__":
    main()

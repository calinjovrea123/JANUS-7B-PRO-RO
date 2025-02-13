import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi import FastAPI, Request
import uvicorn
from peft import PeftModel

# Specify your base model identifier (this should be the original model you started with)
base_model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# Load the tokenizer and the full base model (this ensures we have a valid config with a "model_type")
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
base_model = AutoModelForCausalLM.from_pretrained(base_model_id, trust_remote_code=True)

# Path to your saved adapter (fine-tuned) weights/configuration
adapter_path = "./deepseek-r1-1.5B-qwen-distill-finetuned"

# Load the adapter (PEFT) weights on top of the base model
model = PeftModel.from_pretrained(base_model, adapter_path)

# Set the model to evaluation mode
model.eval()


app = FastAPI()

@app.post("/generate")
async def generate_text(request: Request):
    data = await request.json()
    prompt = data.get("prompt", "")
    # Tokenize and generate (adjust parameters as needed)
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs, 
        max_new_tokens=100, 
        do_sample=True, 
        temperature=0.7
    )
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": response_text}

if __name__ == "__main__":
    # For local testing, run on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)

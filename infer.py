
import torch
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images

# specify the path to the model
model_path = "deepseek-ai/Janus-Pro-7B"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True
)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

question = "I'm Looking to do a program, that parses logs and sends alerts based if an error arrives in the logs, help me with this. 1. What programming language do you want to use for the program? python 2. What type of logs do you want to analyze (e.g., system logs, application logs, etc.)? program logs, application logs 3. What kind of alerts do you want to send (e.g., email, SMS, etc.)? email 4. What is the expected format of the logs (e.g., CSV, JSON, etc.)? txt, csv, json 5. Do you have any specific requirements or constraints for the program (e.g., performance, scalability, etc.)? it should run seemlessly smoothly without any bugs, but stellarly fulfilled, please create the program "

conversation = [
    {
        "role": "<|User|>",
        "content": f"{question}",
        "images": [],
    },
    {"role": "<|Assistant|>", "content": ""},
]

# load images and prepare for inputs
pil_images = load_pil_images(conversation)
prepare_inputs = vl_chat_processor(
    conversations=conversation, images=pil_images, force_batchify=True
).to(vl_gpt.device)

# # run image encoder to get the image embeddings
inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

# # run the model to get the response
outputs = vl_gpt.language_model.generate(
    inputs_embeds=inputs_embeds,
    attention_mask=prepare_inputs.attention_mask,
    pad_token_id=tokenizer.eos_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    max_new_tokens=512,
    do_sample=False,
    use_cache=True,
)

answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
print(f"{prepare_inputs['sft_format'][0]}", answer)

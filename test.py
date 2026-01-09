import torch
from transformers import AutoModelForCausalLM
path = "Qwen/Qwen2.5-32B-Instruct"
# path = "meta-llama/Llama-3.1-8B-Instruct"
#get cuda device and count 
print(torch.cuda.device_count())
# device = torch.device(f'cuda')
model = AutoModelForCausalLM.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    max_memory={
        0: "80GiB",
        1: "80GiB",
    },
    attn_implementation="flash_attention_2",
)
model.eval()
print(model.hf_device_map)
import ipdb; ipdb.set_trace()
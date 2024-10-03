import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

model_id = "/proj/checkpoints/stallone/models/preview/granite-3b-instruct-preview-4k-r240917a"
#peft_id = "data/outputs/granite-3b-instruct-preview-16k-100krt/checkpoint-3602"
peft_id = "data/outputs/granite-3b-instruct-preview-32k-500krt/checkpoint-2122"
#merged_model_path = "data/outputs/granite-3b-instruct-preview-16k-100krt/merged/" 
merged_model_path = "data/outputs/granite-3b-instruct-preview-32k-500krt/merged/"

torch_dtype = torch.bfloat16
# place the model on GPU
device_map = {"": "cuda"}

tokenizer = AutoTokenizer.from_pretrained(model_id)

base_model = AutoModelForCausalLM.from_pretrained(
  model_id, 
  torch_dtype=torch.bfloat16,
  device_map=device_map,
  attn_implementation="flash_attention_2",

  # NOTE: expand rope base
  rope_theta=500e3,
)

model = PeftModel.from_pretrained(
    base_model, 
    peft_id,
    torch_dtype=torch.bfloat16,
    device_map=device_map,
)
# NOTE: merge LoRA weights
merged_model = model.merge_and_unload().eval()

#p print save the model and tokenizer...
merged_model.save_pretrained(merged_model_path)
tokenizer.save_pretrained(merged_model_path)

import os
import torch
from transformers import AutoConfig, AutoModelForCausalLM
from specforge_het.models import *


revision = "3ffd1f50b179e643d839c86df9ffbbefcb0d5018"
if not os.path.exists('./output/temp_save'):
    base_model_path = 'Qwen/Qwen3-30B-A3B-Instruct-2507'
    base_config = AutoConfig.from_pretrained(base_model_path, revision=revision)
    algo_kwargs = dict(draft_layers=2)
    model = SpeculativeQwen3MoeForCausalLM.from_basemodel(
        base_config, base_model_path,
        AlgoClass=EagleV2, algo_kwargs=algo_kwargs,
        revision=revision, _fast_init=False, device_map="auto",
        torch_dtype=torch.bfloat16
    )
    draft_config_path = 'meta-llama/Llama-2-7b-chat-hf'
    draft_config = AutoConfig.from_pretrained(draft_config_path)
    draft_model = LlamaDrafter(draft_config, model)
    model.set_draft_model(draft_model)
    model.save_pretrained('./output/temp_save')

else:
    model = AutoModelForCausalLM.from_pretrained('./output/temp_save',
        device_map="auto", revision=revision,
        torch_dtype=torch.bfloat16, trust_remote_code=True
    )

print(model)

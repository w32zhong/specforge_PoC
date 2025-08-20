import os
import json
from collections import defaultdict
import torch
import transformers
import safetensors
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from colorama import Fore, Style

from specforge_het.models import *

print(transformers.__path__)
print(transformers.__version__)


def freeze_model(model):
    for param_path, param in model.named_parameters():
        param.requires_grad = False


def load_speculative_model_if_possible(configs, freeze_base_model=True, **kwargs):
    try:
        model = AutoModelForCausalLM.from_pretrained(configs.model_path,
            trust_remote_code=True, **kwargs)
        if freeze_base_model:
            freeze_model(model)

    except Exception as e:
        # base model
        base_class_name, base_model_path = configs.init_base_model
        base_config = AutoConfig.from_pretrained(base_model_path)
        if configs.free_base_layers:
            setattr(base_config, configs.free_base_layers, 0)
        algo_class_name, algo_kwargs = configs.init_speculative_algorithm
        model = eval(base_class_name).from_basemodel(
            base_config, base_model_path,
            AlgoClass=eval(algo_class_name), algo_kwargs=eval(algo_kwargs),
            _fast_init=False, **kwargs
        )
        if freeze_base_model:
            freeze_model(model)

        # (uninitialized) draft model
        draft_class_name, draft_config_path = configs.init_draft_config
        draft_config = AutoConfig.from_pretrained(draft_config_path)
        draft_model = eval(draft_class_name)(draft_config, model)

        # attach draft model for the specified algorithm
        model.set_draft_model(draft_model)

    return model


def load_models(configs, world_size=1, rank=0, use_deepspeed=False):
    tokenizer = AutoTokenizer.from_pretrained(configs.tokenizer_path,
        add_bos_token=False, add_eos_token=False, trust_remote_code=True)

    chat_template_path = os.path.join('specforge_het', configs.chat_template)
    with open(chat_template_path, 'r') as fh:
        lines = fh.readlines()
    cleaned_template = "".join(line.lstrip() for line in lines)
    tokenizer.chat_template = cleaned_template

    for token_key, token in eval(configs.tokenizer_add_tokens).items():
        setattr(tokenizer, token_key, token)

    assert tokenizer.pad_token is not None # for training padding
    assert tokenizer.eos_token is not None # for generation stop sign
    assert tokenizer.unk_token is not None # for speculative index tree_Q

    test_messages = [
        {"role": "system", "content": "You are a friendly chatbot!"},
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
        {"role": "user", "content": "I'd like to show off how chat templating works!"}
    ]
    output_test_prompt = tokenizer.apply_chat_template(test_messages, tokenize=False)
    print('[test chat template]', Fore.YELLOW, '\n' + output_test_prompt, Style.RESET_ALL)

    if use_deepspeed:
        device_map = None # the model will only contain "meta weights" if use_deepspeed
    else:
        device_map='auto' if world_size == 1 else f'cuda:{rank}'

    model = load_speculative_model_if_possible(configs,
        attn_implementation="eager", device_map=device_map, torch_dtype=eval(configs.dtype)
    )

    # ensure no module is on an abstract device
    for path, module in model.named_modules():
        try:
            module_device = next(module.parameters()).device
        except StopIteration:
            continue
        if str(module_device) == 'meta':
            assert False, f'[offloaded meta module] {path}'

    return tokenizer, model

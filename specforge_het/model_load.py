import os, json, copy, re, time
from collections import defaultdict
from colorama import Fore, Style

import torch
import transformers
import safetensors
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

from specforge_het.models import *
from specforge_het.utils import master_print, get_num_parameters

master_print(transformers.__path__)
master_print(transformers.__version__)


def local_model_path(model_path):
    try:
        model_path = snapshot_download(model_path)
    except:
        pass
    return model_path


def freeze_model(model):
    for param_path, param in model.named_parameters():
        param.requires_grad = False


def get_state_dict(model_path):
    for get_local_path_fn in [os.path.join, hf_hub_download]:
        for fname in [
            'model.safetensors',
            'model.safetensors.index.json',
            'pytorch_model.bin',
            'states.pt' # our naming convention for the draft model
        ]:
            try:
                local_path = get_local_path_fn(model_path, fname)
                assert os.path.exists(local_path)

                if local_path.endswith('model.safetensors.index.json'):
                    with open(local_path, "r") as fh:
                        index = json.load(fh)
                    sharps = index["weight_map"]
                    dir = os.path.dirname(local_path)
                    state_dicts = [
                        safetensors.torch.load_file(
                            os.path.join(dir, filename)
                        )
                        for filename in set(shards.values())
                    ]
                elif local_path.endswith('model.safetensors'):
                    state_dicts = [
                        safetensors.torch.load_file(local_path)
                    ]
                else:
                    state_dicts = [
                        torch.load(local_path)
                    ]

                state_dict = dict()
                for partial_state_dict in state_dicts:
                    state_dict.update(partial_state_dict)
                return state_dict

            except Exception:
                continue
    return None


def load_speculative_model_if_possible(configs, freeze_base_model=True, **kwargs):
    try:
        if configs.stand_alone_draft_model_path:
            raise Exception('go to except')

        model_path = local_model_path(configs.model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path,
            trust_remote_code=True, **kwargs)

    except Exception as e:
        master_print(e, "\n" "Fallback to separated base/draft loading...")
        time.sleep(3)

        # base model
        base_class_name, base_model_path = configs.init_base_model
        base_config = AutoConfig.from_pretrained(base_model_path)
        if configs.free_base_layers:
            n_base_layers = getattr(base_config, configs.free_base_layers)
            setattr(base_config, configs.free_base_layers + '_free', n_base_layers)
            setattr(base_config, configs.free_base_layers, 0)
        algo_class_name, algo_kwargs = configs.init_speculative_algorithm
        model = eval(base_class_name).from_basemodel(
            base_config, base_model_path,
            AlgoClass=eval(algo_class_name), algo_kwargs=eval(algo_kwargs),
            **kwargs
        )

        # (uninitialized) draft model
        draft_class_name, draft_config_path = configs.init_draft_config
        draft_config = AutoConfig.from_pretrained(draft_config_path)
        draft_config = copy.deepcopy(draft_config)
        base = model.config # for eval short hands
        for key, val in configs.draft_config_modify.items():
            val = eval(val)
            old_val = getattr(draft_config, key)
            if old_val != val:
                setattr(draft_config, key, val)
                master_print(f'drafter config[{key}]: {old_val} -> {val}')

        draft_model = eval(draft_class_name)(draft_config, model)
        draft_model.to(model.dtype)
        #MLP = draft_model.layers[0].mlp
        #MoEs = model.model.layers[0].mlp.experts[:model.config.num_experts_per_tok]
        #assert get_num_parameters(MLP) == get_num_parameters(MoEs)

        # attach draft model for the specified algorithm
        model.set_draft_model(draft_model)

        if configs.stand_alone_draft_model_path:
            draft_state_dict = get_state_dict(configs.stand_alone_draft_model_path)
            key_adapt = configs.stand_alone_draft_model_key_adapt
            key_modify = configs.stand_alone_draft_model_key_modify
            for key in list(draft_state_dict.keys()):
                for pattern, repl in key_modify.get(key_adapt, []):
                    if not re.match(pattern, key):
                        continue
                    if repl is None:
                        del draft_state_dict[key]
                    else:
                        new_key = re.sub(pattern, repl, key)
                        draft_state_dict[new_key] = draft_state_dict.pop(key)
            master_print('draft model:', model.draft_model)
            master_print('stand-alone loading keys:', draft_state_dict.keys())
            model.draft_model.load_state_dict(draft_state_dict, strict=True)

    if freeze_base_model and hasattr(model, 'base_model'):
        freeze_model(model.base_model)
    return model


def load_models(configs, world_size=1, rank=0, use_deepspeed=False):
    tokenizer = AutoTokenizer.from_pretrained(configs.tokenizer_path,
        add_bos_token=False, add_eos_token=False, trust_remote_code=True)

    chat_template_path = os.path.join('specforge_het', configs.chat_template)
    with open(chat_template_path, 'r') as fh:
        lines = fh.readlines()
    cleaned_template = "".join(line.strip() for line in lines)
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
    test_prompt = tokenizer.apply_chat_template(test_messages, tokenize=False)
    master_print('[chat template]', Fore.YELLOW, '\n' + test_prompt, Style.RESET_ALL)
    test_prompt_encoded = tokenizer.encode(test_prompt)
    master_print('[encode-decode]', Fore.RED, [tokenizer.decode(test_prompt_encoded)], Style.RESET_ALL)

    if use_deepspeed:
        device_map = None # the model will only contain "meta weights" if use_deepspeed
    else:
        device_map='auto' if world_size == 1 else f'cuda:{rank}'

    model = load_speculative_model_if_possible(configs,
        attn_implementation="eager", device_map=device_map,
        torch_dtype=eval(configs.dtype), max_memory=configs.max_memory
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

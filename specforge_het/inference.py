import os
import torch
import time
from specforge_het.configs import Configs
from specforge_het.model_load import load_models
from specforge_het.specforge_lm import is_speculative_model

sys_instructions = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."

question = "Thomas is very healthy, but he has to go to the hospital every day. What could be the reasons?"


def main(config_file='configs.ini', use_saved_json_config=None, **injects):
    configs = Configs.from_config_file(config_file, **injects)

    # inference needs to keep base layers
    configs.set_obj('modeling.free_base_layers', None)

    if use_saved_json_config:
        assert (isinstance(use_saved_json_config, str)
            and os.path.exists(use_saved_json_config))
        configs.load_json(use_saved_json_config,
            warn_change_key_prefix='modeling.',
            ignore_keys=['modeling.dtype', 'modeling.free_base_layers']
                + list(injects.keys())
        )

    tokenizer, model = load_models(configs.modeling)

    model.eval()
    model.tokenizer = tokenizer
    model.inference_configs = configs.inference

    test_messages = [
        #{"role": "system", "content": sys_instructions},
        {"role": "user", "content": question},
    ]
    test_prompt = tokenizer.apply_chat_template(test_messages, tokenize=False, add_generation_prompt=True)
    test_inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)

    start_time = time.perf_counter()

    total_generated_tokens = 0
    print(tokenizer.batch_decode(test_inputs.input_ids), end='\n', flush=True)
    with torch.no_grad():
        if is_speculative_model(model):
            eos = tokenizer.eos_token_id
            accept_length, iter_cost = [], []
            start_iter_time = time.perf_counter()
            for tokens in model.speculative_generate(**test_inputs):
                eos_pos = (tokens == eos).nonzero()
                accept_tokens = tokens[0] if eos_pos.numel() == 0 else tokens[0, :eos_pos[0,1]]
                total_generated_tokens += len(accept_tokens)
                print(tokenizer.decode(accept_tokens), end=' ', flush=True)
                #print(accept_tokens)
                if eos_pos.numel() > 0:
                    break
                accept_length.append(len(accept_tokens))
                iter_cost.append(time.perf_counter() - start_iter_time)
                start_iter_time = time.perf_counter()
            print('\n')
            accept_length.pop(0) # exclude pre-fill
            prefill_cost = iter_cost.pop(0)
            n_iters = len(iter_cost)

            print(accept_length)
            print('prefilling cost:', round(prefill_cost, 3))
            print('decoding cost:', round(sum(iter_cost) / n_iters, 3))
            print('max accept_length:', max(accept_length))
            print('min accept_length:', min(accept_length))
            print('avg accept_length:', round(sum(accept_length) / n_iters, 3))
        else:
            from transformers import GenerationConfig, TextStreamer
            generation_config = GenerationConfig.from_pretrained(
                configs.modeling.model_path,
                do_sample=False,
                max_new_tokens=8000
            )
            generated = model.generate(**test_inputs,
                generation_config=generation_config,
                streamer=TextStreamer(tokenizer),
            )
            print('\n')
            total_generated_tokens = len(generated[0])

    seconds = time.perf_counter() - start_time
    print('num output tokens:', total_generated_tokens, f'in {seconds} sec')
    print('tokens per second:', round(total_generated_tokens / seconds, 3))


if __name__ == '__main__':
    import fire
    fire.Fire(main)

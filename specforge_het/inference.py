import os, json, time, sys
import torch
from specforge_het.configs import Configs
from specforge_het.model_load import load_models
from specforge_het.specforge_lm import is_speculative_model

from specforge_het.sglang_adapter_utils import run_mtbench
from specforge_het.sys_prompts import sys_prompt_lib


def generate(model, inputs):
    tokenizer = model.tokenizer
    print(tokenizer.batch_decode(inputs.input_ids), end='\n', flush=True)
    meta_info = dict(completion_tokens=0, spec_verify_ct=0, accept_tokens=[])
    with torch.no_grad():
        if is_speculative_model(model):
            eos = tokenizer.eos_token_id
            new_tokens = []
            for tokens in model.speculative_generate(**inputs):
                eos_pos = (tokens == eos).nonzero()
                accept_tokens = (tokens[0] if eos_pos.numel() == 0
                                 else tokens[0, :eos_pos[0,1]]).tolist()
                print(tokenizer.decode(accept_tokens), end=' ', flush=True)
                new_tokens += accept_tokens

                if eos_pos.numel() > 0:
                    break
                elif len(new_tokens) >= model.inference_configs.max_new_tokens:
                    break

                meta_info['completion_tokens'] += len(accept_tokens)

                meta_info['spec_verify_ct'] += 1
                meta_info['accept_tokens'].append(accept_tokens)

            new_text = tokenizer.decode(new_tokens)

        else:
            from transformers import GenerationConfig, TextStreamer
            generation_config = GenerationConfig.from_pretrained(
                configs.modeling.model_path,
                do_sample=False,
                max_new_tokens=model.inference_configs.max_new_tokens
            )
            generated = model.generate(**inputs,
                generation_config=generation_config,
                streamer=TextStreamer(tokenizer),
            )
            meta_info['completion_tokens'] += len(generated[0])
            new_text = tokenizer.decode(generated[0])

    return new_text, meta_info


def calc_metrics(meta_info, d=3):
    m = meta_info.copy()
    if accept_tokens := m.pop('accept_tokens', []):
        m['accept_lens'] = [len(ac) for ac in accept_tokens]
        m['accept_lens.sum'] = sum(m['accept_lens'])
        m['accept_lens.max'] = max(m['accept_lens'])
    if m['spec_verify_ct'] > 0:
        m['avg_accept_len'] = round(m['completion_tokens'] / m['spec_verify_ct'], d)
    m['throughputs'] = round(m['completion_tokens'] / m['time_cost'], d)
    m['time_cost'] = round(m['time_cost'], 2)
    return m


def main(config_file='configs.ini', use_saved_json_config=None, sys_prompt=None,
         mtbench=None, mtbench_use_sgl_chat_template=False, outfile=None, **injects):

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

    begin = time.perf_counter()
    if mtbench:
        def callbk(model, sgl_prompt, messages, _):
            if mtbench_use_sgl_chat_template:
                prompt = sgl_prompt
            else:
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            new_text, meta_info = generate(model, inputs)
            return new_text, meta_info

        res = run_mtbench(callbk, model, mtbench, sampling_params={},
                          sys_prompt=sys_prompt_lib[sys_prompt],
                          sgl_chat_template=mtbench_use_sgl_chat_template,
                          num_threads=1)

        meta_info = dict(completion_tokens=0, spec_verify_ct=0)
        for i, res in enumerate(res):
            mi = res.get_meta_info('answer_1')
            meta_info['completion_tokens'] += mi['completion_tokens']
            meta_info['spec_verify_ct'] += mi['spec_verify_ct']

            mi = res.get_meta_info('answer_2')
            meta_info['completion_tokens'] += mi['completion_tokens']
            meta_info['spec_verify_ct'] += mi['spec_verify_ct']

    else:
        # one-shot example
        question = "Thomas is very healthy, but he has to go to the hospital every day. What could be the reasons?"
        test_messages = list(filter(lambda c: c["content"], [
            {"role": "system", "content": sys_prompt_lib[sys_prompt]},
            {"role": "user", "content": question},
        ]))
        test_prompt = tokenizer.apply_chat_template(test_messages,
                        tokenize=False, add_generation_prompt=True)
        test_inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
        _, meta_info = generate(model, test_inputs)

    meta_info['time_cost'] = time.perf_counter() - begin
    metrics = calc_metrics(meta_info)

    for key, val in metrics.items():
        print(f'{key:>30}:', val)

    if outfile is not None:
        with open(outfile, 'a') as fh:
            j = json.dumps(dict(
                    argv=sys.argv[1:],
                    **metrics
                ), sort_keys=True)
            print(j, file=fh)


if __name__ == '__main__':
    import fire
    os.environ["PAGER"] = "cat"
    fire.Fire(main)

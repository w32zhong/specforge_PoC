from specforge_het.configs import Configs
from specforge_het.model_load import load_models


def main(config_file='configs.ini', **injects):
    configs = Configs.from_config_file(config_file, **injects)
    tokenizer, model = load_models(configs.modeling)
    model.eval()
    model.tokenizer = tokenizer
    model.inference_configs = configs.inference

    test_messages = [
        {"role": "system", "content": "You are a friendly chatbot!"},
        {"role": "user", "content": "where does the last name hogan come from?"},
    ]
    test_prompt = tokenizer.apply_chat_template(test_messages, tokenize=False, add_generation_prompt=True)
    test_inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)

    eos = tokenizer.eos_token_id
    cnt, accept_length = 0.01, []
    print(tokenizer.batch_decode(test_inputs.input_ids), end='\n', flush=True)
    for tokens in model.speculative_generate(**test_inputs):
        eos_pos = (tokens == eos).nonzero()
        accept_tokens = tokens[0] if eos_pos.numel() == 0 else tokens[0, :eos_pos[0,1]]
        print(tokenizer.decode(accept_tokens), end='', flush=True)
        if eos_pos.numel() > 0:
            break
        accept_length.append(len(accept_tokens))
        cnt += 1
    print('\n')

    print('max accept_length:', max(accept_length))
    print('min accept_length:', min(accept_length))
    print('avg accept_length:', round(sum(accept_length) / cnt, 3))


if __name__ == '__main__':
    import fire
    fire.Fire(main)

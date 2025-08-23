from specforge_het.configs import Configs
from specforge_het.model_load import load_models


def main(config_file='configs.ini', **injects):
    configs = Configs.from_config_file(config_file, **injects)
    configs.set_obj('modeling.free_base_layers', False)

    tokenizer, model = load_models(configs.modeling)
    model.eval()
    model.tokenizer = tokenizer
    model.training_configs = configs.training

    test_messages = [
        {"role": "system", "content": "You are a friendly chatbot!"},
        {"role": "user", "content": "Hello, how are you?"},
    ]
    test_prompt = tokenizer.apply_chat_template(test_messages, tokenize=False, add_generation_prompt=True)
    test_inputs = tokenizer(test_prompt)

    output = model.speculative_generate(**test_inputs)
    breakpoint()


if __name__ == '__main__':
    import fire
    fire.Fire(main)

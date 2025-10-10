import re
import numpy as np
from functools import partial

g_count = 0

def count(d):
    global g_count
    g_count += d
    return g_count


def cmp_tensor_w_another(tensor):
    import torch
    try:
        another_tensor = torch.load('/tmp/debug_cmp_tensor_w_another.pt')
        print(torch.allclose(tensor.float(), another_tensor.float()))
    except Exception as e:
        print(e)
        pass
    torch.save(tensor, '/tmp/debug_cmp_tensor_w_another.pt')


def test_nan_grad(model):
    import torch
    for name, param in model.named_parameters():
        if param.grad is not None and torch.any(param.grad.isnan()):
            print(name, param.grad)
            return True
    return False


def hook_fn(model, tokenizer, path, module, inputs, output):
    if 'embed_tokens' in path or re.match(r'layers.\d+$', path):
        output = output[0] if isinstance(output, tuple) else output
        inputs = inputs[0] if isinstance(inputs, tuple) else inputs
        print(path, output.shape)
        if 'embed_tokens' in path:
            if tokenizer:
                print([tokenizer.decode(t) for t in inputs[0]])
            print('INP:', inputs)
        else:
            print('INP:', inputs.sum(-1))
        print('OUT:', output.sum(-1))
        print('\n')


def hook_model(model, tokenizer=None):
    for path, module in model.named_modules():
        module.register_forward_hook(
            partial(hook_fn, model, tokenizer, path)
        )

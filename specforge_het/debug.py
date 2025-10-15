import re
import copy
import inspect
import numpy as np
from functools import partial
from colorama import Fore, Style

g_count = 0
g_tensor_ckpt = dict()

def count(d=1):
    global g_count
    g_count += d
    return g_count


def debug_hook_model(model, stack_regex_filters=[], path_regex_filters=[], verbose=False, prefix=''):
    import torch
    def hook_fn(path, prefix, module, inputs, output):
        if verbose: print(path)
        if path_regex_filters:
            for regex in path_regex_filters:
                if re.match(regex, path):
                    break
            else:
                return
        stack_filenames = [s.filename for s in inspect.stack()]
        for i, filename in enumerate(stack_filenames):
            if filename == __file__:
                continue
            for regex in stack_regex_filters:
                if re.match(regex, filename):
                    break
            else:
                continue
            break
        else:
            i = 0
        try:
            assert isinstance(inputs, tuple)

            serializable_inputs = []
            for input in inputs:
                if isinstance(input, torch.Tensor):
                    serializable_inputs.append(input)
                else:
                    serializable_inputs.append(str(input))

            serializable_inputs = copy.deepcopy(serializable_inputs)
            output = copy.deepcopy(output)

            frame = inspect.stack()[i]
            file, line = frame.filename, frame.lineno
            key = f'{count():03}_{prefix}{path}'
            print(f'[hook] [{key}] -> {file}:{line}')

        except:
            import rpdb; rpdb.set_trace()

        g_tensor_ckpt[key] = dict(
            loc=f'{file}:{line}', name=f'{module}@{path}',
            inputs=serializable_inputs, output=output
        )

    for path, module in model.named_modules():
        module.register_forward_hook(
            partial(hook_fn, path, prefix)
        )


def cmp_tensor_w_another(tensor):
    import torch
    try:
        another_tensor = torch.load('/tmp/debug_cmp_tensor_w_another.pt')
        print(torch.allclose(tensor.float(), another_tensor.float()))
    except Exception as e:
        print(e)
        pass
    torch.save(tensor, '/tmp/debug_cmp_tensor_w_another.pt')


def tensor_error(t1, t2):
    import torch
    if not isinstance(t1, torch.Tensor) or not isinstance(t2, torch.Tensor):
        return 'not tensor'
    elif t1.shape != t2.shape:
        return f'shape mismatch ({t1.shape} != {t2.shape})'
    else:
        d = t1.float() - t2.float()
        return (f'{t1.shape}'
                + f' max={d.max().item()},'
                + f' min={d.min().item()},'
                + f' mean={d.mean().item()}.')


def first_matrix(x, use_attr=None):
    import torch
    if use_attr and hasattr(x, use_attr):
        return first_matrix(getattr(x, use_attr))
    elif isinstance(x, torch.Tensor):
        return x.squeeze()
    elif isinstance(x, tuple) and len(x) > 0:
        return first_matrix(x[0])
    elif isinstance(x, list) and len(x) > 0:
        return first_matrix(x[0])
    else:
        return x


def interactive_diff_hook_ckpts(ckpt_1_path, ckpt_2_path, window=12,
                                extra_out_attrs=['last_hidden_state', 'next_token_logits']):
    import torch
    from types import SimpleNamespace
    ckpt_1 = torch.load(ckpt_1_path, weights_only=False)
    ckpt_2 = torch.load(ckpt_2_path, weights_only=False)
    ckpt_1_keys = list(sorted(ckpt_1.keys()))
    ckpt_2_keys = list(sorted(ckpt_2.keys()))
    cnt_1, cnt_2 = 0, 0
    while True:
        print(ckpt_1_keys)
        print(ckpt_2_keys)
        choices_1 = ckpt_1_keys[cnt_1: cnt_1 + window]
        choices_2 = ckpt_2_keys[cnt_2: cnt_2 + window]
        k = None
        while k is None:
            print('~' * 80)
            print(ckpt_1_path, choices_1)
            print(ckpt_2_path, choices_2)
            c1 = SimpleNamespace(**ckpt_1[ckpt_1_keys[cnt_1]])
            c2 = SimpleNamespace(**ckpt_2[ckpt_2_keys[cnt_2]])
            i1, i2 = first_matrix(c1.inputs), first_matrix(c2.inputs)
            print(Fore.BLUE, 'Input error:', tensor_error(i1, i2), Style.RESET_ALL)
            o1 = first_matrix(c1.output, use_attr=extra_out_attrs[0])
            o2 = first_matrix(c2.output, use_attr=extra_out_attrs[1])
            print(Fore.CYAN, 'Output error:', tensor_error(o1, o2), Style.RESET_ALL)
            breakpoint()
        try:
            cnt_1 = ckpt_1_keys.index(k)
        except:
            pass
        try:
            cnt_2 = ckpt_2_keys.index(k)
        except:
            pass


def test_nan_grad(model):
    import torch
    for name, param in model.named_parameters():
        if param.grad is not None and torch.any(param.grad.isnan()):
            print(name, param.grad)
            return True
    return False


if __name__ == '__main__':
    import fire, os
    os.environ["PAGER"] = "cat"
    fire.Fire(interactive_diff_hook_ckpts)

import sys
import json


def extract_arg(argv, key):
    for idx, arg in enumerate(argv):
        if arg == key and idx + 1 < len(argv):
            return argv[idx + 1]
    return None


def extract_key(d, keys):
    return {k: d if k == '*' else d.get(k, None) for k in keys}


def process_argv_filter(argv, filter):
    if isinstance(filter, list):
        return all([process_argv_filter(argv, f) for f in filter])
    else:
        if '=' in filter:
            key, val = filter.split('=')
            return extract_arg(argv, key) == val
        else:
            key = filter
            return key in argv


def filter_json(j, *keys_filter, **argv_filter):
    for key, filters in argv_filter.items():
        if not process_argv_filter(j[key], filters):
            return None
    else:
        kv = extract_key(j, keys_filter)
        return kv


def filter_json_array(arr, *keys_filter, **argv_filter):
    matches = []
    for j in arr:
        if kv := filter_json(j, *keys_filter, **argv_filter):
            matches.append(kv)
    return matches


def filter_jsonl(path, *keys_filter, **argv_filter):
    j_arr = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            j = json.loads(line)
            j_arr.append(j)

    return filter_json_array(j_arr, *keys_filter, **argv_filter)


def first_match(matches, key, none_char='-', round_to=2):
    if len(matches) == 0:
        return none_char
    else:
        v = matches[0].get(key, none_char)
        try:
            v = str(round(float(v), round_to))
        except:
            pass
        return v


def multi_layer_results(path):
    for ckpt in 'blooming-silence-78,laced-wood-90,trim-waterfall-88'.split(',')[::-1]:
        for i, col_filter in enumerate([
            f'--bs=1&--speculative_tree=6,10,60&w32zhong/{ckpt}',
            f'--bs=4&--speculative_tree=6,10,60&w32zhong/{ckpt}',
            f'--bs=8&--speculative_tree=6,10,60&w32zhong/{ckpt}',
            f'--bs=1&--speculative_tree=3,1,4&w32zhong/{ckpt}',
            f'--bs=4&--speculative_tree=3,1,4&w32zhong/{ckpt}',
            f'--bs=8&--speculative_tree=3,1,4&w32zhong/{ckpt}',
        ]):
            matches = filter_jsonl(path, '*', argv=col_filter.split('&'))
            m1 = filter_json_array([m['*'] for m in matches], 'throughputs', 'avg_accept_len',
                              argv='--disable_cuda_graph=False')
            m2 = filter_json_array([m['*'] for m in matches], 'throughputs', 'avg_accept_len',
                              argv='--disable_cuda_graph=True')
            if i == 0:
                m3 = filter_jsonl(path, 'avg_accept_len', argv=[
                    '--@qwen3_4B_base_and_qwen3_4B_drafter_using_eagle2',
                    f'w32zhong/{ckpt}'
                ])
                assert len(m3) == 1
                sgl_accept_len = first_match(m1, 'avg_accept_len')
                ref_accept_len = first_match(m3, 'avg_accept_len')
                cell = f'{sgl_accept_len} | {ref_accept_len}'
                print(cell, end=', ')
            cell = f'{first_match(m1, 'throughputs')}/{first_match(m2, 'throughputs')}'
            print(cell, end=', ')
        print()


def usage_examples():
    example1 = """experiments.jsonl avg_accept_len throughputs --argv "['--bs=4','w32zhong/blooming-silence-78','--speculative_tree=3,1,4']"
    """
    print('python', sys.argv[0], 'filter_jsonl', example1)


if __name__ == "__main__":
    import fire, os
    os.environ["PAGER"] = "cat"
    fire.Fire(dict(
        filter_jsonl=filter_jsonl,
        multi_layer_results=multi_layer_results,
        usage_examples=usage_examples
    ))

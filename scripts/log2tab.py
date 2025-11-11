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
    MODELS=""
    MODELS=f"{MODELS} blooming-silence-78 laced-wood-90 trim-waterfall-88"
    MODELS=f"{MODELS} silvery-planet-91 royal-breeze-92 lemon-hill-93"
    MODELS=f"{MODELS} dulcet-cloud-94 daily-puddle-95 stellar-monkey-97 upbeat-bee-96"
    MODELS=MODELS.strip().split()
    for row, ckpt in enumerate(MODELS[::-1]):
        print(ckpt, end=', ')
        print(len(MODELS) - row, end=', ')

        filters = f'--bs=1&--speculative_tree=6,10,60&w32zhong/{ckpt}' # 0
        matches = filter_jsonl(path, 'throughputs', 'avg_accept_len', argv=filters.split('&'))
        print(first_match(matches, 'avg_accept_len'), end=', ')
        print(first_match(matches, 'throughputs'), end=', ')

        filters = f'--bs=4&--speculative_tree=6,10,60&w32zhong/{ckpt}' # 1
        matches = filter_jsonl(path, 'throughputs', argv=filters.split('&'))
        print(first_match(matches, 'throughputs'), end=', ')

        filters = f'--bs=8&--speculative_tree=6,10,60&w32zhong/{ckpt}' # 2
        matches = filter_jsonl(path, 'throughputs', argv=filters.split('&'))
        print(first_match(matches, 'throughputs'), end=', ')

        filters = f'--bs=16&--speculative_tree=6,10,60&w32zhong/{ckpt}' # 3
        matches = filter_jsonl(path, 'throughputs', argv=filters.split('&'))
        print(first_match(matches, 'throughputs'), end=', ')

        filters = f'--bs=1&--speculative_tree=3,1,4&w32zhong/{ckpt}' # 4
        matches = filter_jsonl(path, 'throughputs', 'avg_accept_len', argv=filters.split('&'))
        print(first_match(matches, 'avg_accept_len'), end=', ')
        print(first_match(matches, 'throughputs'), end=', ')

        filters = f'--bs=4&--speculative_tree=3,1,4&w32zhong/{ckpt}' # 5
        matches = filter_jsonl(path, 'throughputs', argv=filters.split('&'))
        print(first_match(matches, 'throughputs'), end=', ')

        filters = f'--bs=8&--speculative_tree=3,1,4&w32zhong/{ckpt}' # 5
        matches = filter_jsonl(path, 'throughputs', argv=filters.split('&'))
        print(first_match(matches, 'throughputs'), end=', ')

        filters = f'--bs=16&--speculative_tree=3,1,4&w32zhong/{ckpt}' # 6
        matches = filter_jsonl(path, 'throughputs', argv=filters.split('&'))
        print(first_match(matches, 'throughputs'), end=', ')

        print()


def bs1timecost_results(path):
    for model in ['meta-llama/Llama-2-7b-chat-hf', 'w32zhong/blooming-silence-78']:
        for cg in ['--disable_cuda_graph=False', '--disable_cuda_graph=True']:
            for n, tree in [(3, '--speculative_tree=3,1,4'), (6, '--speculative_tree=6,10,60')]:
                print([model, cg, tree])
                m = filter_jsonl(path, '*', argv=[model, cg, tree])
                fm = first_match(m, '*')
                j = json.loads(fm['scheduler.draft_worker.mytimer'])

                print(fm['throughputs'], end=', ')
                print(fm['avg_accept_len'], end=' | ')

                amortized_norm = j['verify']['cnt'] / j['prefill']['cnt']
                prefill_or_jit = j['prefill']['mean']
                amortized_prefill_or_jit = prefill_or_jit / amortized_norm
                print(round(amortized_prefill_or_jit, 3), end=', ')

                draft = j['draft']['mean']
                print(round(draft, 3), end=', ')

                verify = j['verify']['mean']
                print(round(verify, 3), end=', ')

                misc = j['draft_extend']['mean']
                print(round(misc, 3), end=', ')

                sum_iter = amortized_prefill_or_jit + draft + verify + misc
                print(round(sum_iter, 3), end=', ')

                real_iter = j['forward_batch_generation']['mean']
                print(round(real_iter, 3), end=', ')

                proj_throughputs = 1_000 * fm['avg_accept_len'] / sum_iter
                print(round(proj_throughputs, 3), end=', ')

                if j['draft forward']['cnt'] >= j['draft']['cnt']:
                    draft_select_topk = j['select_top_k_tokens']['mean']
                    print(round(draft_select_topk, 3), end=', ')

                    draft_forward = j['draft forward']['mean']
                    print(round(draft_forward, 3), end=', ')

                    draft_topk = j['draft topk']['mean']
                    print(round(draft_topk, 3), end=', ')

                    draft_loop_proj = draft_select_topk + draft_forward + draft_topk
                    print(round(draft_loop_proj, 3), end=', ')

                    draft_loop_real = j['draft loop']['mean']
                    print(round(draft_loop_real, 3), end=', ')

                    draft_loopheads = j['draft prepare']['mean'] + j['draft prepare inner']['mean']
                    print(round(draft_loopheads, 3), end=', ')

                    proj_draft_time = draft_loopheads + n * (draft_select_topk + draft_forward + draft_topk)
                    print(round(proj_draft_time, 3), end=', ')

                else:
                    # this case is running in CUDA graph replay
                    pass

                print()


def acceptlens_histogram(path):
    for model in [
        #'--draft_model=zhuyksir/EAGLE3-Qwen3-30B-A3B-Instruct-2507-residual-ttt',
        '--draft_model=zhuyksir/EAGLE3-Qwen3-30B-A3B-Instruct-2507-baseline',
        'w32zhong/blooming-silence-78'
    ]:
        matches = filter_jsonl(path, 'avg_accept_len', 'accept_lens_freqs', argv=[model])
        accept_lens_freqs = first_match(matches, 'accept_lens_freqs')
        print(model)
        print(matches)

        freq = {int(k): v for k, v in accept_lens_freqs.items()}
        max_len = max(freq)
        good = [0] * (max_len - 1)
        bad  = [0] * (max_len - 1)
        for k, cnt in freq.items():
            for i in range(k - 1):
                good[i] += cnt
            # don't mark a failure when fully accepted
            if k < max_len:
                bad[k - 1] += cnt

        good_rate = [good[i] / (good[i] + bad[i]) for i in range(max_len - 1)]
        print([(g, b) for g,b in zip(good, bad)])
        print([round(r, 2) for r in good_rate])

        import plotext as plt
        #samples = [k for k, cnt in freq.items() for _ in range(cnt)]
        #plt.hist(samples, bins=len(accept_lens_freqs))
        plt.bar(good_rate)
        xticks = list(range(1, len(accept_lens_freqs) + 1))
        plt.xticks(xticks, xticks)
        plt.show()
        plt.clear_figure()


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
        bs1timecost_results=bs1timecost_results,
        acceptlens_histogram=acceptlens_histogram,
        usage_examples=usage_examples
    ))

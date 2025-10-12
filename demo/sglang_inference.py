import asyncio
import threading
import time, json, os, sys, argparse
from functools import partial
from queue import Queue
from transformers import AutoTokenizer
from colorama import Fore, Style

import sglang as sgl
from sglang.srt.server_args import ServerArgs
from sglang.srt.entrypoints.http_server import launch_server
from sglang.srt.utils import kill_process_tree

import specforge_het.sglang_adapter as sgl_adapter
from specforge_het.sglang_adapter_utils import run_mtbench


class LoopRunner:
    """ Becasue submit() can be called from different threads.
        This class is designed to owns a dedicated asyncio loop
        living on a background thread, and relaying all submit
        calls in a run_coroutine_threadsafe().
    """

    def __init__(self):
        self._loop = None
        self._loop_ready = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._loop_ready.wait()

    def _run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._loop_ready.set()
        loop.run_forever()

    def scheduler_internal_state(self, llm):
        return self.submit(
            llm.tokenizer_manager.get_internal_state()
        ).result()[0]

    @property
    def loop(self):
        self._loop_ready.wait()
        return self._loop

    def submit(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

    def shutdown(self):
        def _shutdown():
            for task in asyncio.all_tasks(loop=self.loop):
                task.cancel()
            self.loop.stop()
        self.loop.call_soon_threadsafe(_shutdown)
        self._thread.join()
        self.loop.close()


def stream_generate(llm, tokenizer, prompt, sampling_params):
    queue = Queue()
    async def _stream_once():
        try:
            generator = await llm.async_generate(prompt, sampling_params, stream=True)
            async for chunk in generator:
                queue.put(chunk)
        finally:
            queue.put(None)

    runner = llm._loop_runner
    future = runner.submit(_stream_once())

    output_ids = []
    meta_info = dict(completion_tokens=0, spec_verify_ct=0, accept_tokens=[])
    while True:
        chunk = queue.get()
        if chunk is None:
            break

        new_ids = chunk["output_ids"][len(output_ids):]
        meta_info['completion_tokens'] = chunk["meta_info"]['completion_tokens']
        meta_info['spec_verify_ct'] += 1
        meta_info['accept_tokens'].append(new_ids)
        output_ids = chunk["output_ids"][:]

        print(tokenizer.decode(new_ids), end=" ", flush=True)

    # Surface any exception from the background coroutine.
    future.result()
    return tokenizer.decode(output_ids), meta_info


def batch_generate(llm, tokenizer, prompts, sampling_params):
    async def _batch_once():
        return await llm.async_generate(prompts, sampling_params, stream=False)

    runner = llm._loop_runner
    outputs = runner.submit(_batch_once()).result()

    batch_texts, batch_meta_info = [], []
    for prompt, output in zip(prompts, outputs):
        text = tokenizer.decode(output["output_ids"])
        batch_texts.append(text)
        batch_meta_info.append(output['meta_info'])
    return batch_texts, batch_meta_info


def run_one_example(llm, prompts, sampling_params, bs, warm_up=True):
    tokenizer = llm.tokenizer_manager.tokenizer

    if warm_up:
        batch_generate(llm, tokenizer, prompts, sampling_params)

    # timed run
    begin = time.perf_counter()
    if bs > 1:
        meta_info = dict(completion_tokens=0, spec_verify_ct=0)
        batch_texts, batch_meta_info = batch_generate(
            llm, tokenizer, prompts, sampling_params
        )
        for prompt, text, mi in zip(prompts, batch_texts, batch_meta_info):
            print('=' * 80)
            print([prompt])
            print(text)
            meta_info['completion_tokens'] += mi['completion_tokens']
            meta_info['spec_verify_ct'] += mi['spec_verify_ct']
    else:
        print('=' * 80)
        print(prompts)
        _, meta_info = stream_generate(llm, tokenizer, prompts[0], sampling_params)
        print()
    print('-' * 80)
    time_cost = time.perf_counter() - begin

    return meta_info, time_cost


def get_metrics(llm, meta_info, time_cost, d=3):
    m = meta_info.copy()
    sis = llm._loop_runner.scheduler_internal_state(llm)
    m['scheduler_avg_accept_len'] = round(sis['avg_spec_accept_length'], d)
    if accept_tokens := m.pop('accept_tokens', []):
        m['accept_lens'] = [len(ac) for ac in accept_tokens]
        m['accept_lens.sum'] = sum(m['accept_lens'])
        m['accept_lens.max'] = max(m['accept_lens'])
    m['avg_accept_len'] = round(m['completion_tokens'] / m['spec_verify_ct'], d)
    m['time_cost'] = round(time_cost, 2)
    m['throughputs'] = round(m['completion_tokens'] / time_cost, d)
    return m


def engine_mode(model_path, draft_model=None, dtype='auto', bs=1, tp_size=1,
    disable_cuda_graph=False, disable_radix_cache=True, max_new_tokens=None,
    temperature=0, speculative_algorithm=None, speculative_tree=(6, 10, 60),
    mtbench=None, outfile=None, log_level="INFO", one_example_warmup=False,
    skip_tokenizer_init=True, mem_fraction_static=0.7, batch_invariant=False):

    if draft_model is None:
        base_model_path, draft_model_path = sgl_adapter.adapted(model_path)
    else:
        base_model_path, draft_model_path = model_path, draft_model

    engine_kwargs = dict(
        model_path=base_model_path,
        dtype=dtype,
        tp_size=tp_size,
        cuda_graph_max_bs=bs,
        disable_cuda_graph=disable_cuda_graph,
        disable_radix_cache=disable_radix_cache,
        log_level=log_level,
        watchdog_timeout=3600,
        enable_deterministic_inference=batch_invariant,
        attention_backend='flashinfer',

        # manually set tokenizer to avoid unexpected behaviours such as `add_bos_token`:
        skip_tokenizer_init=skip_tokenizer_init,

        mem_fraction_static=mem_fraction_static,

        # speculative decoding algorithm related
        speculative_algorithm=speculative_algorithm,
        speculative_draft_model_path=draft_model_path,
        speculative_num_steps=speculative_tree[0],
        speculative_eagle_topk=speculative_tree[1],
        speculative_num_draft_tokens=speculative_tree[2],
    )
    sampling_params = {"temperature": temperature, "max_new_tokens": max_new_tokens}

    llm = sgl.Engine(**engine_kwargs)
    llm._loop_runner = LoopRunner()

    if skip_tokenizer_init:
        llm.tokenizer_manager.tokenizer = AutoTokenizer.from_pretrained(base_model_path,
            add_bos_token=False, add_eos_token=False, trust_remote_code=True)

    # Testing tokenizer template
    test_messages = [
        {"role": "system", "content": "You are a friendly chatbot!"},
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
        {"role": "user", "content": "I'd like to show off how chat templating works!"},
    ]
    tokenizer = llm.tokenizer_manager.tokenizer
    test_prompt = tokenizer.apply_chat_template(test_messages, tokenize=False)
    print('[chat template]', Fore.YELLOW, '\n' + test_prompt, Style.RESET_ALL)
    test_prompt_encoded = tokenizer.encode(test_prompt)
    print('[encode-decode]', Fore.RED, [tokenizer.decode(test_prompt_encoded)], Style.RESET_ALL)

    if mtbench is None:
        questions = [
            "Thomas is very healthy, but he has to go to the hospital every day. What could be the reasons?",
            "Who is the president of the United States?",
            "Write an essay about the future of AI.",
            "What is your favorite book?",
            "What is your least favorite book?",
            "What is your favorite programming language?",
            "What is your least favorite programming language?",
            "Write a short, neutral self-introduction for a fictional character.",
            "Provide a concise factual statement about France’s capital city."
        ]
        bs = min(len(questions), bs)
        messages = lambda question: [{"role": "user", "content": question}]
        tokenizer = llm.tokenizer_manager.tokenizer
        prompts = [
            tokenizer.apply_chat_template(
                messages(Q), tokenize=False, add_generation_prompt=True
            ) for Q in questions[:bs]
        ]

        meta_info, time_cost = run_one_example(
            llm, prompts, sampling_params, bs, warm_up=one_example_warmup)

    else:
        sys_prompt = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
        meta_info, time_cost = run_mtbench(batch_generate, llm, mtbench,
                                           sampling_params, sys_prompt='',
                                           num_threads=min(questions_num, bs))

    metrics = get_metrics(llm, meta_info, time_cost)
    for key, val in metrics.items():
        print(f'{key:>30}:', val)

    if outfile is not None:
        with open(outfile, 'a') as fh:
            j = json.dumps(dict(
                    argv=sys.argv[2:],
                    **metrics
                ), sort_keys=True)
            print(j, file=fh)

    llm._loop_runner.shutdown()
    llm.shutdown()


def server_mode():
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    raw_args = parser.parse_args(sys.argv[2:]) # skip <script name> and <Fire mode>
    if raw_args.speculative_draft_model_path is None:
        base_model_path, draft_model_path = sgl_adapter.adapted(raw_args.model_path)
        raw_args.model_path = base_model_path
        raw_args.speculative_draft_model_path = draft_model_path
    server_args = ServerArgs.from_cli_args(raw_args)
    try:
        launch_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


if __name__ == '__main__':
    import fire
    os.environ["PAGER"] = "cat"
    fire.Fire(dict(engine_mode=engine_mode, server_mode=server_mode))

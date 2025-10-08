import asyncio
import threading
import time, json, os, sys, argparse
from functools import partial
from queue import Queue

import sglang as sgl
from sglang.utils import trim_overlap
from sglang.srt.server_args import ServerArgs
from sglang.srt.entrypoints.http_server import launch_server
from sglang.srt.utils import kill_process_tree
from sglang.lang.backend.base_backend import BaseBackend
from sglang.global_config import global_config
from sglang.lang.ir import SglGen

import specforge_het.sglang_adapter as sgl_adapter


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

    text = ""
    meta_info = dict(completion_tokens=0, spec_verify_ct=0, accept_tokens=[])
    while True:
        item = queue.get()
        if item is None:
            break

        chunk_text = item["text"]
        chunk_meta_info = item["meta_info"]
        cleaned_chunk = trim_overlap(text, chunk_text)
        text += cleaned_chunk
        print(tokenizer.decode(item["output_ids"]), end=" ", flush=True)

        meta_info['completion_tokens'] = chunk_meta_info['completion_tokens']
        meta_info['spec_verify_ct'] += 1
        meta_info['accept_tokens'].append(item["output_ids"])

    # Surface any exception from the background coroutine.
    future.result()
    return text, meta_info


def batch_generate(llm, tokenizer, prompts, sampling_params):
    async def _batch_once():
        return await llm.async_generate(prompts, sampling_params, stream=False)

    runner = llm._loop_runner
    outputs = runner.submit(_batch_once()).result()

    batch_texts, batch_meta_info = [], []
    for prompt, output in zip(prompts, outputs):
        batch_texts.append(output["text"])
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


def run_mtbench(llm, questions, sampling_params, num_threads):
    global_config.enable_precache_with_tracing = False
    sgl.set_default_backend(MonkeyPatchLangBackend(llm))

    #from sglang.lang.backend.runtime_endpoint import RuntimeEndpoint
    #backend = RuntimeEndpoint('http://127.0.0.1:30000')
    #sgl.set_default_backend(backend)

    question_turns = [
        {"question_1": q["turns"][0], "question_2": q["turns"][1]}
        for q in questions
    ]

    begin = time.perf_counter()
    res = answer_mt_bench.run_batch(
        question_turns,
        **sampling_params,
        num_threads=num_threads,
        progress_bar=True
    )
    time_cost = time.perf_counter() - begin

    meta_info = dict(completion_tokens=0, spec_verify_ct=0)
    for i, Q in enumerate(question_turns):
        print('=' * 80)
        print([Q['question_1']])
        print(res[i].get_var('answer_1'))
        mi = res[i].get_meta_info('answer_1')
        meta_info['completion_tokens'] += mi['completion_tokens']
        meta_info['spec_verify_ct'] += mi['spec_verify_ct']

        print('=' * 80)
        print([Q['question_2']])
        print(res[i].get_var('answer_2'))
        mi = res[i].get_meta_info('answer_2')
        meta_info['completion_tokens'] += mi['completion_tokens']
        meta_info['spec_verify_ct'] += mi['spec_verify_ct']
    print('-' * 80)

    return meta_info, time_cost


def load_mtbench(filename):
    questions = []
    with open(filename, "r") as fin:
        for line in fin:
            obj = json.loads(line)
            questions.append(obj)
    return questions


class MonkeyPatchLangBackend(BaseBackend):
    def __init__(self, llm):
        super().__init__()
        self.monkey_patch_llm = llm

        from sglang.lang.chat_template import get_chat_template_by_model_path
        self.chat_template = get_chat_template_by_model_path(
            'meta-llama/Llama-2-7b-chat-hf'
        )

    def get_chat_template(self):
        return self.chat_template

    def generate(self, prompt, sampling_params):
        llm = self.monkey_patch_llm
        tokenizer = llm.tokenizer_manager.tokenizer
        batch_texts, batch_meta_info = batch_generate(
            llm,
            tokenizer,
            [prompt],
            sampling_params.to_srt_kwargs(),
        )
        return batch_texts[0], batch_meta_info[0]


def monkey_patch_execute(self, origin_execute_fn, state, other):
    if isinstance(other, SglGen):
        messages = self.messages_
        llm = self.backend.monkey_patch_llm
        tokenizer = llm.tokenizer_manager.tokenizer
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        text, meta_info = self.backend.generate(
            self.text_, sampling_params=self.default_sampling_para
        )

        name = other.name
        self.text_ += text
        self.variables[name] = text
        self.meta_info[name] = meta_info
        self.variable_event[name].set()

    else:
        return origin_execute_fn(other)


@sgl.function
def answer_mt_bench(s, question_1, question_2):
    s.stream_executor._execute = partial(
        monkey_patch_execute,
        s.stream_executor,
        s.stream_executor._execute,
        s
    )
    s += sgl.system(
        "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
    )
    s += sgl.user(question_1)
    s += sgl.assistant(sgl.gen("answer_1"))
    s += sgl.user(question_2)
    s += sgl.assistant(sgl.gen("answer_2"))


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
    mtbench=None, outfile=None, log_level="INFO", one_example_warmup=False):

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

        speculative_algorithm=speculative_algorithm,
        speculative_draft_model_path=draft_model_path,
        speculative_num_steps=speculative_tree[0],
        speculative_eagle_topk=speculative_tree[1],
        speculative_num_draft_tokens=speculative_tree[2],
    )
    sampling_params = {"temperature": temperature, "max_new_tokens": max_new_tokens}

    llm = sgl.Engine(**engine_kwargs)
    llm._loop_runner = LoopRunner()
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
        questions_jsonl, *after = mtbench.split(':')
        questions_num = 80 if len(after) == 0 else int(after[0])
        questions = load_mtbench(questions_jsonl)[: questions_num]
        meta_info, time_cost = run_mtbench(llm, questions, sampling_params,
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

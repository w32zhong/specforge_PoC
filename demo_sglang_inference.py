import time
import asyncio
from transformers import AutoTokenizer

import sglang as sgl
from sglang.utils import trim_overlap

import os, sys, argparse
from sglang.srt.server_args import ServerArgs
from sglang.srt.entrypoints.http_server import launch_server
from sglang.srt.server_args import prepare_server_args
from sglang.srt.utils import kill_process_tree

import specforge_het.sglang_adapter as sf_adapter


async def generate(llm, tokenizer, prompt, sampling_params):
    final_text = ""
    cnt_tokens = []
    print(prompt)
    generator = await llm.async_generate(prompt, sampling_params, stream=True)
    async for chunk in generator:
        chunk_text = chunk["text"]
        cleaned_chunk = trim_overlap(final_text, chunk_text)
        final_text += cleaned_chunk
        print(tokenizer.decode(chunk['output_ids']), end="", flush=True)
        cnt_tokens.append(len(chunk['output_ids']))
    return cnt_tokens


def batch_generate(llm, tokenizer, prompts, sampling_params):
    cnt_tokens = []
    outputs = llm.generate(prompts, sampling_params)
    for prompt, output in zip(prompts, outputs):
        print("===============================")
        print(f"Prompt: {prompt}\nGenerated text: {output['text']}")
        tokens = tokenizer.encode(output['text'])
        cnt_tokens.append(len(tokens))
    return cnt_tokens


def direct_mode(model_path, speculative_algorithm=None,
         speculative_tree=(6, 10, 60), bs=1, tp_size=1, disable_cuda_graph=False):
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
    ][:bs]
    messages = lambda question: [{"role": "user", "content": question}]

    base_model_path, draft_model_path = sf_adapter.adapted(model_path)

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    prompts = [
        tokenizer.apply_chat_template(
            messages(Q), tokenize=False, add_generation_prompt=True
        ) for Q in questions
    ]

    llm = sgl.Engine(
        model_path=base_model_path,
        tp_size=tp_size,
        cuda_graph_max_bs=bs,
        disable_cuda_graph=disable_cuda_graph,

        speculative_algorithm=speculative_algorithm,
        speculative_draft_model_path=draft_model_path,
        speculative_num_steps=speculative_tree[0],
        speculative_eagle_topk=speculative_tree[1],
        speculative_num_draft_tokens=speculative_tree[2],
    )


    sampling_params = {"temperature": 0, "max_new_tokens": 8000}

    # Use a shared event loop for both sync and async paths so background
    # tokenizer tasks stay on the same loop even after the warmup request.
    loop = asyncio.get_event_loop()
    try:
        # warm-up run
        if bs > 1:
            batch_generate(llm, tokenizer, prompts, sampling_params)
        else:
            loop.run_until_complete(
                generate(llm, tokenizer, prompts[0], sampling_params)
            )

        # timed run
        print('\n', '=' * 30)
        begin = time.perf_counter()
        if bs > 1:
            cnt_tokens = batch_generate(llm, tokenizer, prompts, sampling_params)
        else:
            cnt_tokens = loop.run_until_complete(
                generate(llm, tokenizer, prompts[0], sampling_params)
            )
            if cnt_tokens:
                cnt_tokens.pop(0)
        time_cost = time.perf_counter() - begin

    finally:
        llm.shutdown()
        loop.close()

    print()
    print(cnt_tokens)
    print('tokens and time:', sum(cnt_tokens), time_cost)
    print('e2e throughputs:', sum(cnt_tokens) / time_cost)
    print('max accept length:', max(cnt_tokens))
    print('min accept length:', min(cnt_tokens))
    print('avg accept length:', sum(cnt_tokens) / len(cnt_tokens))


def server_mode():
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    raw_args = parser.parse_args(sys.argv[2:]) # skip <script name> and <Fire mode>
    base_model_path, draft_model_path = sf_adapter.adapted(raw_args.model_path)
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
    fire.Fire(dict(direct_mode=direct_mode, server_mode=server_mode))

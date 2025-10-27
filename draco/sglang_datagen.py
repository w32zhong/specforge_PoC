import os, time
import multiprocessing as mp
import torch
from transformers import AutoConfig
import sglang


def worker_loop(ready_barrier, tasks, results, worker_id,
                model_path, capture_states_of_layers):
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{worker_id}"

    llm = sglang.Engine(
        model_path=model_path,
        dtype='bfloat16',
        mem_fraction_static=0.7,
        capture_states_of_layers=capture_states_of_layers,
        enable_return_hidden_states=True
    )

    ready_barrier.wait()
    while True:
        task = tasks.get()
        if task is None:
            break

        print(f'worker#{worker_id} task: {task}')
        res = llm.generate(
            sampling_params=dict(max_new_tokens=0, temperature=0),
            return_hidden_states=True, **task
        )
        hidden_states = res['meta_info'].pop('hidden_states')
        hidden_states = torch.Tensor(hidden_states)
        print(f'worker#{worker_id}: {res} -> tensor:', hidden_states.shape)

    llm.shutdown()
    print(f'worker#{worker_id}: Shutdown.')


def main(num_workers, model_path='Qwen/Qwen3-4B-Instruct-2507', capture_states_of_layers=None):
    tasks, results = mp.Queue(), mp.Queue()
    ready_barrier = mp.Barrier(num_workers + 1)

    if capture_states_of_layers is None:
        model_config = AutoConfig.from_pretrained(model_path)
        num_layers = model_config.num_hidden_layers
        capture_states_of_layers = f'1,{num_layers // 2}'

    worker_pool = []
    for i in range(num_workers):
        p = mp.Process(target=worker_loop, args=(ready_barrier, tasks, results, i,
                model_path, capture_states_of_layers), daemon=False)
        p.start()
        worker_pool.append(p)

    ready_barrier.wait()
    print('all workers ready.')

    generate_kwargs = dict(
        prompt = 'Why people age?',
    )
    tasks.put(generate_kwargs)
    time.sleep(3)
    generate_kwargs = dict(
        prompt = 'Explain quantum computing in simple terms.',
    )
    tasks.put(generate_kwargs)

    for _ in worker_pool: tasks.put(None)
    for worker in worker_pool: worker.join()
    print('Quit')


if __name__ == '__main__':
    import fire
    os.environ["PAGER"] = "cat"
    fire.Fire(main)

# specforge PoC
This is a Proof-of-Concept implementation for a speculative decoding framework that supports heterogeneous (het.) base and draft models.

For a demo of what a heterogeneous speculative decoding model is, please check out a [demo here](./demo/concept.py).

## Setup
```
git submodule update --init --recursive --progress
pip install -r specforge_het/requirements.txt
```

## Training Data Generation (Optional)
Generate dataset:
```sh
rm -rf output/datasets/ds_Llama-2-7b-chat-hf*
CUDA_VISIBLE_DEVICES=0 python -m specforge_het.data_gen gen_dataset --@llama2_7b --ds_range 0,17500
CUDA_VISIBLE_DEVICES=1 python -m specforge_het.data_gen gen_dataset --@llama2_7b --ds_range 17500,35000
CUDA_VISIBLE_DEVICES=2 python -m specforge_het.data_gen gen_dataset --@llama2_7b --ds_range 35000,52500
CUDA_VISIBLE_DEVICES=3 python -m specforge_het.data_gen gen_dataset --@llama2_7b --ds_range 52500,69999
python -m specforge_het.data_gen merge_datasets \
    output/datasets/ds_Llama-2-7b-chat-hf \
    output/datasets/ds_Llama-2-7b-chat-hf__range*
```

Useful options:
* Use `--dataset_generation.output_dir` to specify an alternative output directory
* To specify a different source dataset and save prefix, e.g., `--dataset_generation.ds_prefix eval_ds_ --dataset_generation.sharegpt_path w32zhong/qwen3_moe_30A3B_instr_2507_mt-bench` (useful for generating evaluation dataset)

## Training
```sh
CUDA_VISIBLE_DEVICES=0 python -m specforge_het.train \
    --dataset.path /mnt/asus_card/temp_llama_dataset/datasets/ds_Llama-2-7b-chat-hf \
    --training.eval_strategy no --modeling.dtype torch.float32 \
    --@llama2_7b_base_and_llama2_7b_drafter_using_eagle2 \
    # --training.report_to wandb --training.project eagle4
```

Useful options:
* If you choose to use EAGLE-format offline-training data, replace `--dataset.path <path>` to `--dataset.read_eagle_format --dataset.path <path/to/sharegpt_0_67999_mufp16>`
* To add evaluation data: `--dataset.eval_path <path>`.
* To adjust modeling: `--modeling.init_speculative_algorithm "'EagleV2','dict(draft_layers=2, vloss_w=0.6, ploss_w=0.4)'"`

Alternatively, download pre-trained models from my HuggingFace hub: https://huggingface.co/w32zhong/models

A robust and fast way to download HuggingFace models is using [hfdownloader](https://github.com/bodaay/HuggingFaceModelDownloader).
This is the reason you see some of the command lines containing example model paths under a folder called `hfdownloader`.

## Built-In Inference
Currently, the built-in inference only guarantees correctness algorithmically, without any optimizations for inference speed.
```sh
CUDA_VISIBLE_DEVICES=0,1 python -m specforge_het.inference \
    --@qwen3_4B_base_and_qwen3_4B_drafter_using_eagle2 \
    --modeling.model_path /mnt/asus_card/hfdownloader/w32zhong_deft-bee-66
```

To use a stand-alone draft checkpoint:
```sh
CUDA_VISIBLE_DEVICES=0,1 python -m specforge_het.inference \
    --@llama2_7b_base_and_llama2_7b_drafter_using_eagle2 \
    --modeling.stand_alone_draft_model_path yuhuili/EAGLE-llama2-chat-7B \
    --modeling.stand_alone_draft_model_key_adapt yuhuili
```

To be free from specifying model architecture, use the saved JSON config:
```sh
CUDA_VISIBLE_DEVICES=0 python -m specforge_het.inference \
    --use_saved_json_config output/stellar-monkey-97/specforge_het.json \
    --modeling.stand_alone_draft_model_path output/stellar-monkey-97/checkpoint-84510/draft_model
```
Or, even better, with some CLI utilities, specify the checkpoint directory only:
```sh
read -p "# " path; echo $path | CUDA_VISIBLE_DEVICES=0 \
    xargs -I {} python -m specforge_het.inference \
    --use_saved_json_config {}/../specforge_het.json \
    --modeling.stand_alone_draft_model_path {}/draft_model
# output/stellar-monkey-97/checkpoint-84510/
```

The inference script can be directly compared to the script from the original EAGLE reference implementation:
```sh
cd eagle_v2/eagle
git checkout v2
CUDA_VISIBLE_DEVICES=0,1 python application/test.py
```

Useful options:
* To specify GPU VRAM allocations: `--modeling.max_memory "{0: '17.5GiB', 1: '17GiB', 2: '13.5GiB', 3: '17GiB'}"`

## SGLang Inference
For a demo of using SGLang as an inference engine for a model trained by this framework:
```sh
# engine mode (single pass)
CUDA_VISIBLE_DEVICES=0 python -m demo.sglang_inference engine_mode \
    /mnt/asus_card/hfdownloader/w32zhong_deft-bee-66 \
    --dtype bfloat16 --disable_cuda_graph \
    --speculative_algorithm EAGLE # EAGLE-v2
```

To run an official MT-Bench evaluation pipeline, use server mode:
```sh
# server mode
CUDA_VISIBLE_DEVICES=0 python -m demo.sglang_inference server_mode \
    --model output/deft-bee-66/ \
    --speculative-algo EAGLE \
    --speculative-num-steps 6 \
    --speculative-eagle-topk 10 \
    --speculative-num-draft-tokens 60 \
    --dtype bfloat16 --mem-fraction-static 0.7
    # Some useful debug options:
    # --disable-cuda-graph
    # --disable-radix-cache
```
```
cd path/to/sglang/benchmark/mtbench
wget -O question.jsonl https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl
python bench_sglang_eagle.py --parallel 1 --num-questions 10
```

Alternatively, use engine mode to evaluate MT-Bench without running two commands:
```sh
CUDA_VISIBLE_DEVICES=0 python -m demo.sglang_inference engine_mode \
    meta-llama/Llama-2-7b-chat-hf \
    --draft_model lmsys/sglang-EAGLE-llama2-chat-7B \
    --dtype bfloat16 --disable_cuda_graph \
    --speculative_algorithm EAGLE --max_new_tokens 2048 \
    --log_level ERROR --mtbench question.jsonl --outfile out.log

```
(the example here is for evaluating `lmsys/sglang-EAGLE-llama2-chat-7B` which is
the same checkpoint of original EAGLE-v2 paper but with config.json `architecture`
field modified to indicate SGLang that we are using an EAGLE speculative draft model)

Because SGLang is somewhat a complicated software stack and is hard to install,
it is recommended to use a container build. In this case, a good workflow would be:
```sh
source docker_utils.sh
build specforge_het_and_sglang
# run a detached container in the background
HF_TOKEN=YOUR_TOKEN docker run -d \
    --env HF_TOKEN=$HF_TOKEN --gpus all --ipc=host \
    -v $HOME/.cache:/root/.cache -v `pwd`/nvim_plugins:/root/.local/share/nvim \
    -v `pwd`:/workspace/mnt -v ~/.codex:/root/.codex \
    -v /mnt/asus_card/hfdownloader:/workspace/hfdownloader \
    -it specforge_het_and_sglang /bin/bash
docker ps # find this active container
docker exec -it <container name or ID> bash # attach to it
```

Note: the default underlying backend (FlashInfer) will build cache files under `~/.cache/flashinfer` on inference startup, be sure to remove them if you want to measure the real load-up time.

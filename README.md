# specforge PoC
This is a Proof-of-Concept implementation for speculative decoding framework that supports heterogeneous (het.) base and draft models.

For a demo of what is heterogeneous models, refer to the [demo.py](./demo.py) file.

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

## Inference
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

To free from specifying model architecture, use saved JSON config:
```sh
CUDA_VISIBLE_DEVICES=0 python -m specforge_het.inference \
    --use_saved_json_config output/stellar-monkey-97/specforge_het.json \
    --modeling.stand_alone_draft_model_path output/stellar-monkey-97/checkpoint-84510/draft_model
```
or even better, with some CLI utilities, specify the checkpoint directory only:
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

# specforge PoC
This is a Proof-of-Concept implementation for speculative decoding framework that supports heterogeneous (het.) base and draft models.

For a demo of what is heterogeneous models, refer to the [demo.py](./demo.py) file.

## Training Data Generation (Optional)
Generate dataset (use `--dataset_generation.output_dir` to specify an alternative output directory):
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

## Training
```sh
CUDA_VISIBLE_DEVICES=0 python -m specforge_het.train \
    --dataset.path /mnt/asus_card/temp_llama_dataset/datasets/ds_Llama-2-7b-chat-hf/
    --training.eval_strategy no --modeling.dtype torch.float32 \
    --@llama2_7b_base_and_llama2_7b_drafter_using_eagle2 \
    # --training.report_to wandb --training.project eagle4
```

If you choose to use EAGLE-format offline-training data, replace `--dataset.path <path>` to `--dataset.read_eagle_format --dataset.path <path/to/sharegpt_0_67999_mufp16>`

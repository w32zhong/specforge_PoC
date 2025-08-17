# specforge PoC
This is a Proof-of-Concept implementation for speculative decoding framework that supports heterogeneous (het.) base and draft models.

## Training Data Generation
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

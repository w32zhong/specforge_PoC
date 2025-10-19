MODEL=w32zhong/deft-bee-66
DATA_RANGE=:1

# Ada devices
dev=1
CUDA_VISIBLE_DEVICES=$dev flock gpu${dev}.lock \
  python -m demo.sglang_inference engine_mode \
    --mtbench question.jsonl${DATA_RANGE} \
    --outfile ./ada_sgl.log \
    --max_new_tokens 2048 --dtype bfloat16 \
    --disable_cuda_graph True \
    --speculative_algorithm EAGLE \
    $MODEL &

dev=2
CUDA_VISIBLE_DEVICES=$dev flock gpu${dev}.lock \
  python -m specforge_het.inference \
    --mtbench question.jsonl${DATA_RANGE} \
    --outfile ./ada_het.log \
    --inference.max_new_tokens 2048 \
    --modeling.dtype torch.bfloat16 \
    --@qwen3_4B_base_and_qwen3_4B_drafter_using_eagle2 \
    --modeling.model_path $MODEL &

# Blackwell devices
dev=4
CUDA_VISIBLE_DEVICES=$dev flock gpu${dev}.lock \
  python -m demo.sglang_inference engine_mode \
    --mtbench question.jsonl${DATA_RANGE} \
    --outfile ./blackwell_sgl.log \
    --max_new_tokens 2048 --dtype bfloat16 \
    --disable_cuda_graph True \
    --speculative_algorithm EAGLE \
    $MODEL &

dev=4
CUDA_VISIBLE_DEVICES=$dev flock gpu${dev}.lock \
  python -m specforge_het.inference \
    --mtbench question.jsonl${DATA_RANGE} \
    --outfile ./blackwell_het.log \
    --inference.max_new_tokens 2048 \
    --modeling.dtype torch.bfloat16 \
    --@qwen3_4B_base_and_qwen3_4B_drafter_using_eagle2 \
    --modeling.model_path $MODEL &

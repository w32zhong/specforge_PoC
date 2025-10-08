TOTAL_GPUS=8

MODELS="$MODELS blooming-silence-78 laced-wood-90 trim-waterfall-88"
MODELS="$MODELS silvery-planet-91 royal-breeze-92 lemon-hill-93"
MODELS="$MODELS dulcet-cloud-94 daily-puddle-95 stellar-monkey-97 upbeat-bee-96"
MODELS="$MODELS lucky-valley-98 absurd-cosmos-99 rich-snow-100"
MODELS="$MODELS daily-bee-101 soft-durian-102 radiant-night-103"
MODELS="$MODELS radiant-salad-104 trim-blaze-105"

set -e
for model in $MODELS; do
  if [[ -e output/$model/checkpoint-84510 ]]; then
    path=output/$model/checkpoint-84510
  elif [[ -e output/$model/checkpoint-81500 ]]; then
    path=output/$model/checkpoint-81500
  elif [[ -e output/$model/checkpoint-77500 ]]; then
    path=output/$model/checkpoint-77500
  elif [[ -e output/$model/checkpoint-76000 ]]; then
    path=output/$model/checkpoint-76000
  else
    path=output/$model/checkpoint-80000
  fi
  ls $path
  MODEL_PATHS="$MODEL_PATHS $path"
done
set +e

rm -f results_*.log
cnt=0
for model_path in $MODEL_PATHS; do
  for bs in 1 4 8; do
    for tree in 6,10,60 3,1,4; do
      dev=$((cnt % $TOTAL_GPUS))
      set -x
      CUDA_VISIBLE_DEVICES=$dev flock grid_search_gpu${dev}.lock \
        python demo_sglang_inference.py engine_mode --bs $bs \
        --dtype bfloat16 --disable_cuda_graph --speculative_algorithm EAGLE \
        --mtbench question.jsonl --outfile ./grid_search_gpu${dev}.log \
        -speculative_tree $tree $model_path &
      set +x
      let "cnt += 1"
    done
  done
done

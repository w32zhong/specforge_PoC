source $(dirname $0)/experiment_utils.sh

GPUS=$(experiment_argparse --gpus 1 $@)
GPU0=$(experiment_argparse --gpu0 0 $@)
HGF_USER=$(experiment_argparse --user w32zhong $@)
DATA_RANGE=$(experiment_argparse --range "" $@)
SESSION_END=$(experiment_argparse --session-end "exit" $@)

MODELS="$MODELS blooming-silence-78 laced-wood-90 trim-waterfall-88"
MODELS="$MODELS silvery-planet-91 royal-breeze-92 lemon-hill-93"
MODELS="$MODELS dulcet-cloud-94 daily-puddle-95 stellar-monkey-97 upbeat-bee-96"
#MODELS="$MODELS lucky-valley-98 absurd-cosmos-99 rich-snow-100"
#MODELS="$MODELS daily-bee-101 soft-durian-102 radiant-night-103"
#MODELS="$MODELS radiant-salad-104 trim-blaze-105"

for model in $MODELS; do
  MODEL_PATHS="$MODEL_PATHS $HGF_USER/$model"
done

rm -f gpu_*.lock
cnt=0
for model_path in $MODEL_PATHS; do
  for bs in 1 4 8; do
    for tree in 6,10,60 3,1,4; do
      for disable_cuda_graph in False; do
        for tp_size in 2; do
          devices=$(experiment_alloc_devices $cnt $GPU0 $GPUS $tp_size)
          let 'cnt+=1'
          echo CUDA_VISIBLE_DEVICES=$devices
          session=$model_path-bs$bs-$tree-CG$disable_cuda_graph-tp$tp_size
          session=$(experiment_sanitize $session)
          if tmux has-session -t "exp_$session"; then
            echo "session exists: exp_$session"; continue
          fi
          experiment_session $session \
            "(flock 200; CUDA_VISIBLE_DEVICES=$devices \
              python -m demo.sglang_inference engine_mode \
                --mtbench question.jsonl${DATA_RANGE} \
                --outfile ./output/$session.log \
                --disallow_outfile_overwrite \
                --bs $bs --max_new_tokens 2048 \
                --dtype bfloat16 --tp_size $tp_size \
                --disable_cuda_graph $disable_cuda_graph \
                --speculative_algorithm EAGLE --speculative_tree $tree \
                $model_path;
                echo 'UNLOCK'; flock --unlock 200) 200>gpu_${devices}.lock"
          experiment_session $session $SESSION_END
        done
      done
    done
  done
done

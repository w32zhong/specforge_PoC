HGF_USER=w32zhong
TOTAL_GPUS=8
DATA_RANGE= #:1

MODELS="$MODELS blooming-silence-78 laced-wood-90 trim-waterfall-88"
MODELS="$MODELS silvery-planet-91 royal-breeze-92 lemon-hill-93"
MODELS="$MODELS dulcet-cloud-94 daily-puddle-95 stellar-monkey-97 upbeat-bee-96"
#MODELS="$MODELS lucky-valley-98 absurd-cosmos-99 rich-snow-100"
#MODELS="$MODELS daily-bee-101 soft-durian-102 radiant-night-103"
#MODELS="$MODELS radiant-salad-104 trim-blaze-105"

for model in $MODELS; do
  MODEL_PATHS="$MODEL_PATHS $HGF_USER/$model"
done

rm -f gpu*.lock
set -e
cnt=0

for model_path in $MODEL_PATHS; do
  for bs in 1 4 8 16 32; do
    for tree in 6,10,60 3,1,4; do
      for disable_cuda_graph in True False; do
        dev=$((cnt % $TOTAL_GPUS))
        let "cnt += 1"
        session_ID=$model_path-bs$bs-$tree-$disable_cuda_graph
        if tmux has-session -t $session_ID; then
          echo "session exists: $session_ID"
          continue;
        fi
        set -x
        tmux new-session -d -s $session_ID -- bash -c "
          (flock 200;
            CUDA_VISIBLE_DEVICES=$dev \
            python -m demo.sglang_inference engine_mode --bs $bs \
              --mtbench question.jsonl${DATA_RANGE} \
              --outfile ./output/$session_ID.log \
              --max_new_tokens 2048 \
              --dtype bfloat16 \
              --disable_cuda_graph $disable_cuda_graph \
              --speculative_algorithm EAGLE --speculative_tree $tree \
              $model_path;
	   flock --unlock 200) 200>gpu${dev}.lock
        "
        set +x
      done
    done
  done
done

#for model_path in $MODEL_PATHS; do
#    dev=$((cnt % $TOTAL_GPUS))
#    set -x
#    CUDA_VISIBLE_DEVICES=$dev flock gpu${dev}.lock \
#      python -m specforge_het.inference \
#        --mtbench question.jsonl${DATA_RANGE} \
#        --outfile ./gpu${dev}.log \
#        --inference.max_new_tokens 2048 \
#        --modeling.dtype torch.bfloat16 \
#        --@qwen3_4B_base_and_qwen3_4B_drafter_using_eagle2 \
#        --modeling.model_path $model_path &
#    set +x
#    let "cnt += 1"
#done

##### Tips to manage sessions #####
# To kill all sessions by PIDs: pkill -f $HGF_USER
# To inspect a session: tmux capture-pane -pt <session_ID>
# To list all sessions: tmux list-sessions -F '#S' -f "#{m:$HGF_USER*,#S}"
# Pipe above to kill sessions: | xargs -n 1 tmux kill-session -t

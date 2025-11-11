source $(dirname $0)/experiment_utils.sh

TP_SIZE=$(experiment_argparse --tp_size 1 $@)
GPUS=$(experiment_argparse --gpus $TP_SIZE $@)
GPU0=$(experiment_argparse --gpu0 0 $@)
DATA_RANGE=$(experiment_argparse --range "" $@)
SESSION_END=$(experiment_argparse --session-end "exit" $@)
ALGORITHM=$(experiment_argparse --algorithm "EAGLE3" $@)

rm -f gpu_*.lock
cnt=0

## EAGLE-3 ##
for models in \
  "Qwen/Qwen3-30B-A3B-Instruct-2507 --draft_model zhuyksir/EAGLE3-Qwen3-30B-A3B-Instruct-2507-residual-ttt" \
  "Qwen/Qwen3-30B-A3B-Instruct-2507 --draft_model zhuyksir/EAGLE3-Qwen3-30B-A3B-Instruct-2507-baseline" \
  ; do


## EAGLE-2 ##
#for models in \
#  "meta-llama/Llama-2-7b-chat-hf --draft_model lmsys/sglang-EAGLE-llama2-chat-7B"; do

  for bs in 1; do
    for tree in 6,10,60; do
      for disable_cuda_graph in True; do
        for tp_size in $TP_SIZE; do
          devices=$(experiment_alloc_devices $cnt $GPU0 $GPUS $tp_size)
          let 'cnt+=1'
          echo CUDA_VISIBLE_DEVICES=$devices
          session="${models}_acceptlens_histogram"
          session=$(experiment_sanitize "$session")
          if tmux has-session -t "exp_$session"; then
            echo "session exists: exp_$session"; continue
          fi
          echo $models
          experiment_session $session \
            "(flock 200; CUDA_VISIBLE_DEVICES=$devices \
              python -m demo.sglang_inference engine_mode $models \
                --dtype bfloat16 --disable_cuda_graph $disable_cuda_graph \
                --speculative_algorithm $ALGORITHM --speculative_tree $tree \
                --bs $bs --tp_size $tp_size --max_new_tokens 2048 \
                --mtbench question.jsonl${DATA_RANGE} --stream_if_bs1 \
                --outfile ./output/$session.log;
                echo 'UNLOCK'; flock --unlock 200) 200>gpu_${devices}.lock"
          experiment_session $session $SESSION_END
        done
      done
    done
  done
done

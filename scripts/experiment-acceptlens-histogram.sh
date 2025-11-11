source $(dirname $0)/experiment_utils.sh

TP_SIZE=$(experiment_argparse --tp_size 1 $@)
GPUS=$(experiment_argparse --gpus 1 $@)
GPU0=$(experiment_argparse --gpu0 0 $@)
DATA_RANGE=$(experiment_argparse --range "" $@)
SESSION_END=$(experiment_argparse --session-end "exit" $@)

rm -f gpu_*.lock
cnt=0

for models in \
  "meta-llama/Llama-3.1-8B-Instruct --draft_model jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B --speculative_algorithm EAGLE3" \
  "meta-llama/Llama-2-7b-chat-hf --draft_model lmsys/sglang-EAGLE-llama2-chat-7B --speculative_algorithm EAGLE"; do

  for bs in 1; do
    for tree in 6,10,60 6,1,7; do
      for disable_cuda_graph in True; do
        for tp_size in $TP_SIZE; do
          devices=$(experiment_alloc_devices $cnt $GPU0 $GPUS $tp_size)
          let 'cnt+=1'
          echo CUDA_VISIBLE_DEVICES=$devices
          session="${models}_tree${tree}"
          session=$(experiment_sanitize "$session")
          if tmux has-session -t "exp_$session"; then
            echo "session exists: exp_$session"; continue
          fi
          experiment_session $session \
            "(flock 200; CUDA_VISIBLE_DEVICES=$devices SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
              python -m demo.sglang_inference engine_mode $models \
                --dtype bfloat16 --disable_cuda_graph $disable_cuda_graph \
                --speculative_tree $tree \
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

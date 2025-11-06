source $(dirname $0)/experiment_utils.sh

GPUS=$(experiment_argparse --gpus 2 $@)
GPU0=$(experiment_argparse --gpu0 0 $@)
DATA_RANGE=$(experiment_argparse --range ":70" $@)
SESSION_END=$(experiment_argparse --session-end "exit" $@)

rm -f gpu_*.lock
cnt=0
for models in \
  "meta-llama/Llama-2-7b-chat-hf --draft_model lmsys/sglang-EAGLE-llama2-chat-7B" \
  "w32zhong/blooming-silence-78" \
  ; do
  for bs in 1; do
    for tree in 6,10,60 3,1,4; do
      for disable_cuda_graph in True False; do
        for tp_size in 2; do
          devices=$(experiment_alloc_devices $cnt $GPU0 $GPUS $tp_size)
          let 'cnt+=1'
          echo CUDA_VISIBLE_DEVICES=$devices
          session="${models}_bs${bs}_tree${tree}_noCG${disable_cuda_graph}"
          session=$(experiment_sanitize "$session")
          if tmux has-session -t "exp_$session"; then
            echo "session exists: exp_$session"; continue
          fi
          echo $models
          experiment_session $session \
            "(flock 200; CUDA_VISIBLE_DEVICES=$devices \
              python -m demo.sglang_inference engine_mode $models \
                --dtype bfloat16 --disable_cuda_graph $disable_cuda_graph \
                --speculative_algorithm EAGLE --speculative_tree $tree \
                --bs $bs --tp_size $tp_size --max_new_tokens 2048 \
                --mtbench question.jsonl${DATA_RANGE} \
                --outfile ./output/$session.log;
                echo 'UNLOCK'; flock --unlock 200) 200>gpu_${devices}.lock"
          experiment_session $session $SESSION_END
        done
      done
    done
  done
done

source $(dirname $0)/experiment_utils.sh

TP_SIZE=$(experiment_argparse --tp_size 1 $@)
GPUS=$(experiment_argparse --gpus 1 $@)
GPU0=$(experiment_argparse --gpu0 0 $@)
DATA_RANGE=$(experiment_argparse --range ":70" $@)
SESSION_END=$(experiment_argparse --session-end "exit" $@)

mkdir -p ./output/pondering_eagle
rm -f gpu_*.lock
cnt=0

run() {
  devices=$1
  session=$2
  shift 2
  experiment_session $session \
    "(flock 200; CUDA_VISIBLE_DEVICES=$devices SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
      python -m demo.sglang_inference engine_mode $@ \
        --dtype bfloat16 --disable_cuda_graph True \
        --bs 1 --tp_size $TP_SIZE --max_new_tokens 2048 \
        --disallow_outfile_overwrite \
        --mtbench question.jsonl${DATA_RANGE} \
        --outfile ./output/pondering_eagle/$session.log; \
        echo 'UNLOCK'; flock --unlock 200) 200>gpu_${devices}.lock"
  experiment_session $session $SESSION_END
}

# Vanilla baseline
for model in "meta-llama/Meta-Llama-3.1-8B-Instruct"; do
  devices=$(experiment_alloc_devices $cnt $GPU0 $GPUS $TP_SIZE)
  let 'cnt+=1'
  session="vanilla_baseline_${model}"
  session=$(experiment_sanitize "$session")
  if tmux has-session -t "exp_$session"; then
    echo "session exists: exp_$session"; continue
  fi
  run $devices $session $model --draft_model none
done

# EAGLE baseline
for model in "w32zhong/jolly-elevator__pondering_baseline_ep1step120k"; do
  for tree in 5,1,6 10,1,11; do
    devices=$(experiment_alloc_devices $cnt $GPU0 $GPUS $TP_SIZE)
    let 'cnt+=1'
    session="speculative_baseline_${model}_${tree}"
    session=$(experiment_sanitize "$session")
    if tmux has-session -t "exp_$session"; then
      echo "session exists: exp_$session"; continue
    fi
    run $devices $session \
      meta-llama/Meta-Llama-3.1-8B-Instruct \
      --draft_model $model \
      --speculative_algorithm EAGLE3 \
      --speculative_tree $tree
  done
done

# Pondering EAGLE baseline
for model in \
  "w32zhong/toasty-durian-227__tau3" \
  ; do
  for tree in 10,1,11; do
    for pondering_threshold in 0.6 0.7 0.8 0.9; do
      for pondering_options in random; do
        devices=$(experiment_alloc_devices $cnt $GPU0 $GPUS $TP_SIZE)
        let 'cnt+=1'
        session="pondering_baseline_${model}_${tree}_${pondering_threshold}"
        session=$(experiment_sanitize "$session")
        if tmux has-session -t "exp_$session"; then
          echo "session exists: exp_$session"; continue
        fi
        run $devices $session \
          meta-llama/Meta-Llama-3.1-8B-Instruct \
          --draft_model $model \
          --speculative_algorithm EAGLE3 \
          --speculative_tree $tree \
          --speculative_pondering_threshold $pondering_threshold \
          --speculative_pondering_options $pondering_options
      done
    done
  done
done

# Pondering EAGLE quick model filter
for model in \
  "w32zhong/toasty-durian-227__tau3" \
  "w32zhong/lilac-microwave-225__pondering_tau30" \
  "w32zhong/kind-star-226__pondering_tau300" \
  ; do
  for tree in 10,1,11 20,1,21; do
    for pondering_threshold in 0.8; do
      for pondering_options in default; do
        devices=$(experiment_alloc_devices $cnt $GPU0 $GPUS $TP_SIZE)
        let 'cnt+=1'
        session="pondering_models_${model}_${tree}_${pondering_threshold}"
        session=$(experiment_sanitize "$session")
        if tmux has-session -t "exp_$session"; then
          echo "session exists: exp_$session"; continue
        fi
        run $devices $session \
          meta-llama/Meta-Llama-3.1-8B-Instruct \
          --draft_model $model \
          --speculative_algorithm EAGLE3 \
          --speculative_tree $tree \
          --speculative_pondering_threshold $pondering_threshold \
          --speculative_pondering_options $pondering_options
      done
    done
  done
done

# Pondering EAGLE grid search
for model in \
  "w32zhong/toasty-durian-227__tau3" \
  ; do
  for tree in 10,1,11 20,1,21; do
    for pondering_threshold in 0.5 0.6 0.7 0.8 0.9 1.0; do
      for pondering_options in default; do
        devices=$(experiment_alloc_devices $cnt $GPU0 $GPUS $TP_SIZE)
        let 'cnt+=1'
        session="pondering_grid_search_${model}_${tree}_${pondering_threshold}"
        session=$(experiment_sanitize "$session")
        if tmux has-session -t "exp_$session"; then
          echo "session exists: exp_$session"; continue
        fi
        run $devices $session \
          meta-llama/Meta-Llama-3.1-8B-Instruct \
          --draft_model $model \
          --speculative_algorithm EAGLE3 \
          --speculative_tree $tree \
          --speculative_pondering_threshold $pondering_threshold \
          --speculative_pondering_options $pondering_options
      done
    done
  done
done

echo "total experiments: $cnt"

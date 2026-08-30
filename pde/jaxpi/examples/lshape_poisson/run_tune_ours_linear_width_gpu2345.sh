#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

run_one() {
  local gpu="$1"
  local cfg="$2"
  local name
  name="$(basename "$cfg" .py)"
  CUDA_VISIBLE_DEVICES="$gpu" XLA_PYTHON_CLIENT_PREALLOCATE=true \
    PYTHONPATH=/home/qjw/code/python_code/modifiedsinckan/pde/jaxpi \
    /home/qjw/anaconda3/envs/modified_net/bin/python ./main.py --workdir=. --config="./configs/${cfg}" \
    > "${name}_gpu${gpu}.out" 2> "${name}_gpu${gpu}.err"
}

run_queue() {
  local gpu="$1"
  shift
  for cfg in "$@"; do
    run_one "$gpu" "$cfg"
  done
}

run_queue 2 tune_ours_linear_w100_l4.py tune_ours_linear_w250_l4.py &
run_queue 3 tune_ours_linear_w120_l4.py tune_ours_linear_w300_l4.py &
run_queue 4 tune_ours_linear_w150_l4.py &
run_queue 5 tune_ours_linear_w200_l4.py &

wait

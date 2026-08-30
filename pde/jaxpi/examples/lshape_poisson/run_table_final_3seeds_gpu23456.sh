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

run_queue 2 \
  table_final_mlp_seed42.py \
  table_final_mlp_seed123.py \
  table_final_mlp_seed2026.py &

run_queue 3 \
  table_final_modifiedmlp_seed42.py \
  table_final_modifiedmlp_seed123.py \
  table_final_modifiedmlp_seed2026.py &

run_queue 4 \
  table_final_kan_seed42.py \
  table_final_kan_seed123.py \
  table_final_kan_seed2026.py &

run_queue 5 \
  table_final_sinckan_seed42.py \
  table_final_sinckan_seed123.py \
  table_final_sinckan_seed2026.py &

run_queue 6 \
  table_final_ours_seed42.py \
  table_final_ours_seed123.py \
  table_final_ours_seed2026.py &

wait

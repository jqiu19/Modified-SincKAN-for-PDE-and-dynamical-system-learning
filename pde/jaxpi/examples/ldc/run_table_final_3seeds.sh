#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON="/home/qjw/anaconda3/envs/modified_net/bin/python"
export PYTHONPATH="../..:${PYTHONPATH:-}"

run_one() {
  local gpu="$1"
  local config="$2"
  local name
  name="$(basename "${config}")"
  echo "[$(date '+%F %T')] GPU${gpu} starting ${name}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" ./main_modifiedsinckan.py \
    --workdir=. \
    --config="./configs/${config}.py" \
    > "${name}_gpu${gpu}.out" \
    2> "${name}_gpu${gpu}.err"
  echo "[$(date '+%F %T')] GPU${gpu} finished ${name}"
}

(
  run_one 2 table_final_mlp_seed42
  run_one 2 table_final_mlp_seed123
  run_one 2 table_final_mlp_seed2026
  run_one 2 table_final_ours_seed123
) &

(
  run_one 3 table_final_modifiedmlp_seed42
  run_one 3 table_final_modifiedmlp_seed123
  run_one 3 table_final_modifiedmlp_seed2026
  run_one 3 table_final_ours_seed2026
) &

(
  run_one 4 table_final_kan_seed42
  run_one 4 table_final_kan_seed123
  run_one 4 table_final_kan_seed2026
) &

(
  run_one 5 table_final_sinckan_seed42
  run_one 5 table_final_sinckan_seed123
  run_one 5 table_final_sinckan_seed2026
) &

wait
echo "[$(date '+%F %T')] table final 3-seed runs finished"

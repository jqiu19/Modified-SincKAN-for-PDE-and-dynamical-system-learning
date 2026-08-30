#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON="/home/qjw/anaconda3/envs/modified_net/bin/python"
export PYTHONPATH="../..:${PYTHONPATH:-}"

run_one() {
  local gpu="$1"
  local config="$2"
  local tag="$3"
  local name
  name="$(basename "${config}")"
  echo "[$(date '+%F %T')] GPU${gpu} starting ${name}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" ./main_modifiedsinckan.py \
    --workdir=. \
    --config="./configs/${config}.py" \
    > "${name}_${tag}_gpu${gpu}.out" \
    2> "${name}_${tag}_gpu${gpu}.err"
  echo "[$(date '+%F %T')] GPU${gpu} finished ${name}"
}

(
  run_one 2 table_ours_best_full_seed42 rerun
) &

(
  run_one 3 table_modifiedmlp_w256_l4_short rerun
) &

(
  run_one 4 table_kan_d16_w128_l4_short rerun
  run_one 4 table_chebykan_d8_w128_l4_short rerun
) &

(
  run_one 5 table_sinckan_d8h1_w256_l4_short rerun
  run_one 5 table_sinckan_d16h3_w256_l4_short rerun
) &

wait
echo "[$(date '+%F %T')] table stage1 remaining detached finished"

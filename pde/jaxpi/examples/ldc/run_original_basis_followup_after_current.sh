#!/usr/bin/env bash
set -euo pipefail

CURRENT_PID="${1:-2018100}"

cd "$(dirname "$0")"

PYTHON="/home/qjw/anaconda3/envs/modified_net/bin/python"
export PYTHONPATH="../..:${PYTHONPATH:-}"

echo "[$(date '+%F %T')] waiting for current basis queue PID ${CURRENT_PID}"
while kill -0 "${CURRENT_PID}" 2>/dev/null; do
  sleep 120
done
echo "[$(date '+%F %T')] current queue finished; starting follow-up basis sweep"

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
  run_one 2 modifiedsinckan_original_basis_v_linear_u12h2_lr5e4
  run_one 2 modifiedsinckan_original_basis_v_linear_u20h3_lr5e4
  run_one 2 modifiedsinckan_original_basis_v_linear_u24h4_lr5e4
) &

(
  run_one 3 modifiedsinckan_original_basis_v_linear_u16h3_lr3e4
  run_one 3 modifiedsinckan_original_basis_v_linear_u16h3_lr1e3
) &

(
  run_one 4 modifiedsinckan_original_basis_v_linear_alpha005
  run_one 4 modifiedsinckan_original_basis_v_linear_alpha02
  run_one 4 modifiedsinckan_original_basis_v_linear_glinear
) &

(
  run_one 5 modifiedsinckan_original_basis_v_cheby_d12
  run_one 5 modifiedsinckan_original_basis_v_sinc_u12h2_v24h4
) &

wait
echo "[$(date '+%F %T')] follow-up basis sweep finished"

#!/usr/bin/env bash
set -euo pipefail

WAIT_PID="${1:-2321739}"

cd "$(dirname "$0")"

PYTHON="/home/qjw/anaconda3/envs/modified_net/bin/python"
export PYTHONPATH="../..:${PYTHONPATH:-}"

echo "[$(date '+%F %T')] waiting for refine3 queue PID ${WAIT_PID}"
while kill -0 "${WAIT_PID}" 2>/dev/null; do
  sleep 120
done
echo "[$(date '+%F %T')] refine3 finished; starting capacity/g-basis ablation"

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
  run_one 2 modifiedsinckan_original_basis_v_sinc_u16h3_v28h5_w192_l4
  run_one 2 modifiedsinckan_original_basis_v_sinc_u16h3_v28h5_w320_l4
  run_one 2 modifiedsinckan_original_basis_v_sinc_u16h3_v28h5_w256_l3
  run_one 2 modifiedsinckan_original_basis_v_sinc_u16h3_v28h5_w256_l5
) &

(
  run_one 3 modifiedsinckan_original_basis_v_sinc_u16h3_v28h5_glinear
  run_one 3 modifiedsinckan_original_basis_v_sinc_u16h3_v28h5_g16h3
  run_one 3 modifiedsinckan_original_basis_v_sinc_u16h3_v28h5_g28h5
  run_one 3 modifiedsinckan_original_basis_v_sinc_u16h3_v28h5_gcheby_d16
) &

wait
echo "[$(date '+%F %T')] capacity/g-basis ablation finished"

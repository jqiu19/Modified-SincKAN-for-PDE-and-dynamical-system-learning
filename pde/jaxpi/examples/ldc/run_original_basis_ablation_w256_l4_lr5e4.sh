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
  run_one 2 modifiedsinckan_original_basis_v_sinc_u16h3_v24h4
  run_one 2 modifiedsinckan_original_basis_v_sinc_u16h3_v32h5
) &

(
  run_one 3 modifiedsinckan_original_basis_v_jacobi_d16_q075
  run_one 3 modifiedsinckan_original_basis_v_jacobi_d24_q05
) &

(
  run_one 4 modifiedsinckan_original_basis_v_spline_d16
  run_one 4 modifiedsinckan_original_basis_v_spline_d24
) &

(
  run_one 5 modifiedsinckan_original_basis_v_linear
  run_one 5 modifiedsinckan_original_basis_v_cheby_d16
  run_one 5 modifiedsinckan_original_basis_v_cheby_d24
) &

wait
echo "[$(date '+%F %T')] all basis ablations finished"

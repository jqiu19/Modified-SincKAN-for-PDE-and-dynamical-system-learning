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
  run_one 2 modifiedsinckan_original_basis_v_sinc_u16h3_v28h5_residual
  run_one 2 modifiedsinckan_original_basis_v_sinc_u16h3_v28h5_skip2
) &

(
  run_one 3 modifiedsinckan_original_basis_v_sinc_u16h3_v28h5_skip3
  run_one 3 modifiedsinckan_original_basis_v_sinc_u16h3_v28h5_swish
) &

(
  run_one 4 modifiedsinckan_original_basis_v_sinc_u16h3_v28h5_gelu
  run_one 4 modifiedsinckan_original_basis_v_sinc_u16h3_v28h5_weight_m05
) &

(
  run_one 5 modifiedsinckan_original_basis_v_sinc_u16h3_v28h5_weight_update500
  run_one 5 modifiedsinckan_original_basis_v_sinc_u16h3_v28h5_lamb
) &

wait
echo "[$(date '+%F %T')] training dynamics ablation finished"

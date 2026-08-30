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
  run_one 2 modifiedsinckan_original_basis_v_sinc_u16h3_v28h5_alpha0075
  run_one 2 modifiedsinckan_original_basis_v_sinc_u16h3_v28h5_g4h1
) &

(
  run_one 3 modifiedsinckan_original_basis_v_sinc_u16h3_v28h5_alpha009
  run_one 3 modifiedsinckan_original_basis_v_sinc_u16h3_v28h5_g6h1
) &

(
  run_one 4 modifiedsinckan_original_basis_v_sinc_u16h3_v28h5_alpha011
) &

(
  run_one 5 modifiedsinckan_original_basis_v_sinc_u16h3_v28h5_glinear_alpha009
) &

wait
echo "[$(date '+%F %T')] alpha/g refine finished"

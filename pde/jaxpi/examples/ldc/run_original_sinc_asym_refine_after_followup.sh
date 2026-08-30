#!/usr/bin/env bash
set -euo pipefail

FOLLOWUP_PID="${1:-2042116}"

cd "$(dirname "$0")"

PYTHON="/home/qjw/anaconda3/envs/modified_net/bin/python"
export PYTHONPATH="../..:${PYTHONPATH:-}"

echo "[$(date '+%F %T')] waiting for follow-up queue PID ${FOLLOWUP_PID}"
while kill -0 "${FOLLOWUP_PID}" 2>/dev/null; do
  sleep 120
done
echo "[$(date '+%F %T')] follow-up queue finished; starting sinc-asym refinement"

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
  run_one 2 modifiedsinckan_original_basis_v_sinc_u16h3_v28h5
  run_one 2 modifiedsinckan_original_basis_v_sinc_u16h3_v36h5
) &

(
  run_one 3 modifiedsinckan_original_basis_v_sinc_u16h3_v32h4
  run_one 3 modifiedsinckan_original_basis_v_sinc_u16h3_v32h6
) &

(
  run_one 4 modifiedsinckan_original_basis_v_sinc_u14h3_v32h5
  run_one 4 modifiedsinckan_original_basis_v_sinc_u18h3_v32h5
) &

(
  run_one 5 modifiedsinckan_original_basis_v_sinc_u16h2_v32h5
  run_one 5 modifiedsinckan_original_basis_v_sinc_u16h4_v32h5
) &

wait
echo "[$(date '+%F %T')] sinc-asym refinement finished"

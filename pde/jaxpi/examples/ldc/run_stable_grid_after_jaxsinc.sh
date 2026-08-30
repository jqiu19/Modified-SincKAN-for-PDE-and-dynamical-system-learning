#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON="/home/qjw/anaconda3/envs/modified_net/bin/python"
export PYTHONPATH="../..:${PYTHONPATH:-}"
JAXSINC_SAME_PID=1724328
JAXSINC_ASYM_PID=1724876

echo "[$(date '+%F %T')] Waiting for jnp.sinc PIDs ${JAXSINC_SAME_PID} and ${JAXSINC_ASYM_PID}..."
while kill -0 "${JAXSINC_SAME_PID}" 2>/dev/null || kill -0 "${JAXSINC_ASYM_PID}" 2>/dev/null; do
  sleep 60
done

echo "[$(date '+%F %T')] Starting stable sinc degree/len_h sweep."

CUDA_VISIBLE_DEVICES=2 setsid "${PYTHON}" ./main_modifiedsinckan.py \
  --workdir=. \
  --config=./configs/modifiedsinckan_uv_same_d16h3_stablesinc.py \
  > modifiedsinckan_uv_same_d16h3_stablesinc_gpu2.log \
  2> modifiedsinckan_uv_same_d16h3_stablesinc_gpu2.err &
PID_A=$!
echo "[$(date '+%F %T')] Started d16h3 stable sigmoid on GPU 2 with PID ${PID_A}"

CUDA_VISIBLE_DEVICES=3 setsid "${PYTHON}" ./main_modifiedsinckan.py \
  --workdir=. \
  --config=./configs/modifiedsinckan_uv_same_d24h4_stablesinc.py \
  > modifiedsinckan_uv_same_d24h4_stablesinc_gpu3.log \
  2> modifiedsinckan_uv_same_d24h4_stablesinc_gpu3.err &
PID_B=$!
echo "[$(date '+%F %T')] Started d24h4 stable sigmoid on GPU 3 with PID ${PID_B}"

wait "${PID_A}" "${PID_B}"

echo "[$(date '+%F %T')] Starting separate_residual gate checks."

CUDA_VISIBLE_DEVICES=2 setsid "${PYTHON}" ./main_modifiedsinckan.py \
  --workdir=. \
  --config=./configs/modifiedsinckan_uv_same_d32h5_residual_stablesinc.py \
  > modifiedsinckan_uv_same_d32h5_residual_stablesinc_gpu2.log \
  2> modifiedsinckan_uv_same_d32h5_residual_stablesinc_gpu2.err &
PID_C=$!
echo "[$(date '+%F %T')] Started d32h5 stable residual on GPU 2 with PID ${PID_C}"

CUDA_VISIBLE_DEVICES=3 setsid "${PYTHON}" ./main_modifiedsinckan.py \
  --workdir=. \
  --config=./configs/modifiedsinckan_uv_same_d16h3_residual_stablesinc.py \
  > modifiedsinckan_uv_same_d16h3_residual_stablesinc_gpu3.log \
  2> modifiedsinckan_uv_same_d16h3_residual_stablesinc_gpu3.err &
PID_D=$!
echo "[$(date '+%F %T')] Started d16h3 stable residual on GPU 3 with PID ${PID_D}"

wait "${PID_C}" "${PID_D}"

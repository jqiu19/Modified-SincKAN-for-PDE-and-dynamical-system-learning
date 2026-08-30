#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON="/home/qjw/anaconda3/envs/modified_net/bin/python"
export PYTHONPATH="../..:${PYTHONPATH:-}"
SAFE_SAME_PID=1703167
SAFE_ASYM_PID=1703721

echo "[$(date '+%F %T')] Waiting for safe2 PIDs ${SAFE_SAME_PID} and ${SAFE_ASYM_PID}..."
while kill -0 "${SAFE_SAME_PID}" 2>/dev/null || kill -0 "${SAFE_ASYM_PID}" 2>/dev/null; do
  sleep 60
done

echo "[$(date '+%F %T')] safe2 runs finished. Starting jnp.sinc 40% ablations."

CUDA_VISIBLE_DEVICES=2 setsid "${PYTHON}" ./main_modifiedsinckan.py \
  --workdir=. \
  --config=./configs/modifiedsinckan_uv_same_d32h5_jaxsinc.py \
  > modifiedsinckan_uv_same_d32h5_jaxsinc_gpu2.log \
  2> modifiedsinckan_uv_same_d32h5_jaxsinc_gpu2.err &
echo "[$(date '+%F %T')] Started same d32h5 jaxsinc on GPU 2 with PID $!"

CUDA_VISIBLE_DEVICES=3 setsid "${PYTHON}" ./main_modifiedsinckan.py \
  --workdir=. \
  --config=./configs/modifiedsinckan_uv_asym_u8h2_v64h8_jaxsinc.py \
  > modifiedsinckan_uv_asym_u8h2_v64h8_jaxsinc_gpu3.log \
  2> modifiedsinckan_uv_asym_u8h2_v64h8_jaxsinc_gpu3.err &
echo "[$(date '+%F %T')] Started asym u8h2/v64h8 jaxsinc on GPU 3 with PID $!"

wait

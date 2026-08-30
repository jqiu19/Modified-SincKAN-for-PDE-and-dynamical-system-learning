#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON="/home/qjw/anaconda3/envs/modified_net/bin/python"
export PYTHONPATH="../..:${PYTHONPATH:-}"

echo "[$(date '+%F %T')] Starting d16h2/d20h3 stable sinc separate_sigmoid pair."

CUDA_VISIBLE_DEVICES=2 setsid "${PYTHON}" ./main_modifiedsinckan.py \
  --workdir=. \
  --config=./configs/modifiedsinckan_uv_same_d16h2_stablesinc.py \
  > modifiedsinckan_uv_same_d16h2_stablesinc_gpu2.log \
  2> modifiedsinckan_uv_same_d16h2_stablesinc_gpu2.err &
PID_A=$!
echo "[$(date '+%F %T')] Started d16h2 stable sigmoid on GPU 2 with PID ${PID_A}"

CUDA_VISIBLE_DEVICES=3 setsid "${PYTHON}" ./main_modifiedsinckan.py \
  --workdir=. \
  --config=./configs/modifiedsinckan_uv_same_d20h3_stablesinc.py \
  > modifiedsinckan_uv_same_d20h3_stablesinc_gpu3.log \
  2> modifiedsinckan_uv_same_d20h3_stablesinc_gpu3.err &
PID_B=$!
echo "[$(date '+%F %T')] Started d20h3 stable sigmoid on GPU 3 with PID ${PID_B}"

wait "${PID_A}" "${PID_B}"

echo "[$(date '+%F %T')] Finished d16h2/d20h3 stable sinc separate_sigmoid pair."

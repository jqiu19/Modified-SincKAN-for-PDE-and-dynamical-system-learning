#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON="/home/qjw/anaconda3/envs/modified_net/bin/python"
export PYTHONPATH="../..:${PYTHONPATH:-}"

echo "[$(date '+%F %T')] Starting degree=16 len_h=5/6 pair on GPU 4/5."

CUDA_VISIBLE_DEVICES=4 setsid "${PYTHON}" ./main_modifiedsinckan.py \
  --workdir=. \
  --config=./configs/modifiedsinckan_uv_same_d16h5_stablesinc.py \
  > modifiedsinckan_uv_same_d16h5_stablesinc_gpu4.log \
  2> modifiedsinckan_uv_same_d16h5_stablesinc_gpu4.err &
PID_A=$!
echo "[$(date '+%F %T')] Started d16h5 stable sigmoid on GPU 4 with PID ${PID_A}"

CUDA_VISIBLE_DEVICES=5 setsid "${PYTHON}" ./main_modifiedsinckan.py \
  --workdir=. \
  --config=./configs/modifiedsinckan_uv_same_d16h6_stablesinc.py \
  > modifiedsinckan_uv_same_d16h6_stablesinc_gpu5.log \
  2> modifiedsinckan_uv_same_d16h6_stablesinc_gpu5.err &
PID_B=$!
echo "[$(date '+%F %T')] Started d16h6 stable sigmoid on GPU 5 with PID ${PID_B}"

wait "${PID_A}" "${PID_B}"

echo "[$(date '+%F %T')] Finished degree=16 len_h=5/6 pair on GPU 4/5."

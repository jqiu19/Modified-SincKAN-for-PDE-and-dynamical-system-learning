#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON="/home/qjw/anaconda3/envs/modified_net/bin/python"
export PYTHONPATH="../..:${PYTHONPATH:-}"

echo "[$(date '+%F %T')] Starting degree=16 len_h sweep."

CUDA_VISIBLE_DEVICES=2 setsid "${PYTHON}" ./main_modifiedsinckan.py \
  --workdir=. \
  --config=./configs/modifiedsinckan_uv_same_d16h1_stablesinc.py \
  > modifiedsinckan_uv_same_d16h1_stablesinc_gpu2.log \
  2> modifiedsinckan_uv_same_d16h1_stablesinc_gpu2.err &
PID_A=$!
echo "[$(date '+%F %T')] Started d16h1 stable sigmoid on GPU 2 with PID ${PID_A}"

CUDA_VISIBLE_DEVICES=3 setsid "${PYTHON}" ./main_modifiedsinckan.py \
  --workdir=. \
  --config=./configs/modifiedsinckan_uv_same_d16h4_stablesinc.py \
  > modifiedsinckan_uv_same_d16h4_stablesinc_gpu3.log \
  2> modifiedsinckan_uv_same_d16h4_stablesinc_gpu3.err &
PID_B=$!
echo "[$(date '+%F %T')] Started d16h4 stable sigmoid on GPU 3 with PID ${PID_B}"

wait "${PID_A}" "${PID_B}"

echo "[$(date '+%F %T')] Starting upper len_h pair for degree=16."

CUDA_VISIBLE_DEVICES=2 setsid "${PYTHON}" ./main_modifiedsinckan.py \
  --workdir=. \
  --config=./configs/modifiedsinckan_uv_same_d16h5_stablesinc.py \
  > modifiedsinckan_uv_same_d16h5_stablesinc_gpu2.log \
  2> modifiedsinckan_uv_same_d16h5_stablesinc_gpu2.err &
PID_C=$!
echo "[$(date '+%F %T')] Started d16h5 stable sigmoid on GPU 2 with PID ${PID_C}"

CUDA_VISIBLE_DEVICES=3 setsid "${PYTHON}" ./main_modifiedsinckan.py \
  --workdir=. \
  --config=./configs/modifiedsinckan_uv_same_d16h6_stablesinc.py \
  > modifiedsinckan_uv_same_d16h6_stablesinc_gpu3.log \
  2> modifiedsinckan_uv_same_d16h6_stablesinc_gpu3.err &
PID_D=$!
echo "[$(date '+%F %T')] Started d16h6 stable sigmoid on GPU 3 with PID ${PID_D}"

wait "${PID_C}" "${PID_D}"

echo "[$(date '+%F %T')] Finished degree=16 len_h sweep."

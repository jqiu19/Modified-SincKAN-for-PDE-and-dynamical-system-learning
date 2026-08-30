#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON="/home/qjw/anaconda3/envs/modified_net/bin/python"
export PYTHONPATH="../..:${PYTHONPATH:-}"

echo "[$(date '+%F %T')] Starting h3 fine degree sweep around d16."

CUDA_VISIBLE_DEVICES=2 setsid "${PYTHON}" ./main_modifiedsinckan.py \
  --workdir=. \
  --config=./configs/modifiedsinckan_uv_same_d14h3_stablesinc.py \
  > modifiedsinckan_uv_same_d14h3_stablesinc_gpu2.log \
  2> modifiedsinckan_uv_same_d14h3_stablesinc_gpu2.err &
PID_A=$!
echo "[$(date '+%F %T')] Started d14h3 stable sigmoid on GPU 2 with PID ${PID_A}"

CUDA_VISIBLE_DEVICES=3 setsid "${PYTHON}" ./main_modifiedsinckan.py \
  --workdir=. \
  --config=./configs/modifiedsinckan_uv_same_d15h3_stablesinc.py \
  > modifiedsinckan_uv_same_d15h3_stablesinc_gpu3.log \
  2> modifiedsinckan_uv_same_d15h3_stablesinc_gpu3.err &
PID_B=$!
echo "[$(date '+%F %T')] Started d15h3 stable sigmoid on GPU 3 with PID ${PID_B}"

wait "${PID_A}" "${PID_B}"

echo "[$(date '+%F %T')] Starting upper-side h3 fine degree pair."

CUDA_VISIBLE_DEVICES=2 setsid "${PYTHON}" ./main_modifiedsinckan.py \
  --workdir=. \
  --config=./configs/modifiedsinckan_uv_same_d17h3_stablesinc.py \
  > modifiedsinckan_uv_same_d17h3_stablesinc_gpu2.log \
  2> modifiedsinckan_uv_same_d17h3_stablesinc_gpu2.err &
PID_C=$!
echo "[$(date '+%F %T')] Started d17h3 stable sigmoid on GPU 2 with PID ${PID_C}"

CUDA_VISIBLE_DEVICES=3 setsid "${PYTHON}" ./main_modifiedsinckan.py \
  --workdir=. \
  --config=./configs/modifiedsinckan_uv_same_d18h3_stablesinc.py \
  > modifiedsinckan_uv_same_d18h3_stablesinc_gpu3.log \
  2> modifiedsinckan_uv_same_d18h3_stablesinc_gpu3.err &
PID_D=$!
echo "[$(date '+%F %T')] Started d18h3 stable sigmoid on GPU 3 with PID ${PID_D}"

wait "${PID_C}" "${PID_D}"

echo "[$(date '+%F %T')] Finished h3 fine degree sweep around d16."

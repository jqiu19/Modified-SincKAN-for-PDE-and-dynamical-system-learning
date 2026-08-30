#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON="/home/qjw/anaconda3/envs/modified_net/bin/python"
export PYTHONPATH="../..:${PYTHONPATH:-}"

echo "[$(date '+%F %T')] Starting original d16h3 small-width lr5e-4 sweep."

CUDA_VISIBLE_DEVICES=2 setsid "${PYTHON}" ./main_modifiedsinckan.py \
  --workdir=. \
  --config=./configs/modifiedsinckan_original_d16h3_w64_l4_lr5e4.py \
  > modifiedsinckan_original_d16h3_w64_l4_lr5e4_gpu2.log \
  2> modifiedsinckan_original_d16h3_w64_l4_lr5e4_gpu2.err &
PID_A=$!
echo "[$(date '+%F %T')] Started w64_l4_lr5e4 on GPU 2 with PID ${PID_A}"

CUDA_VISIBLE_DEVICES=3 setsid "${PYTHON}" ./main_modifiedsinckan.py \
  --workdir=. \
  --config=./configs/modifiedsinckan_original_d16h3_w128_l4_lr5e4.py \
  > modifiedsinckan_original_d16h3_w128_l4_lr5e4_gpu3.log \
  2> modifiedsinckan_original_d16h3_w128_l4_lr5e4_gpu3.err &
PID_B=$!
echo "[$(date '+%F %T')] Started w128_l4_lr5e4 on GPU 3 with PID ${PID_B}"

CUDA_VISIBLE_DEVICES=4 setsid "${PYTHON}" ./main_modifiedsinckan.py \
  --workdir=. \
  --config=./configs/modifiedsinckan_original_d16h3_w32_l4_lr5e4.py \
  > modifiedsinckan_original_d16h3_w32_l4_lr5e4_gpu4.log \
  2> modifiedsinckan_original_d16h3_w32_l4_lr5e4_gpu4.err &
PID_C=$!
echo "[$(date '+%F %T')] Started w32_l4_lr5e4 on GPU 4 with PID ${PID_C}"

wait "${PID_A}" "${PID_B}" "${PID_C}"

echo "[$(date '+%F %T')] Finished original d16h3 small-width lr5e-4 sweep."

#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON="/home/qjw/anaconda3/envs/modified_net/bin/python"
export PYTHONPATH="../..:${PYTHONPATH:-}"

echo "[$(date '+%F %T')] Starting original d16h3 capacity/lr ablation on GPU 2-5."

CUDA_VISIBLE_DEVICES=2 setsid "${PYTHON}" ./main_modifiedsinckan.py \
  --workdir=. \
  --config=./configs/modifiedsinckan_d16h3_w512_l4_lr1e3.py \
  > modifiedsinckan_original_d16h3_w512_l4_lr1e3_gpu2.log \
  2> modifiedsinckan_original_d16h3_w512_l4_lr1e3_gpu2.err &
PID_A=$!
echo "[$(date '+%F %T')] Started w512_l4_lr1e3 on GPU 2 with PID ${PID_A}"

CUDA_VISIBLE_DEVICES=3 setsid "${PYTHON}" ./main_modifiedsinckan.py \
  --workdir=. \
  --config=./configs/modifiedsinckan_d16h3_w256_l6_lr1e3.py \
  > modifiedsinckan_original_d16h3_w256_l6_lr1e3_gpu3.log \
  2> modifiedsinckan_original_d16h3_w256_l6_lr1e3_gpu3.err &
PID_B=$!
echo "[$(date '+%F %T')] Started w256_l6_lr1e3 on GPU 3 with PID ${PID_B}"

CUDA_VISIBLE_DEVICES=4 setsid "${PYTHON}" ./main_modifiedsinckan.py \
  --workdir=. \
  --config=./configs/modifiedsinckan_d16h3_w512_l4_lr5e4.py \
  > modifiedsinckan_original_d16h3_w512_l4_lr5e4_gpu4.log \
  2> modifiedsinckan_original_d16h3_w512_l4_lr5e4_gpu4.err &
PID_C=$!
echo "[$(date '+%F %T')] Started w512_l4_lr5e4 on GPU 4 with PID ${PID_C}"

CUDA_VISIBLE_DEVICES=5 setsid "${PYTHON}" ./main_modifiedsinckan.py \
  --workdir=. \
  --config=./configs/modifiedsinckan_original_d16h3_w256_l4_lr5e4.py \
  > modifiedsinckan_original_d16h3_w256_l4_lr5e4_gpu5.log \
  2> modifiedsinckan_original_d16h3_w256_l4_lr5e4_gpu5.err &
PID_D=$!
echo "[$(date '+%F %T')] Started w256_l4_lr5e4 on GPU 5 with PID ${PID_D}"

wait "${PID_A}" "${PID_B}" "${PID_C}" "${PID_D}"

echo "[$(date '+%F %T')] Finished original d16h3 capacity/lr ablation."

#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON="/home/qjw/anaconda3/envs/modified_net/bin/python"
export PYTHONPATH="../..:${PYTHONPATH:-}"
PIRATE_PID=1160474
MODIFIED_PID=1160475

echo "[$(date '+%F %T')] Waiting for current PirateNet PID ${PIRATE_PID} and ModifiedSincKAN PID ${MODIFIED_PID}..."
while kill -0 "${PIRATE_PID}" 2>/dev/null || kill -0 "${MODIFIED_PID}" 2>/dev/null; do
  sleep 60
done

echo "[$(date '+%F %T')] Current runs finished. Starting adaptive ModifiedSincKAN ablations."

CUDA_VISIBLE_DEVICES=2 setsid "${PYTHON}" ./main_modifiedsinckan.py \
  --workdir=. \
  --config=./configs/modifiedsinckan_adaptive_degree.py \
  > modifiedsinckan_adaptive_degree_gpu2.log \
  2> modifiedsinckan_adaptive_degree_gpu2.err &
echo "[$(date '+%F %T')] Started adaptive_degree on GPU 2 with PID $!"

CUDA_VISIBLE_DEVICES=3 setsid "${PYTHON}" ./main_modifiedsinckan.py \
  --workdir=. \
  --config=./configs/modifiedsinckan_adaptive_both.py \
  > modifiedsinckan_adaptive_both_gpu3.log \
  2> modifiedsinckan_adaptive_both_gpu3.err &
echo "[$(date '+%F %T')] Started adaptive_both on GPU 3 with PID $!"

wait

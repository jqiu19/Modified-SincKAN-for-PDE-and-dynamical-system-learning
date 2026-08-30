#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON="/home/qjw/anaconda3/envs/modified_net/bin/python"
export PYTHONPATH=../..

gpu_has_compute_process() {
  local gpu="$1"
  nvidia-smi pmon -c 1 2>/dev/null | awk -v gpu="$gpu" '
    $1 == gpu && $2 ~ /^[0-9]+$/ && $2 != "-" { found = 1 }
    END { exit found ? 0 : 1 }
  '
}

wait_for_gpus() {
  while true; do
    if ! gpu_has_compute_process 5 && ! gpu_has_compute_process 6 && ! gpu_has_compute_process 7; then
      return 0
    fi
    date "+%F %T waiting for GPUs 5,6,7 to become free"
    sleep 60
  done
}

wait_for_gpus

CUDA_VISIBLE_DEVICES=7 nohup "$PYTHON" ./main.py \
  --workdir=. --config=./configs/pirate.py \
  > pirate_b4096_repro_run.log 2> pirate_b4096_repro_run.err &
echo $! > pirate_b4096_repro.pid

CUDA_VISIBLE_DEVICES=6 nohup "$PYTHON" ./main_sinc.py \
  --workdir=. --config=./configs/sinc.py \
  > sinc_b4096_repro_run.log 2> sinc_b4096_repro_run.err &
echo $! > sinc_b4096_repro.pid

CUDA_VISIBLE_DEVICES=5 nohup "$PYTHON" ./main_modifiedsinckan.py \
  --workdir=. --config=./configs/modifiedsinckan.py \
  > modifiedsinckan_b4096_repro_run.log 2> modifiedsinckan_b4096_repro_run.err &
echo $! > modifiedsinckan_b4096_repro.pid

wait

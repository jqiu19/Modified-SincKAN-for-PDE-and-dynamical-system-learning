#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

COMMON=(--n 50 --coupling 4.0 --omega_std 1.0 --epochs 30000 --log_every 100 --results_dir results_short)

python train_kuramoto.py --model mlp --run_name mlp_w64_d2_lr5e4 --width 64 --depth 2 --lr_max 5e-4 "${COMMON[@]}" 2>&1 | tee mlp_w64_d2_lr5e4.log
python train_kuramoto.py --model mlp --run_name mlp_w128_d2_lr5e4 --width 128 --depth 2 --lr_max 5e-4 "${COMMON[@]}" 2>&1 | tee mlp_w128_d2_lr5e4.log

python train_kuramoto.py --model modifiedmlp --run_name modifiedmlp_w64_d2_lr5e4 --width 64 --depth 2 --lr_max 5e-4 "${COMMON[@]}" 2>&1 | tee modifiedmlp_w64_d2_lr5e4.log
python train_kuramoto.py --model modifiedmlp --run_name modifiedmlp_w128_d2_lr5e4 --width 128 --depth 2 --lr_max 5e-4 "${COMMON[@]}" 2>&1 | tee modifiedmlp_w128_d2_lr5e4.log

python train_kuramoto.py --model kan --run_name kan_w64_d2_g16 --width 64 --depth 2 --degree 16 "${COMMON[@]}" 2>&1 | tee kan_w64_d2_g16.log
python train_kuramoto.py --model kan --run_name kan_w64_d2_g24 --width 64 --depth 2 --degree 24 "${COMMON[@]}" 2>&1 | tee kan_w64_d2_g24.log

python train_kuramoto.py --model sinckan --run_name sinckan_w64_d2_d16h1 --width 64 --depth 2 --degree 16 --len_h 1 "${COMMON[@]}" 2>&1 | tee sinckan_w64_d2_d16h1.log
python train_kuramoto.py --model sinckan --run_name sinckan_w64_d2_d24h1 --width 64 --depth 2 --degree 24 --len_h 1 "${COMMON[@]}" 2>&1 | tee sinckan_w64_d2_d24h1.log

python train_kuramoto.py --model modifiedsinckan --run_name modifiedsinckan_w64_d2_u16h1v32h1 --width 64 --depth 2 --u_degree 16 --u_len_h 1 --v_degree 32 --v_len_h 1 --alpha 0.09 "${COMMON[@]}" 2>&1 | tee modifiedsinckan_w64_d2_u16h1v32h1.log
python train_kuramoto.py --model modifiedsinckan --run_name modifiedsinckan_w64_d2_u16h1v48h1 --width 64 --depth 2 --u_degree 16 --u_len_h 1 --v_degree 48 --v_len_h 1 --alpha 0.09 "${COMMON[@]}" 2>&1 | tee modifiedsinckan_w64_d2_u16h1v48h1.log

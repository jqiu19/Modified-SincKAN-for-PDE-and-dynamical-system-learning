#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Edit the selected configs after short sweep. Keep run names prefixed by model
# so summarize_kuramoto.py can group them automatically.
SEEDS=(0 123 2026)
COMMON=(--n 50 --coupling 4.0 --omega_std 1.0 --epochs 100000 --log_every 100 --results_dir results_table)

for seed in "${SEEDS[@]}"; do
  python train_kuramoto.py --model mlp --run_name mlp_best_seed${seed} --seed "$seed" --width 64 --depth 2 --lr_max 5e-4 "${COMMON[@]}" 2>&1 | tee mlp_best_seed${seed}.log
done

for seed in "${SEEDS[@]}"; do
  python train_kuramoto.py --model modifiedmlp --run_name modifiedmlp_best_seed${seed} --seed "$seed" --width 64 --depth 2 --lr_max 5e-4 "${COMMON[@]}" 2>&1 | tee modifiedmlp_best_seed${seed}.log
done

for seed in "${SEEDS[@]}"; do
  python train_kuramoto.py --model kan --run_name kan_best_seed${seed} --seed "$seed" --width 64 --depth 2 --degree 16 --lr_max 5e-4 "${COMMON[@]}" 2>&1 | tee kan_best_seed${seed}.log
done

for seed in "${SEEDS[@]}"; do
  python train_kuramoto.py --model sinckan --run_name sinckan_best_seed${seed} --seed "$seed" --width 64 --depth 2 --degree 16 --len_h 1 --lr_max 5e-4 "${COMMON[@]}" 2>&1 | tee sinckan_best_seed${seed}.log
done

for seed in "${SEEDS[@]}"; do
  python train_kuramoto.py --model modifiedsinckan --run_name modifiedsinckan_best_seed${seed} --seed "$seed" --width 64 --depth 2 --u_degree 16 --u_len_h 1 --v_degree 32 --v_len_h 1 --alpha 0.09 --lr_max 5e-4 "${COMMON[@]}" 2>&1 | tee modifiedsinckan_best_seed${seed}.log
done

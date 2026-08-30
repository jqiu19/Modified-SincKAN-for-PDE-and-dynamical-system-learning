#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/c/Users/Qiu Jingwei/Documents/New project/SincKAN-main/SincKAN-main/approximation"
PY="/root/modifiedkan-gpu/bin/python"

cd "$ROOT"

run() {
  echo "[$(date '+%F %T')] $*"
  "$@"
}

# Shared protocol:
# ntrain=2000, ntest=4000, epochs=4000, ite=20, interval=0,1

# modifiedsinckan remaining seed
run "$PY" interpolation_hd.py \
  --epochs 4000 --ite 20 --lr 1e-3 \
  --network modifiedsinckan --datatype fractal --dim 2 \
  --features 96 --layers 6 --degree 96 --len_h 4 --init_h 2 \
  --decay inverse --activation tanh --uv_activation none \
  --gate_mode separate_sigmoid --skip 1 \
  --interval=0.0,1.0 --normalization 0 \
  --ntrain 2000 --ntest 4000 --seed 2 --device 0

# modifiedmlp remaining seeds
for seed in 1 2; do
  run "$PY" interpolation_hd.py \
    --epochs 4000 --ite 20 --lr 1e-3 \
    --network modifiedmlp --datatype fractal --dim 2 \
    --features 100 --layers 10 \
    --interval=0.0,1.0 --normalization 0 \
    --ntrain 2000 --ntest 4000 --seed "$seed" --device 0
done

# mlp remaining seeds
for seed in 1 2; do
  run "$PY" interpolation_hd.py \
    --epochs 4000 --ite 20 --lr 1e-3 \
    --network mlp --datatype fractal --dim 2 \
    --features 100 --layers 10 \
    --interval=0.0,1.0 --normalization 0 \
    --ntrain 2000 --ntest 4000 --seed "$seed" --device 0
done

# sinckan all seeds
for seed in 0 1 2; do
  run "$PY" interpolation_hd.py \
    --epochs 4000 --ite 20 --lr 1e-3 \
    --network sinckan --datatype fractal --dim 2 \
    --kanshape 32 --degree 100 --len_h 6 --noise 0 \
    --activation none --interval=0.0,1.0 --skip 0 --normalization 0 \
    --ntrain 2000 --ntest 4000 --seed "$seed" --device 0
done

# kan all seeds
for seed in 0 1 2; do
  run "$PY" interpolation_hd.py \
    --epochs 4000 --ite 20 --lr 1e-3 \
    --network kan --datatype fractal --dim 2 \
    --kanshape 32,32 --degree 32 \
    --interval=0.0,1.0 --normalization 0 \
    --ntrain 2000 --ntest 4000 --seed "$seed" --device 0
done

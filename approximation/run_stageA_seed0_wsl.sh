#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/c/Users/Qiu Jingwei/Documents/New project/SincKAN-main/SincKAN-main"
PY="/root/modifiedkan-gpu/bin/python"

cd "$ROOT/approximation"

run() {
  echo "[$(date '+%F %T')] $*"
  "$@"
}

# 1D: spectral_bias
run "$PY" approximation_1d.py \
  --epochs 5000 --ite 20 --lr 1e-3 \
  --network modifiedsinckan --datatype spectral_bias \
  --features 64 --layers 5 --degree 96 --len_h 8 --init_h 2 \
  --decay inverse --normalization 1 --activation tanh --uv_activation none \
  --gate_mode separate_residual --skip True \
  --npoints 5000 --ntrain 3000 --ntest 10000 --interval=-1.0,1.0 \
  --seed 0 --device 0

run "$PY" approximation_1d.py \
  --epochs 5000 --ite 20 --lr 1e-3 \
  --network mlp --datatype spectral_bias \
  --features 100 --layers 10 \
  --npoints 5000 --ntrain 3000 --ntest 10000 --interval=-1.0,1.0 \
  --seed 0 --device 0

run "$PY" approximation_1d.py \
  --epochs 5000 --ite 20 --lr 1e-3 \
  --network modifiedmlp --datatype spectral_bias \
  --features 100 --layers 10 \
  --npoints 5000 --ntrain 3000 --ntest 10000 --interval=-1.0,1.0 \
  --seed 0 --device 0

run "$PY" approximation_1d.py \
  --epochs 5000 --ite 20 --lr 1e-3 \
  --network kan --datatype spectral_bias \
  --kanshape 56,56,56 --degree 5 \
  --npoints 5000 --ntrain 3000 --ntest 10000 --interval=-1.0,1.0 \
  --seed 0 --device 0

run "$PY" approximation_1d.py \
  --epochs 5000 --ite 20 --lr 1e-3 \
  --network sinckan --datatype spectral_bias \
  --kanshape 168 --degree 5 --normalization 1 --len_h 9 --init_h 2 \
  --activation none \
  --npoints 5000 --ntrain 3000 --ntest 10000 --interval=-1.0,1.0 \
  --seed 0 --device 0

# 2D: poisson_2d_solution
run "$PY" interpolation_hd.py \
  --epochs 3000 --ite 20 --lr 1e-3 \
  --network modifiedsinckan --datatype poisson_2d_solution --dim 2 \
  --features 64 --layers 5 --degree 32 --len_h 3 --init_h 2 \
  --decay inverse --normalization 0 --activation tanh --uv_activation none \
  --gate_mode separate_residual --skip 1 \
  --ntrain 2000 --ntest 4000 --interval=0.0,1.0 \
  --seed 0 --device 0

run "$PY" interpolation_hd.py \
  --epochs 3000 --ite 20 --lr 1e-3 \
  --network mlp --datatype poisson_2d_solution --dim 2 \
  --features 100 --layers 10 \
  --ntrain 2000 --ntest 4000 --interval=0.0,1.0 \
  --seed 0 --device 0

run "$PY" interpolation_hd.py \
  --epochs 3000 --ite 20 --lr 1e-3 \
  --network modifiedmlp --datatype poisson_2d_solution --dim 2 \
  --features 100 --layers 10 \
  --ntrain 2000 --ntest 4000 --interval=0.0,1.0 \
  --seed 0 --device 0

run "$PY" interpolation_hd.py \
  --epochs 3000 --ite 20 --lr 1e-3 \
  --network kan --datatype poisson_2d_solution --dim 2 \
  --kanshape 32,32 --degree 16 \
  --ntrain 2000 --ntest 4000 --interval=0.0,1.0 \
  --seed 0 --device 0

run "$PY" interpolation_hd.py \
  --epochs 3000 --ite 20 --lr 1e-3 \
  --network sinckan --datatype poisson_2d_solution --dim 2 \
  --kanshape 32 --degree 32 --len_h 4 --init_h 2 \
  --normalization 0 --activation tanh --skip 0 \
  --ntrain 2000 --ntest 4000 --interval=0.0,1.0 \
  --seed 0 --device 0

# 4D: pde_param_4d
run "$PY" interpolation_hd.py \
  --epochs 3000 --ite 20 --lr 1e-3 \
  --network modifiedsinckan --datatype pde_param_4d --dim 4 \
  --features 64 --layers 5 --degree 24 --len_h 3 --init_h 2 \
  --decay inverse --normalization 0 --activation tanh --uv_activation none \
  --gate_mode separate_residual --skip 1 \
  --ntrain 3000 --ntest 4000 --interval=0.0,1.0 \
  --seed 0 --device 0

run "$PY" interpolation_hd.py \
  --epochs 3000 --ite 20 --lr 1e-3 \
  --network mlp --datatype pde_param_4d --dim 4 \
  --features 100 --layers 10 \
  --ntrain 3000 --ntest 4000 --interval=0.0,1.0 \
  --seed 0 --device 0

run "$PY" interpolation_hd.py \
  --epochs 3000 --ite 20 --lr 1e-3 \
  --network modifiedmlp --datatype pde_param_4d --dim 4 \
  --features 100 --layers 10 \
  --ntrain 3000 --ntest 4000 --interval=0.0,1.0 \
  --seed 0 --device 0

run "$PY" interpolation_hd.py \
  --epochs 3000 --ite 20 --lr 1e-3 \
  --network kan --datatype pde_param_4d --dim 4 \
  --kanshape 24,24 --degree 12 \
  --ntrain 3000 --ntest 4000 --interval=0.0,1.0 \
  --seed 0 --device 0

run "$PY" interpolation_hd.py \
  --epochs 3000 --ite 20 --lr 1e-3 \
  --network sinckan --datatype pde_param_4d --dim 4 \
  --kanshape 24 --degree 24 --len_h 3 --init_h 2 \
  --normalization 0 --activation tanh --skip 0 \
  --ntrain 3000 --ntest 4000 --interval=0.0,1.0 \
  --seed 0 --device 0

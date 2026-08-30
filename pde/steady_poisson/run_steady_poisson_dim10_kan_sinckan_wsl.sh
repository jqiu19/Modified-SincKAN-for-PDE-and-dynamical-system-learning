#!/usr/bin/env bash
set -euo pipefail

cd "/mnt/c/Users/Qiu Jingwei/Documents/New project/SincKAN-main/SincKAN-main/pde"
export PATH="/root/modifiedkan-gpu/lib/python3.10/site-packages/nvidia/cuda_nvcc/bin:${PATH}"

/root/modifiedkan-gpu/bin/python steady_poisson.py --mode train --network kan --datatype poisson --dim 10 --alpha 1.0 --kanshape 16,16,16 --degree 16 --epochs 2000 --ite 20 --n_interior 2000 --n_boundary 100 --seed 0 --device 0
/root/modifiedkan-gpu/bin/python steady_poisson.py --mode train --network kan --datatype poisson --dim 10 --alpha 1.0 --kanshape 16,16,16 --degree 16 --epochs 2000 --ite 20 --n_interior 2000 --n_boundary 100 --seed 1 --device 0
/root/modifiedkan-gpu/bin/python steady_poisson.py --mode train --network kan --datatype poisson --dim 10 --alpha 1.0 --kanshape 16,16,16 --degree 16 --epochs 2000 --ite 20 --n_interior 2000 --n_boundary 100 --seed 2 --device 0

/root/modifiedkan-gpu/bin/python steady_poisson.py --mode train --network sinckan --datatype poisson --dim 10 --alpha 1.0 --kanshape 16 --degree 16 --len_h 2 --init_h 2 --skip 1 --epochs 2000 --ite 20 --n_interior 2000 --n_boundary 100 --seed 0 --device 0
/root/modifiedkan-gpu/bin/python steady_poisson.py --mode train --network sinckan --datatype poisson --dim 10 --alpha 1.0 --kanshape 16 --degree 16 --len_h 2 --init_h 2 --skip 1 --epochs 2000 --ite 20 --n_interior 2000 --n_boundary 100 --seed 1 --device 0
/root/modifiedkan-gpu/bin/python steady_poisson.py --mode train --network sinckan --datatype poisson --dim 10 --alpha 1.0 --kanshape 16 --degree 16 --len_h 2 --init_h 2 --skip 1 --epochs 2000 --ite 20 --n_interior 2000 --n_boundary 100 --seed 2 --device 0

#!/usr/bin/env bash
set -euo pipefail

cd "/mnt/c/Users/Qiu Jingwei/Documents/New project/SincKAN-main/SincKAN-main/pde"
export PATH="/root/modifiedkan-gpu/lib/python3.10/site-packages/nvidia/cuda_nvcc/bin:${PATH}"

# dim = 2, alpha = 0.2
/root/modifiedkan-gpu/bin/python steady_poisson.py --mode train --network mlp --datatype poisson --dim 2 --alpha 0.2 --features 128 --layers 6 --epochs 2000 --ite 20 --n_interior 2000 --n_boundary 100 --seed 0 --device 0
/root/modifiedkan-gpu/bin/python steady_poisson.py --mode train --network mlp --datatype poisson --dim 2 --alpha 0.2 --features 128 --layers 6 --epochs 2000 --ite 20 --n_interior 2000 --n_boundary 100 --seed 1 --device 0
/root/modifiedkan-gpu/bin/python steady_poisson.py --mode train --network mlp --datatype poisson --dim 2 --alpha 0.2 --features 128 --layers 6 --epochs 2000 --ite 20 --n_interior 2000 --n_boundary 100 --seed 2 --device 0

/root/modifiedkan-gpu/bin/python steady_poisson.py --mode train --network modifiedmlp --datatype poisson --dim 2 --alpha 0.2 --features 128 --layers 6 --epochs 2000 --ite 20 --n_interior 2000 --n_boundary 100 --seed 0 --device 0
/root/modifiedkan-gpu/bin/python steady_poisson.py --mode train --network modifiedmlp --datatype poisson --dim 2 --alpha 0.2 --features 128 --layers 6 --epochs 2000 --ite 20 --n_interior 2000 --n_boundary 100 --seed 1 --device 0
/root/modifiedkan-gpu/bin/python steady_poisson.py --mode train --network modifiedmlp --datatype poisson --dim 2 --alpha 0.2 --features 128 --layers 6 --epochs 2000 --ite 20 --n_interior 2000 --n_boundary 100 --seed 2 --device 0

# dim = 10, alpha = 1.0
/root/modifiedkan-gpu/bin/python steady_poisson.py --mode train --network mlp --datatype poisson --dim 10 --alpha 1.0 --features 128 --layers 6 --epochs 2000 --ite 20 --n_interior 2000 --n_boundary 100 --seed 0 --device 0
/root/modifiedkan-gpu/bin/python steady_poisson.py --mode train --network mlp --datatype poisson --dim 10 --alpha 1.0 --features 128 --layers 6 --epochs 2000 --ite 20 --n_interior 2000 --n_boundary 100 --seed 1 --device 0
/root/modifiedkan-gpu/bin/python steady_poisson.py --mode train --network mlp --datatype poisson --dim 10 --alpha 1.0 --features 128 --layers 6 --epochs 2000 --ite 20 --n_interior 2000 --n_boundary 100 --seed 2 --device 0

/root/modifiedkan-gpu/bin/python steady_poisson.py --mode train --network modifiedmlp --datatype poisson --dim 10 --alpha 1.0 --features 128 --layers 6 --epochs 2000 --ite 20 --n_interior 2000 --n_boundary 100 --seed 0 --device 0
/root/modifiedkan-gpu/bin/python steady_poisson.py --mode train --network modifiedmlp --datatype poisson --dim 10 --alpha 1.0 --features 128 --layers 6 --epochs 2000 --ite 20 --n_interior 2000 --n_boundary 100 --seed 1 --device 0
/root/modifiedkan-gpu/bin/python steady_poisson.py --mode train --network modifiedmlp --datatype poisson --dim 10 --alpha 1.0 --features 128 --layers 6 --epochs 2000 --ite 20 --n_interior 2000 --n_boundary 100 --seed 2 --device 0

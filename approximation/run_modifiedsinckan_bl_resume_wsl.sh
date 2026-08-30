#!/usr/bin/env bash
set -euo pipefail

cd "/mnt/c/Users/Qiu Jingwei/Documents/New project/SincKAN-main/SincKAN-main/approximation"
export PATH="/root/modifiedkan-gpu/lib/python3.10/site-packages/nvidia/cuda_nvcc/bin:${PATH}"

/root/modifiedkan-gpu/bin/python approximation_1d.py --epochs 5000 --network modifiedsinckan --datatype bl --features 64 --layers 5 --degree 64 --len_h 4 --init_h 2 --decay inverse --normalization 1 --activation tanh --uv_activation none --gate_mode sigmoid --skip True --npoints 5000 --ntrain 3000 --ntest 10000 --seed 0 --device 0
/root/modifiedkan-gpu/bin/python approximation_1d.py --epochs 5000 --network modifiedsinckan --datatype bl --features 64 --layers 5 --degree 64 --len_h 4 --init_h 2 --decay inverse --normalization 1 --activation tanh --uv_activation none --gate_mode sigmoid --skip True --npoints 5000 --ntrain 3000 --ntest 10000 --seed 1 --device 0
/root/modifiedkan-gpu/bin/python approximation_1d.py --epochs 5000 --network modifiedsinckan --datatype bl --features 64 --layers 5 --degree 64 --len_h 4 --init_h 2 --decay inverse --normalization 1 --activation tanh --uv_activation none --gate_mode sigmoid --skip True --npoints 5000 --ntrain 3000 --ntest 10000 --seed 2 --device 0

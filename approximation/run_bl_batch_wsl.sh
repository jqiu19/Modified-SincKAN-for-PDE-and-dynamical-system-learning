#!/usr/bin/env bash
set -euo pipefail

cd "/mnt/c/Users/Qiu Jingwei/Documents/New project/SincKAN-main/SincKAN-main/approximation"
export PATH="/root/modifiedkan-gpu/lib/python3.10/site-packages/nvidia/cuda_nvcc/bin:${PATH}"
/root/modifiedkan-gpu/bin/python run_table1_benchmarks.py --suite bl --python_cmd /root/modifiedkan-gpu/bin/python

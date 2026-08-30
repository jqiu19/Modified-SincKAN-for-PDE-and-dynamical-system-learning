#!/usr/bin/env bash
set -euo pipefail

cd "/mnt/c/Users/Qiu Jingwei/Documents/New project/jaxpi/examples/ldc"
exec > modifiedsinckan_re3200_run.log 2> modifiedsinckan_re3200_run.err
source /root/modifiedkan-gpu/bin/activate
export PYTHONPATH=../..

python ./main_modifiedsinckan.py --workdir=. --config=./configs/modifiedsinckan_re3200.py

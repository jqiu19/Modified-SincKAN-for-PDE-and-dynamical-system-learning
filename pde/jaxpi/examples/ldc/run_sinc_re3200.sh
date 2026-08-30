#!/usr/bin/env bash
set -euo pipefail

cd "/mnt/c/Users/Qiu Jingwei/Documents/New project/jaxpi/examples/ldc"
exec > sinc_re3200_run.log 2> sinc_re3200_run.err
source /root/modifiedkan-gpu/bin/activate
export PYTHONPATH=../..

python ./main_sinc.py --workdir=. --config=./configs/sinc_re3200.py

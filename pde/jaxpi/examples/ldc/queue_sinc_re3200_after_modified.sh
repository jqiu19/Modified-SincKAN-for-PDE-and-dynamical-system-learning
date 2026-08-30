#!/usr/bin/env bash
set -euo pipefail

while pgrep -f "main_modifiedsinckan.py.*modifiedsinckan_re3200.py" > /dev/null; do
  sleep 300
done

bash "/mnt/c/Users/Qiu Jingwei/Documents/New project/jaxpi/examples/ldc/run_sinc_re3200.sh"

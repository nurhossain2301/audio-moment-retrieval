#!/bin/bash

set -euo pipefail

cd /home/mkhan/audio-moment-retrieval

python3 -m venv AF-tuning/venv
source AF-tuning/venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r AF-tuning/requirements.txt

echo "AF-tuning environment is ready at AF-tuning/venv"

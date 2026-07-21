#!/usr/bin/env bash
# Run once on a fresh RunPod pod (PyTorch template, single 24GB GPU) to install deps and pre-fetch the model.
set -euo pipefail

cd "$(dirname "$0")/.."

pip install --upgrade pip
pip install -r requirements.txt

python3 -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
name = 'Qwen/Qwen2.5-Coder-7B-Instruct'
AutoTokenizer.from_pretrained(name)
AutoModelForCausalLM.from_pretrained(name)
print('model cached')
"

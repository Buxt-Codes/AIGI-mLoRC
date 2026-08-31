#!/usr/bin/env bash
# Creates a venv and installs dependencies.
set -euo pipefail

python3 -m venv venv
venv/bin/pip install --upgrade pip
venv/bin/pip install -r requirements.txt

echo ""
echo "Setup complete. Activate with: source venv/bin/activate"
echo "Then either:"
echo "  huggingface-cli login              # if you have HF access to buxtcodes/TechJam-Modulated-LoRC"
echo "  export HF_TOKEN=hf_...              # or pass a token directly"
echo "Run: python predict.py --input_dir <images_dir> --output results.json"

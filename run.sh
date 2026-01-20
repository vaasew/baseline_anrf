#!/bin/bash
set -e

echo "running full pipeline - prepare dataset -> train -> infer -> eval"

python ./scripts/prepare_dataset.py
python ./scripts/train.py
python ./scripts/infer.py
python ./scripts/eval.py


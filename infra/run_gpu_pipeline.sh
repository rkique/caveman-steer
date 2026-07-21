#!/usr/bin/env bash
# The three GPU-bound steps, run in order on the pod. data_prep.py, judge.py, and analyze.py
# do not need a GPU and are meant to run locally instead.
set -euo pipefail

cd "$(dirname "$0")/../src"

echo "== calibrating constant activation-steering direction/layer/coefficient on dev (steering on top of the caveman-style instruction) =="
python3 -u steering_const.py

echo "== generating all 4 conditions over the test split =="
python3 -u generate.py --split test

#!/usr/bin/env bash
# Full audit run: 400 requests per model/condition (20 prompt variants x 20
# repetitions) x 3 models x 3 conditions = 3,600 generations, then scoring and
# analysis.
set -eo pipefail
cd "$(dirname "$0")"
PY="${PY:-../.venv/bin/python}"   # override with e.g.  PY=python ./run_full.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # less fragmentation
[ -f results/requests.jsonl ] || $PY bias_audit.py build
# every stage resumes from saved output, so a transient crash is handled by
# simply retrying
retry() { for try in 1 2 3 4 5; do "$@" && return 0; echo "retry ($try): $*"; done; return 1; }
for m in qwen3-8b phi4-mini falcon3-7b; do
  retry $PY bias_audit.py generate --model $m --batch-size 64
done
$PY bias_audit.py score
$PY bias_audit.py analyze

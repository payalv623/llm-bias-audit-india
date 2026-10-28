#!/usr/bin/env bash
# Full audit run: 720 requests per model/condition (20 prompts, 6-8 fills
# per group, 2 samples per fill) x 4 models x 3 conditions = 8,640 generations,
# then the independent judge (quality + classifier labels), scoring and analysis.
set -eo pipefail
cd "$(dirname "$0")"
PY="${PY:-../.venv/bin/python}"   # override with e.g.  PY=python ./run_full.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # less fragmentation
[ -f results/requests.jsonl ] || $PY bias_audit.py build
# every stage resumes from saved output, so a transient crash (rare segfaults
# were seen on this machine) is handled by simply retrying
retry() { for try in 1 2 3 4 5; do "$@" && return 0; echo "retry ($try): $*"; done; return 1; }
for m in qwen3-8b phi4-mini granite3.3-8b falcon3-7b; do
  retry $PY bias_audit.py generate --model $m --batch-size 64
done
# wait until every weight shard of the judge model has finished downloading
until $PY -c "
import json, glob, sys
from huggingface_hub import try_to_load_from_cache as c
idx = c('allenai/OLMo-2-1124-13B-Instruct', 'model.safetensors.index.json')
d = idx.rsplit('/', 1)[0]
need = set(json.load(open(idx))['weight_map'].values())
sys.exit(0 if all(glob.glob(d + '/' + f) for f in need) else 1)" 2>/dev/null; do sleep 30; done
# the 13B judge needs ~26 GB for its weights: wait until enough GPU memory is free
until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -ge 28000 ]; do sleep 60; done
retry $PY bias_audit.py judge --judge-batch-size 32
$PY bias_audit.py score
$PY bias_audit.py analyze
echo "ALL DONE"

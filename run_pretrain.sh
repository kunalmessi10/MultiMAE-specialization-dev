#!/usr/bin/env bash

set -euo pipefail

# Auto-detect GPU count for torchrun --nproc_per_node.
if command -v nvidia-smi >/dev/null 2>&1; then
	GPU_COUNT="$(nvidia-smi --list-gpus | wc -l | tr -d ' ')"
else
	GPU_COUNT="$(python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)"
fi

# Fallback to 1 process if GPU detection fails or returns 0.
if [[ -z "${GPU_COUNT}" || "${GPU_COUNT}" -lt 1 ]]; then
	GPU_COUNT=1
fi

OMP_NUM_THREADS=1 torchrun --nproc_per_node="${GPU_COUNT}" run_pretraining_multimae.py \
	--config cfgs/pretrain/multimae-b_98_rgb+-depth-semseg_400e.yaml \
	--data_path /scratch/kunal/mae_expts/data/imagenet_val \
	--in_domains rgb \
	--out_domains rgb

#!/bin/sh
#SBATCH --partition=yss
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=10G
#SBATCH --time=00:30:00
#SBATCH -o ./logs/%j.out
#SBATCH -e ./logs/%j.err

set -eu

mkdir -p ./logs

PERSISTENT_CACHE=/accounts/projects/binyu/hao_huang/.cache/huggingface

if [ -n "${SLURM_TMPDIR:-}" ]; then
    TMP_BASE="$SLURM_TMPDIR"
else
    echo "SLURM_TMPDIR not set, using /tmp"
    TMP_BASE="/tmp/$USER/$SLURM_JOB_ID"
    mkdir -p "$TMP_BASE"
fi

export HF_HOME="$TMP_BASE/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"

mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"
mkdir -p "${PERSISTENT_CACHE}/hub" "${PERSISTENT_CACHE}/datasets"

echo "Using HF cache at: $HF_HOME"
echo "Copying HF cache to local disk..."
rsync -a "${PERSISTENT_CACHE}/hub/" "${HF_HUB_CACHE}/"
rsync -a "${PERSISTENT_CACHE}/datasets/" "${HF_DATASETS_CACHE}/"

echo "running smoke test"

python eval_baseline.py \
  --model_id meta-llama/Meta-Llama-3-8B-Instruct \
  --backend vllm \
  --benchmarks harmful_eval orbench \
  --harmful_eval_path data/splits/harmful_eval.jsonl \
  --orbench_eval_path data/splits/orbench_hard_eval.jsonl \
  --max_samples 50 \
  --max_new_tokens 128

# echo "run full"
# python eval_baseline.py \
#   --model_id meta-llama/Meta-Llama-3-8B-Instruct \
#   --backend vllm \
#   --benchmarks harmbench orbench \
#   --harmbench_split standard contextual copyright \
#   --max_new_tokens 128 \
#   --orbench_difficulty or-bench-80k \


echo "========================================"
echo "Finished : $(date)"
echo "========================================"
#!/bin/bash
# Run BaseUSE training (URGENT 2025) with nohup-style logging.
#
# Usage:
#   bash scripts/train.sh rwsamamba          # RWSAMamba-UNet (SFI multi-fs, no GAN)
#   bash scripts/train.sh semamba            # flat SEMamba (SFI multi-fs, no GAN)
#   bash scripts/train.sh semamba-lenmask    # SEMamba + length-masked losses (see
#                                            #   conf/exp/semamba_2025_lenmask.yaml:
#                                            #   losses averaged over valid frames
#                                            #   instead of the batch's zero padding;
#                                            #   NOT comparable to semamba's val_loss)
#   bash scripts/train.sh                    # default: flowse_2025_dynamic.yaml
#   bash scripts/train.sh bsrnn              # bsrnn_2025_dynamic.yaml
#   CONFIG=conf/exp/my_exp.yaml bash scripts/train.sh
#
# Extra args are forwarded to baseuse.train, e.g.:
#   bash scripts/train.sh bsrnn --batch_size 2
#
# Requires: GPU node (nvidia-smi visible), data manifests already built
# (scripts/build_urgent2025_manifests.sh), ffmpeg on PATH or via FFMPEG_PATH.

set -e
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# ---------------- environment ----------------
# conda env (python + ffmpeg live here); override with PYTHON_BIN=...
PYTHON_BIN="${PYTHON_BIN:-/mnt/data/share-ssd/user/wangsuting/softwares/miniforge3/envs/py310/bin/python}"

# ffmpeg: prefer PATH, else point at the conda env binary
if ! command -v ffmpeg >/dev/null 2>&1; then
    FFMPEG_DIR="$(dirname "${PYTHON_BIN}")"
    export FFMPEG_PATH="${FFMPEG_PATH:-${FFMPEG_DIR}/ffmpeg}"
    echo "ffmpeg not on PATH, using FFMPEG_PATH=${FFMPEG_PATH}"
fi

# dynamic mixing spawns worker processes; keep them single-threaded
export OMP_NUM_THREADS=1

# silence harmless warnings: transformers 5.x needs torch>=2.4 (py310 has 2.2.1)
# and prints 2 lines in every DataLoader worker process; training never uses transformers
export TRANSFORMERS_VERBOSITY=error

# reduce allocator fragmentation across the 7 sampling-rate shape families
# (bsrnn uses a different T/K/F shape per fs; expandable segments avoid the
# "reserved but unallocated" gaps that pushed 80GB cards into OOM)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---------------- experiment selection ----------------
MODEL_ARG="${1:-flowse}"
if [ -n "${CONFIG:-}" ]; then
    CONFIG_FILE="${CONFIG}"
elif [ "${MODEL_ARG}" = "flowse" ]; then
    CONFIG_FILE="conf/exp/flowse_2025_dynamic.yaml"
elif [ "${MODEL_ARG}" = "bsrnn" ]; then
    CONFIG_FILE="conf/exp/bsrnn_2025_dynamic.yaml"
elif [ "${MODEL_ARG}" = "rwsamamba" ]; then
    CONFIG_FILE="conf/exp/rwsamamba_2025_dynamic.yaml"
elif [ "${MODEL_ARG}" = "semamba" ]; then
    CONFIG_FILE="conf/exp/semamba_2025_dynamic.yaml"
elif [ "${MODEL_ARG}" = "semamba-lenmask" ]; then
    CONFIG_FILE="conf/exp/semamba_2025_lenmask.yaml"
else
    echo "unknown model arg '${MODEL_ARG}' (use flowse|bsrnn|rwsamamba|semamba|semamba-lenmask, or set CONFIG=...)" >&2
    exit 1
fi
shift || true
EXTRA_ARGS="$*"

# ---------------- sanity checks ----------------
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "WARNING: nvidia-smi not found - training needs a GPU node!" >&2
fi
if [ ! -f "data/train_sources_2025/speech_sources.scp" ]; then
    echo "ERROR: data/train_sources_2025 not found. Run first:" >&2
    echo "  NUM_WORKERS=64 bash scripts/build_urgent2025_manifests.sh" >&2
    exit 1
fi
if [ ! -f "data/val_2025/wav.scp" ]; then
    echo "ERROR: data/val_2025 not found. Run first:" >&2
    echo "  bash scripts/build_urgent2025_manifests.sh" >&2
    exit 1
fi

# ---------------- launch ----------------
EXP_TAG="$(basename "${CONFIG_FILE}" .yaml)"
LOG_DIR="exp/${EXP_TAG}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"

echo "config : ${CONFIG_FILE}"
echo "python : ${PYTHON_BIN}"
echo "log    : ${LOG_FILE}"
echo "extra  : ${EXTRA_ARGS:-none}"

"${PYTHON_BIN}" -u -m baseuse.train \
    --config_file "${CONFIG_FILE}" \
    ${EXTRA_ARGS} \
    2>&1 | tee "${LOG_FILE}"

echo "done. checkpoints under ${LOG_DIR}/"

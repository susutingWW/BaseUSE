#!/bin/bash
# Build BaseUSE training/validation manifests from the URGENT 2025 repo.
#
# Produces (text manifests only, no audio copied or simulated):
#   data/train_sources_2025/{speech_sources,noise_scoures,wind_noise_scoures,rirs,source_length}.scp
#   data/val_2025/{spk1.scp,wav.scp,utt2fs,speech_length.scp}
#
# Notes:
# - source_length.scp scans ~1.4M audio files in parallel (NUM_WORKERS, default 32);
#   run once. Tune NUM_WORKERS to the network-storage throughput, e.g.:
#     NUM_WORKERS=64 bash scripts/build_urgent2025_manifests.sh
# - For a smoke test add e.g. --subsets ears,vctk --speech_subset_size 2000 --skip_validation
# - Full smoke test variant:
#     bash scripts/build_urgent2025_manifests.sh smoke

set -e
set -u

URGENT25_PATH="${URGENT25_PATH:-/mnt/data/share-oss/user/wangsuting/workspace/USE/urgent2025_challenge}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data}"
NUM_WORKERS="${NUM_WORKERS:-32}"

if [ "${1:-full}" = "smoke" ]; then
    EXTRA_ARGS="--subsets ears,vctk --speech_subset_size 2000 --skip_source_length"
    echo "== smoke mode: small pool, no source_length, no validation =="
else
    EXTRA_ARGS=""
fi

python -m baseuse.data.build_urgent2025 \
    --urgent25_path "${URGENT25_PATH}" \
    --output_root "${OUTPUT_ROOT}" \
    --num_workers "${NUM_WORKERS}" \
    ${EXTRA_ARGS}

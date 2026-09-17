"""Build BaseUSE training/validation manifests from the URGENT 2025 repo.

Only text manifests are produced here; audio files stay in the urgent2025
repo and are referenced by absolute paths. No simulated audio is generated.

Train pool (dynamic mixing source manifests, 3-column `uid fs path`):
    speech_sources.scp       <- 12 speech subsets in urgent25/data/tmp/
    noise_scoures.scp        <- 4 noise subsets
    wind_noise_scoures.scp   <- wind_noise_train.scp (already absolute)
    rirs.scp                 <- dns5_rirs.scp
    source_length.scp        <- per-utterance sample counts (parallel probing, I/O-bound)

Validation (pre-simulated pairs already on disk in urgent25):
    spk1.scp / wav.scp       <- urgent25/data/validation/*, paths made absolute
    utt2fs                   <- copied as-is
    speech_length.scp        <- generated from wav.scp (fast, ~2k files)

Usage (NOT part of training; run once before the first training):

    python -m baseuse.data.build_urgent2025 \
        --urgent25_path /mnt/data/share-oss/user/wangsuting/workspace/USE/urgent2025_challenge \
        --output_root data \
        --subsets all

    # smaller smoke-test pool:
    python -m baseuse.data.build_urgent2025 --subsets ears,vctk --speech_subset_size 2000
"""

import argparse
import os
import random

from baseuse.data.manifests import (
    abspath_kv_scp,
    abspath_source_scp,
    concat_scps,
)

DEFAULT_URGENT25_PATH = "/mnt/data/share-oss/user/wangsuting/workspace/USE/urgent2025_challenge"

SPEECH_SUBSETS = {
    "dns5": "data/tmp/dns5_clean_read_speech_resampled_filtered_train.scp",
    "libritts": "data/tmp/libritts_resampled_train.scp",
    "vctk": "data/tmp/vctk_train.scp",
    "ears": "data/tmp/ears_train.scp",
    "common_en": "data/tmp/commonvoice_19.0_en_resampled_train_track1.scp",
    "common_de": "data/tmp/commonvoice_19.0_de_resampled_train_track1.scp",
    "common_es": "data/tmp/commonvoice_19.0_es_resampled_train_track1.scp",
    "common_fr": "data/tmp/commonvoice_19.0_fr_resampled_train_track1.scp",
    "common_zh": "data/tmp/commonvoice_19.0_zh-CN_resampled_train_track1.scp",
    "mls_de": "data/tmp/mls_german_resampled_train_track1.scp",
    "mls_es": "data/tmp/mls_spanish_resampled_train_track1.scp",
    "mls_fr": "data/tmp/mls_french_resampled_train_track1.scp",
}

NOISE_SUBSETS = [
    "data/tmp/dns5_noise_resampled_train.scp",
    "data/tmp/wham_noise_train.scp",
    "data/tmp/fma_noise_resampled_train.scp",
    "data/tmp/fsd50k_noise_resampled_train.scp",
]

WIND_NOISE_SCP = "data/tmp/wind_noise_train.scp"
RIR_SCP = "data/tmp/dns5_rirs.scp"
VALIDATION_DIR = "data/validation"


def _probe_length(entry):
    """Return ``(uid, num_samples)`` for one audio file (worker-process safe).

    wav/flac are probed from the header only (exact frame count, no decoding);
    a zero header count falls back to full decoding. Other formats are always
    fully decoded because their headers may disagree with the decoded length.
    """
    import soundfile as sf

    uid, path = entry
    if os.path.splitext(path)[1].lower() in (".wav", ".flac"):
        with sf.SoundFile(path) as af:
            frames = af.frames
        if frames > 0:
            return uid, frames
    return uid, sf.read(path)[0].shape[0]


def generate_source_length(input_scp, output_scp, num_workers=1):
    """Write `uid num_samples` for every entry of a (2- or 3-column) speech scp.

    Probing ~1.4M files on network storage is I/O-bound, so it runs in a
    multiprocessing pool when num_workers > 1. `Pool.imap` preserves input
    order, making the output byte-identical to the sequential version.
    """
    from tqdm import tqdm

    entries = []
    with open(input_scp, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) == 3:
                entries.append((parts[0], parts[2]))
            else:
                entries.append((parts[0], parts[1]))

    if num_workers <= 1:
        with open(output_scp, "w") as fout:
            for uid, length in tqdm(map(_probe_length, entries),
                                    total=len(entries),
                                    desc=f"source_length {os.path.basename(input_scp)}"):
                fout.write(f"{uid} {length}\n")
        return

    from multiprocessing import Pool
    with Pool(num_workers, maxtasksperchild=2000) as pool:
        with open(output_scp, "w") as fout:
            for uid, length in tqdm(pool.imap(_probe_length, entries, chunksize=128),
                                    total=len(entries),
                                    desc=f"source_length {os.path.basename(input_scp)}"):
                fout.write(f"{uid} {length}\n")


def subset_scp(in_scp, out_scp, size, seed=0):
    """Randomly keep `size` lines of an scp (for smoke tests)."""
    with open(in_scp, "r") as f:
        lines = [ln for ln in f if ln.strip()]
    rng = random.Random(seed)
    rng.shuffle(lines)
    lines = lines[:size]
    with open(out_scp, "w") as f:
        f.writelines(lines)
    return len(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--urgent25_path", type=str, default=DEFAULT_URGENT25_PATH,
                        help="Path to the urgent2025_challenge repo (audio source)")
    parser.add_argument("--output_root", type=str, default="data",
                        help="BaseUSE data directory (relative to CWD)")
    parser.add_argument("--subsets", type=str, default="all",
                        help="Comma-separated speech subset keys (see SPEECH_SUBSETS), or 'all'")
    parser.add_argument("--speech_subset_size", type=int, default=0,
                        help="If > 0, randomly subsample the pooled speech list to N lines (smoke test)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip_source_length", action="store_true",
                        help="Skip the slow source_length.scp generation (needs all audio readable)")
    parser.add_argument("--skip_validation", action="store_true",
                        help="Skip building the validation manifests")
    parser.add_argument("--num_workers", type=int, default=32,
                        help="Parallel workers for source_length probing (I/O-bound; raise on network storage)")
    args = parser.parse_args()

    urgent25 = os.path.abspath(args.urgent25_path)
    assert os.path.isdir(urgent25), f"urgent2025 repo not found: {urgent25}"

    if args.subsets.strip().lower() == "all":
        subset_keys = list(SPEECH_SUBSETS.keys())
    else:
        subset_keys = [s.strip() for s in args.subsets.split(",") if s.strip()]
        unknown = [k for k in subset_keys if k not in SPEECH_SUBSETS]
        assert not unknown, f"unknown subsets: {unknown} (available: {list(SPEECH_SUBSETS)})"

    # ---------------- train sources ----------------
    train_dir = os.path.join(args.output_root, "train_sources_2025")
    os.makedirs(train_dir, exist_ok=True)

    speech_scps = [os.path.join(urgent25, SPEECH_SUBSETS[k]) for k in subset_keys]
    for scp in speech_scps:
        assert os.path.isfile(scp), f"missing manifest (run prepare_espnet_data.sh in urgent2025?): {scp}"
    n_speech = concat_scps(speech_scps, os.path.join(train_dir, "speech_sources.scp"),
                           base_dir=urgent25, abspath=True)
    print(f"[train] speech_sources.scp: {n_speech} utterances from {len(subset_keys)} subsets")

    if args.speech_subset_size > 0:
        subset_scp(os.path.join(train_dir, "speech_sources.scp"),
                   os.path.join(train_dir, "speech_sources.scp.tmp"),
                   args.speech_subset_size, seed=args.seed)
        os.replace(os.path.join(train_dir, "speech_sources.scp.tmp"),
                   os.path.join(train_dir, "speech_sources.scp"))
        print(f"[train] subsampled speech pool to {args.speech_subset_size} utterances")

    noise_scps = [os.path.join(urgent25, rel) for rel in NOISE_SUBSETS]
    n_noise = concat_scps(noise_scps, os.path.join(train_dir, "noise_scoures.scp"),
                          base_dir=urgent25, abspath=True)
    print(f"[train] noise_scoures.scp: {n_noise} noise files")

    n_wind = abspath_source_scp(os.path.join(urgent25, WIND_NOISE_SCP),
                                os.path.join(train_dir, "wind_noise_scoures.scp"),
                                base_dir=urgent25)
    print(f"[train] wind_noise_scoures.scp: {n_wind} wind-noise files")

    n_rir = abspath_source_scp(os.path.join(urgent25, RIR_SCP),
                               os.path.join(train_dir, "rirs.scp"),
                               base_dir=urgent25)
    print(f"[train] rirs.scp: {n_rir} RIRs")

    if not args.skip_source_length:
        generate_source_length(os.path.join(train_dir, "speech_sources.scp"),
                               os.path.join(train_dir, "source_length.scp"),
                               num_workers=args.num_workers)
    else:
        print("[train] skipped source_length.scp (remember to generate it before training)")

    # ---------------- validation ----------------
    if not args.skip_validation:
        val_src = os.path.join(urgent25, VALIDATION_DIR)
        val_dir = os.path.join(args.output_root, "val_2025")
        os.makedirs(val_dir, exist_ok=True)

        for name in ("spk1.scp", "wav.scp"):
            n = abspath_kv_scp(os.path.join(val_src, name),
                               os.path.join(val_dir, name), base_dir=urgent25)
            print(f"[val] {name}: {n} entries")

        # utt2fs has no paths, copy verbatim
        import shutil
        shutil.copyfile(os.path.join(val_src, "utt2fs"), os.path.join(val_dir, "utt2fs"))

        generate_source_length(os.path.join(val_dir, "wav.scp"),
                               os.path.join(val_dir, "speech_length.scp"),
                               num_workers=args.num_workers)

    print("done.")


if __name__ == "__main__":
    main()

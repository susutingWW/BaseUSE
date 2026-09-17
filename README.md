# BaseUSE

Decoupled universal speech enhancement (USE) baseline, extracted from
[urgent2026_challenge_track1](https://github.com/urgent-challenge/urgent2026_challenge_track1)
`baseline_code`, trainable on the **URGENT 2025** dataset.

Design rules:

- **Model code is byte-compatible with the official baseline** — only package
  names changed, so official pretrained checkpoints (e.g. `bsrnn.ckpt` trained
  on 700h-TBF) load directly.
- **Dataset layer is config-driven**: the on-the-fly simulation recipe that was
  hardcoded in `baseline_code/dataset.py::SimulationConfigs` is now a yaml
  section; ffmpeg path and `retry_when_fails` are configurable.
- **This repo never modifies or imports the urgent2025 / urgent2026 repos**;
  their audio is referenced through absolute-path scp manifests only.

## Layout

```
BaseUSE/
├── baseuse/
│   ├── models/          # SEModel (BSRNN), FlowSEModel (BSRNN-Flow), RWSAMambaSEModel,
│   │   │                # BSRNN blocks, ODE solvers
│   │   └── rwsamamba/   # vendored RWSAMamba-UNet backbone (see below)
│   ├── data/            # datasets, datamodule, manifest utilities
│   │   ├── dataset.py   #   PreSimulated / DynamicMixing datasets + batch sampler + collate
│   │   ├── datamodule.py#   AudioDataModule (reads conf `data:` section)
│   │   ├── manifests.py #   scp readers / path rewriters
│   │   └── build_urgent2025.py  # builds training/val manifests from the 2025 repo
│   ├── simulation/      # noise/RIR/augmentation simulation engine (on-the-fly & offline)
│   ├── evaluation/      # URGENT objective metrics (copied unchanged)
│   ├── train.py         # entry: training
│   ├── inference.py     # entry: inference
│   ├── evaluate.py      # entry: metric orchestration
│   └── config.py        # experiment config (yaml + CLI)
├── conf/exp/            # experiment yamls (model + data + training)
├── scripts/             # manifest builders (not executed automatically)
├── data/                # manifests only (audio stays in the 2025 repo; gitignored)
└── exp/                 # logs / checkpoints (gitignored)
```

## Setup

```bash
conda create -n baseuse python=3.10
conda activate baseuse
conda install ffmpeg                 # needed by on-the-fly wind-noise / torchaudio codecs
cd BaseUSE
pip install -e ./

# evaluation deps (separate, heavy):
pip install pip==24.0
pip install -r baseuse/evaluation/requirements.txt
```

If `ffmpeg` is not on `PATH`, point the simulation engine at it:
`export FFMPEG_PATH=/path/to/ffmpeg`.

## Data manifests (run once, before training)

```bash
# full pool (~1.4M utterances from all URGENT 2025 track1 subsets)
bash scripts/build_urgent2025_manifests.sh

# smoke-test pool instead:
bash scripts/build_urgent2025_manifests.sh smoke
```

Outputs (text only):

```
data/train_sources_2025/{speech_sources,noise_scoures,wind_noise_scoures,rirs,source_length}.scp
data/val_2025/{spk1.scp,wav.scp,utt2fs,speech_length.scp}   # from urgent2025 data/validation
```

## Training

```bash
# dynamic mixing (recommended: no simulated audio on disk)
python -m baseuse.train --config_file conf/exp/bsrnn_2025_dynamic.yaml

# generative FlowSE variant
python -m baseuse.train --config_file conf/exp/flowse_2025_dynamic.yaml

# RWSAMamba-UNet (pure-Mamba variant, SFI-STFT multi-sampling-rate, no GAN)
# needs: pip install torchvision==0.17.1 causal-conv1d>=1.2.0 mamba-ssm==1.2.2
python -m baseuse.train --config_file conf/exp/rwsamamba_2025_dynamic.yaml
```

RWSAMamba-UNet notes: backbone vendored from
[NikolaiKyhne/RWSAMamba-UNet](https://github.com/NikolaiKyhne/RWSAMamba-UNet)
into `baseuse/models/rwsamamba/`; the wrapper (`baseuse/models/rwsa_se.py`)
uses espnet's SFI-STFT encoder/decoder so one model serves 8k-48k (window /
hop duration fixed at the 16 kHz reference 510/120), trains with the upstream
generator loss only (no discriminator / metric loss), and defaults to the
attention-free `TFMambaBlock` variant (`use_rwsa: false`; set true to restore
the paper's shared attention). Checkpoints save/load through the same
train/inference entries as the other models.

Quick self-test (GPU machine with mamba-ssm installed):
`python scripts/rwsamamba_selftest.py` (see its `--help` for optional
official-checkpoint alignment and upstream-equivalence checks).

Caveat shared with the other BaseUSE configs: `baseuse/train.py` re-applies
the yaml over CLI flags (`cfg.read_yaml()`), so any field present in the
experiment yaml must be changed in a copy of the yaml, not via `--flag`.

Checkpoints: `exp/<exp_name>/bsrnn_2025dm/version_0/checkpoints/`, selected by
`val_loss` every 5000 steps; rerunning the same command resumes automatically.

With the full 2025 pool one epoch is ~350k steps (bs=4) — rely on checkpoint
selection and stop when validation plateaus; `num_train_epochs` is an upper
bound. Use `--speech_subset_size N` in the manifest builder for smaller pools.

## Inference

```bash
python -m baseuse.inference \
    --input_scp /path/to/noisy.scp \
    --output_dir ./exp/enhanced \
    --ckpt_path ./exp/bsrnn_2025_dynamic/bsrnn_2025dm/version_0/checkpoints/best_xxx.ckpt
```

Each utterance is processed at its native sampling rate (8k–48k); enhanced
audio is written to `<output_dir>/wav/<uid>.wav` and indexed in `inf.scp`.

## Evaluation

```bash
python -m baseuse.evaluate \
    --inf_scp ./exp/enhanced/inf.scp \
    --ref_scp /path/to/clean.scp \
    --output_dir ./exp/scores \
    --metrics intrusive,dnsmos,nisqa,utmos,speechbert,lps,spksim,wer \
    --meta_tsv /path/to/meta.tsv --utt2lang /path/to/utt2lang \
    --nj 8 --device cuda
```

- `intrusive` (PESQ/ESTOI/SDR/MCD/LSD) needs `--ref_scp`; `wer` needs
  `--meta_tsv` + `--utt2lang`; the rest are non-intrusive.
- DNSMOS needs `sig_bak_ovr.onnx` / `model_v8.onnx` (`--dnsmos_dir`), NISQA
  needs weights under `baseuse/evaluation/lib/NISQA`, scoreq under
  `baseuse/evaluation/lib/scoreq` (copy or symlink from the 2026 repo, or
  download from the official sources).
- For eSpeak-NG based phoneme similarity (LPS), install eSpeak-NG first.

## Checkpoint compatibility

Checkpoints saved by the original `baseline_code` pickle a
`baseline_code.config.Config` object in `hyper_parameters`. `baseuse.utils.compat`
aliases that class path to `baseuse.config.Config`, so official pretrained
checkpoints (BSRNN / BSRNN-Flow from the 2026 challenge) can be loaded with
`--init_from` (training) or `--ckpt_path` (inference) without the original repo.

## Status

- [x] Project scaffold, models, data layer, simulation, entries, configs
- [x] Evaluation scripts copied (dependency weights pending, see above)
- [ ] Data manifests (run `scripts/build_urgent2025_manifests.sh` when ready)
- [ ] Smoke test: small-pool overfit + equivalence check vs original baseline

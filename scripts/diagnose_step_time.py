"""Diagnose where a training step's wall time goes. v2 (units fixed, parts
split, A/B toggles added).

Run ON THE GPU NODE from the BaseUSE repo root:

    # [A] CPU simulation cost only (no CUDA touched - safe before [C])
    python scripts/diagnose_step_time.py --part a

    # [C] dataloader-only cadence (MUST run in a process where CUDA was
    #     never initialized, else forked workers deadlock; this part sets
    #     OMP_NUM_THREADS=1 itself)
    python scripts/diagnose_step_time.py --part c

    # [B] GPU model cost: mamba micro-bench + ablation grid + bs sweep
    python scripts/diagnose_step_time.py --part b
    python scripts/diagnose_step_time.py --part b --quick   # skip bs sweep

Findings this version is designed to confirm (from v1 run):
  - model fwd+bwd @bs=8 takes 1.0-2.4 s/step (130-300 ms/sample) -> the
    training step (~2.7 s) is MODEL-bound, not simulation-bound
  - peak activation 20.6 GiB @bs=8 -> per-sample ~2.6 GiB
"""

import os
import sys


def _pick_part():
    return "--part" in sys.argv


def _set_omp_if_c():
    # must happen BEFORE numpy/torch import; matches train.sh
    if "--part c" in " ".join(sys.argv) or ("--part" in sys.argv and "c" in sys.argv):
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")


if __name__ != "__main__":
    raise SystemExit("script, not a module")

_set_omp_if_c()

import argparse
import time
from collections import defaultdict

import numpy as np
import torch

PARTS = {"a": "simulation", "b": "model", "c": "dataloader"}


def fmt_s(x):
    """seconds -> human string (v1 had a bug printing seconds as 'ms')."""
    if x >= 1.0:
        return f"{x:7.2f} s "
    if x >= 1e-3:
        return f"{x * 1e3:7.1f} ms"
    return f"{x * 1e6:7.0f} us"


def bench_simulation(cfg, n=40, seed=1234):
    """[A] pure per-sample simulation cost, single process."""
    from baseuse.data.datamodule import AudioDataModule
    dm = AudioDataModule(cfg)
    ds = dm.train_dataset
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(ds), size=n, replace=False)

    _ = ds[int(idxs[0])]  # warmup: ffmpeg lazy init, first OSS opens

    t_open = t_sim = 0.0
    for i in idxs:
        i = int(i)
        t0 = time.perf_counter()
        fs, ri = ds._get_from_index(i)
        path = ds.speech_source[fs][ds.speech_uids[fs][ri]]
        import soundfile
        with soundfile.SoundFile(path) as af:
            af.frames
        t1 = time.perf_counter()
        ds[i]
        t2 = time.perf_counter()
        t_open += t1 - t0
        t_sim += t2 - t1

    per = t_sim / n
    print("\n[A] on-the-fly simulation, 1 process, %d samples" % n)
    print("    header-open : %s / sample" % fmt_s(t_open / n))
    print("    getitem     : %s / sample  (includes its own file open)" % fmt_s(per))
    print("    => 1 worker: %.2f samples/s | %d workers -> %.1f samples/s ceiling"
          % (1.0 / per, cfg.num_worker, cfg.num_worker / per))
    print("    => per training batch of %d: %.0f ms of CPU work, amortized to "
          "%.0f ms by %d workers (negligible if model is ~2000 ms/step)"
          % (cfg.batch_size, per * cfg.batch_size * 1e3,
             per * cfg.batch_size / cfg.num_worker * 1e3, cfg.num_worker))


def mamba_microbench():
    """[B0] raw Mamba layer speed: is the CUDA selective-scan kernel live?

    The UNet runs 20 TFMambaBlocks x (time+frequent) x (fwd+flip) = 80 scans
    per forward. If one scan at production size costs >20 ms the kernel is
    missing/slow; ~1-5 ms means the kernel is fine and the cost is legit.
    """
    try:
        import mamba_ssm
        from mamba_ssm.modules.mamba_simple import Mamba
        print("    mamba_ssm %s" % getattr(mamba_ssm, "__version__", "?"))
    except Exception as e:
        print("    mamba_ssm import FAILED: %r" % e)
        return
    try:
        import causal_conv1d  # noqa: F401
        print("    causal_conv1d: OK (fast conv path)")
    except ImportError:
        print("    causal_conv1d: MISSING (manual conv fallback, slower)")

    dev = torch.device("cuda")
    # production size at 48k level-1: batch B*F=8*766, seq T=533, d_model 16->d_inner 64
    for tag, bsz, seq, d in [("48k time-scan", 6128, 533, 16),
                             ("16k time-scan", 2056, 533, 16)]:
        m = Mamba(d_model=d, d_state=16, d_conv=4, expand=4).to(dev)
        x = torch.randn(bsz, seq, d, device=dev, requires_grad=True)
        for _ in range(3):
            m(x).sum().backward()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            m(x).sum().backward()
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / 10
        print("    %-14s bsz=%5d seq=%3d: %s / layer fwd+bwd"
              % (tag, bsz, seq, fmt_s(dt)))


def bench_model_ablation(cfg, fs_list=(16000, 48000), bs_list=None, quick=False):
    """[B] production fwd+bwd with bf16 autocast (like training), ablation
    grid over grad_checkpoint / mamba_fp32, plus a batch-size sweep."""
    from baseuse.models.rwsa_se import RWSAMambaSEModel
    from baseuse.models.semamba_se import SEMambaSEModel
    assert torch.cuda.is_available(), "run part b on the GPU node"
    if cfg.se_model == 'semamba':
        model_cls = SEMambaSEModel
    elif cfg.se_model == 'rwsamamba_unet':
        model_cls = RWSAMambaSEModel
    else:
        raise SystemExit("--part b supports se_model = semamba | rwsamamba_unet"
                         f" (got {cfg.se_model!r})")
    n_params = sum(p.numel() for p in model_cls(cfg).parameters()) / 1e6
    print("\n    model class: %s (%.2fM params)\n" % (model_cls.__name__, n_params))
    torch.set_float32_matmul_precision("medium")  # match train.py
    dev = torch.device("cuda")

    print("\n[B0] mamba micro-bench")
    mamba_microbench()

    def run(bs, fs, gc, fp32):
        mc = dict(cfg.model_configs)
        mc["grad_checkpoint"] = gc
        mc["mamba_fp32"] = fp32
        c = cfg
        c.model_configs = mc
        model = model_cls(cfg=c).to(dev)
        T = min(4 * fs, 96000)
        noisy = torch.randn(bs, 1, T, device=dev) * 0.05
        clean = torch.randn(bs, 1, T, device=dev) * 0.05
        batch = (clean, noisy, torch.tensor(fs, dtype=torch.int32),
                 torch.full((bs,), T, dtype=torch.int32))
        for _ in range(3):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model.training_step(batch)
            loss.backward()
            model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        iters = 10
        for _ in range(iters):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model.training_step(batch)
            loss.backward()
            model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / iters
        mem = torch.cuda.max_memory_allocated() / 2**30
        del model
        torch.cuda.empty_cache()
        return dt, mem

    print("\n[B1] ablation grid (bs=%d, production autocast bf16)" % cfg.batch_size)
    for fs in fs_list:
        for gc in (True, False):
            for fp32 in (True, False):
                dt, mem = run(cfg.batch_size, fs, gc, fp32)
                print("    fs=%5d  grad_ckpt=%-5s mamba_fp32=%-5s : %s/step "
                      "(%.0f ms/sample)  peak %.1f GiB"
                      % (fs, gc, fp32, fmt_s(dt), dt * 1e3 / cfg.batch_size, mem))

    if not quick:
        print("\n[B2] batch-size sweep (fs=48000, grad_ckpt+fp32 as configured)")
        for bs in (bs_list or (8, 16, 24, 32)):
            try:
                dt, mem = run(bs, 48000, True, True)
                print("    bs=%2d: %s/step -> %.2f samples/s  peak %.1f GiB"
                      % (bs, fmt_s(dt), bs / dt, mem))
            except torch.cuda.OutOfMemoryError:
                print("    bs=%2d: OOM" % bs)
                torch.cuda.empty_cache()
    print("\n    NOTE: global batch = 4 GPUs x per-GPU bs; if you raise bs,"
          " either accept global-batch change or rescale lr.")


def bench_dataloader(cfg, max_batches=20):
    """[C] real DataLoader cadence. Run ALONE (`--part c`) so this process
    never initialized CUDA -> fork is safe; OMP already forced to 1."""
    from baseuse.data.datamodule import AudioDataModule
    dm = AudioDataModule(cfg)
    dl = dm.train_dataloader()
    n = min(max_batches, len(dl))
    print("\n[C] dataloader-only cadence (workers=%d, bs=%d, %d batches)"
          % (cfg.num_worker, cfg.batch_size, n))
    t0 = time.perf_counter()
    it = iter(dl)
    first = next(it)
    print("    first batch : %6.1f s (worker spawn + warmup)" % (time.perf_counter() - t0))
    t0 = time.perf_counter()
    k = 0
    for batch in it:
        k += 1
        if k >= n - 1:
            break
    dt = time.perf_counter() - t0
    per = dt / max(k, 1)
    print("    steady state: %6.2f s/batch -> %.2f samples/s (%.2f it/s)"
          % (per, k * cfg.batch_size / dt, 1.0 / per))
    print("    compare with training 0.37 it/s: if this is >> 0.37 it/s the"
          " model is the bottleneck (v1 measurements say it is)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config_file", default="conf/exp/rwsamamba_2025_dynamic.yaml")
    p.add_argument("--part", default="b", choices=["a", "b", "c"],
                    help="a=simulation, b=model(GPU), c=dataloader. Run c alone!")
    p.add_argument("--quick", action="store_true", help="skip the bs sweep in b")
    args = p.parse_args()

    from baseuse.config import Config
    cfg = Config(config_file=args.config_file)
    cfg.read_yaml()
    print("=" * 78)
    print("step-time diagnosis v2 | part [%s] | bs=%d workers=%d"
          % (args.part, cfg.batch_size, cfg.num_worker))
    print("=" * 78)

    if args.part == "a":
        bench_simulation(cfg)
    elif args.part == "b":
        bench_model_ablation(cfg, quick=args.quick)
    elif args.part == "c":
        bench_dataloader(cfg)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Self-test for the RWSAMamba-UNet / BaseUSE integration.

Usage (GPU machine, real mamba-ssm installed):
    python scripts/rwsamamba_selftest.py
    python scripts/rwsamamba_selftest.py --ckpt /path/to/RWSA_s.pth   # + state-dict alignment vs official
    python scripts/rwsamamba_selftest.py --upstream /path/to/RWSAMamba-UNet  # + numerical equivalence vs upstream stfts/loss

CPU machine without mamba-ssm (structure-only, CUDA kernels unavailable):
    python scripts/rwsamamba_selftest.py --stub /path/to/mamba_stub
"""

import argparse
import importlib.util
import os
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)


def setup(args):
    """Returns (device, stub_used). Prefers real mamba_ssm on CUDA; falls back
    to the CPU stub only when the real package is unavailable."""
    has_mamba = importlib.util.find_spec("mamba_ssm") is not None
    stub_dir = args.stub or os.environ.get("RWSA_MAMBA_STUB")
    if has_mamba and torch.cuda.is_available():
        return "cuda", False
    if stub_dir and os.path.isdir(stub_dir):
        sys.path.insert(0, stub_dir)  # stub shadows a real install
        return "cpu", True
    if has_mamba and not torch.cuda.is_available():
        print("mamba_ssm is installed but CUDA is not visible on this machine. "
              "Its selective-scan kernels are CUDA-only: run this self-test on "
              "the GPU machine, or pass --stub <dir> for a structure-only CPU run.")
        sys.exit(2)
    print("ERROR: mamba_ssm not installed (and no --stub given). Aborting.")
    sys.exit(1)


FS_LIST = [8000, 16000, 22050, 24000, 32000, 44100, 48000]


def t1_padding_helpers():
    from baseuse.models.rwsa_se import _freq_bins, _pad_bins, _pad_frames
    for fs in FS_LIST:
        n_fft = 510 * fs // 16000
        hop = 120 * fs // 16000
        bins = _freq_bins(510, fs, 16000)
        bins_pad = _pad_bins(bins)
        assert bins_pad % 8 == 0 and bins_pad - bins < 8, (fs, bins, bins_pad)
        assert bins_pad // 2 % 4 == 0, (fs, bins_pad)
        assert abs(hop - fs * 0.0075) <= 1 and abs(n_fft - fs * 0.031875) <= 1, (fs, hop, n_fft)
    assert _pad_frames(5) == 8 and _pad_frames(8) == 8 and _pad_frames(9) == 12
    print(f"T1 padding helpers OK (bins: {{{', '.join(f'{fs}:{_freq_bins(510, fs, 16000)}' for fs in FS_LIST)}}})")


def t2_pipeline_all_fs(device, num_tfmamba=1):
    from baseuse.models.rwsa_se import RWSAMambaUNet_SE
    torch.manual_seed(0)
    se = RWSAMambaUNet_SE(net_cfg={"num_tfmamba": num_tfmamba}).to(device)
    for fs in FS_LIST:
        noisy = torch.randn(2, fs, device=device)  # 1 second
        out = se.run(noisy, fs)
        assert out["est_wav"].size(0) == 2 and out["est_wav"].size(-1) >= fs
        assert out["est_mag"].shape[1] == out["n_bins"]
        for k in ("est_wav", "est_mag", "est_pha", "est_com"):
            assert torch.isfinite(out[k]).all(), (fs, k)
        est, _ = se(noisy.unsqueeze(1), None, fs)
        assert est.shape == noisy.shape and torch.isfinite(est).all()
    print("T2 full pipeline forward at 7 sampling rates OK")


def t3_loss_backward_optimizer(device):
    from baseuse.config import Config
    from baseuse.models.rwsa_se import RWSAMambaSEModel

    torch.manual_seed(0)
    cfg = Config()
    cfg.model_configs = {
        "hid_feature": 16, "num_tfmamba": 1, "d_state": 16, "d_conv": 4,
        "expand": 4, "norm_epsilon": 1e-5, "beta": 2.0, "compress_factor": 0.3,
        "use_rwsa": False, "loss_mag": 0.9, "loss_pha": 0.3, "loss_com": 0.1,
        "loss_time": 0.2, "loss_con": 0.1,
    }
    cfg.learning_rate, cfg.weight_decay, cfg.lr_gamma = 5e-4, 0.01, 0.99
    model = RWSAMambaSEModel(cfg).to(device)
    model.log = lambda *a, **k: None
    opt, _ = model.configure_optimizers()
    assert isinstance(opt[0], torch.optim.AdamW)
    assert abs(opt[0].defaults["betas"][0] - 0.8) < 1e-9

    def silent_tail(x, frac=0.35):
        """Real audio has exactly-zero STFT bins (silence/padding); the
        consistency-loss re-analysis backward must stay finite there."""
        n = int(x.size(-1) * frac)
        return torch.cat([x[..., :-n], torch.zeros_like(x[..., -n:])], dim=-1) if n else x

    for i, fs in enumerate(FS_LIST):
        L = fs  # 1 s
        clean = torch.randn(2, 1, L, device=device)
        noisy = clean + 0.1 * torch.randn(2, 1, L, device=device)
        if i % 2 == 1:  # silence variant every other fs
            clean, noisy = silent_tail(clean), silent_tail(noisy)
        batch = (clean, noisy, torch.tensor(fs, dtype=torch.int32),
                 torch.tensor([L, L], dtype=torch.int32))
        loss = model.forward_step(batch)
        assert torch.isfinite(loss), (fs, loss)
        loss.backward()
        for p in model.parameters():
            assert p.grad is None or torch.isfinite(p.grad).all(), f"NaN/inf grad at fs={fs}"
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in model.parameters()), fs
        opt[0].step()
        opt[0].zero_grad()
        print(f"    fs={fs:>5}{' (silent tail)' if i % 2 == 1 else '':13s} loss={loss.item():.4f}")
    print("T3 loss + backward + optimizer step at 7 sampling rates (incl. silence) OK")


def t4_lsigmoid_slicing(device):
    from baseuse.models.rwsamamba.lsigmoid import LearnableSigmoid2D
    torch.manual_seed(0)
    ls = LearnableSigmoid2D(768, beta=2.0).to(device)
    for f in (128, 256, 352, 383, 511, 703, 766):
        x = torch.randn(2, f, 16, device=device)
        assert ls(x).shape == x.shape
        assert torch.allclose(ls(x), 2.0 * torch.sigmoid(ls.slope[:f] * x))
    print("T4 LearnableSigmoid2D SFI slicing OK")


def t5_rwsa_variant(device):
    from baseuse.models.rwsa_se import RWSAMambaUNet_SE
    torch.manual_seed(0)
    se = RWSAMambaUNet_SE(net_cfg={"num_tfmamba": 1, "use_rwsa": True}).to(device)
    noisy = torch.randn(1, 16000, device=device)
    est, _ = se(noisy.unsqueeze(1), None, 16000)
    assert est.shape == noisy.shape and torch.isfinite(est).all()
    print("T5 use_rwsa=True variant (shared attention) OK")


def t6_param_count(device):
    from baseuse.models.rwsa_se import RWSAMambaUNet_SE
    n = sum(p.numel() for p in RWSAMambaUNet_SE().net.parameters())
    print(f"T6 backbone params (pure-Mamba, s-size): {n/1e6:.3f}M "
          f"(official RWSA_s reference: 1.99M incl. attention)")


def t7_yaml_build(device):
    from baseuse.config import Config, config_parser
    from baseuse.models.rwsa_se import RWSAMambaSEModel

    yaml_path = os.path.join(REPO_ROOT, "conf/exp/rwsamamba_2025_dynamic.yaml")
    old_argv = sys.argv
    sys.argv = ["x", "--config_file", yaml_path]
    args = config_parser()
    sys.argv = old_argv
    cfg = Config(**vars(args))
    cfg.read_yaml()
    assert cfg.se_model == "rwsamamba_unet" and cfg.model_type == "discriminative"
    assert cfg.model_configs["use_rwsa"] is False
    assert cfg.data["train"].get("max_duration_sec") is not None
    model = RWSAMambaSEModel(cfg).to(device)
    model.log = lambda *a, **k: None
    model.train()
    fs = 16000
    clean = torch.randn(1, 1, fs, device=device)
    batch = (clean, clean + 0.1 * torch.randn_like(clean),
             torch.tensor(fs, dtype=torch.int32), torch.tensor([fs], dtype=torch.int32))
    loss = model.training_step(batch)
    loss.backward()
    assert torch.isfinite(loss)
    print(f"T7 yaml -> RWSAMambaSEModel build + train step OK (loss={loss.item():.4f})")


def t8_state_dict_alignment(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    official = ckpt["generator"] if "generator" in ckpt else ckpt
    from baseuse.models.rwsa_se import RWSAMambaUNet_SE

    se = RWSAMambaUNet_SE(net_cfg={"use_rwsa": True})
    mine = se.net.state_dict()
    common = set(mine) & set(official)
    SKIP = (".time_mamba.", ".freq_mamba.", "shared_attention", "attention.", "lsigmoid")
    bad_official = [k for k in set(official) - set(mine) if not any(s in k for s in SKIP)]
    bad_mine = [k for k in set(mine) - set(official) if not any(s in k for s in SKIP)]
    assert not bad_official and not bad_mine, (bad_official[:5], bad_mine[:5])
    print(f"T8 state-dict alignment OK: {len(common)} common keys, "
          "only Mamba-internals / attention / lsigmoid differ (expected)")


def t9_upstream_equivalence(upstream_dir, device):
    """analyze/synthesize and phase_losses numerically match the upstream repo."""
    sys.path.insert(0, upstream_dir)
    from models.stfts import mag_phase_stft, mag_phase_istft  # upstream
    from baseuse.models.rwsa_se import RWSAMambaUNet_SE, phase_losses

    torch.manual_seed(0)
    se = RWSAMambaUNet_SE().to(device)
    wav = torch.randn(2, 16000, device=device)
    mag, pha, com = se.analyze(wav, 16000)
    mag0, pha0, com0 = mag_phase_stft(wav, 510, 120, 510, 0.3)
    for a, b in ((mag, mag0), (pha, pha0), (com, com0)):
        assert torch.allclose(a, b, atol=1e-5), (a - b).abs().max()
    wav_rec = se.synthesize(mag, pha, 16000, 16000)
    wav0 = mag_phase_istft(mag0, pha0, 510, 120, 510, 0.3)
    assert torch.allclose(wav_rec[:, : wav0.size(-1)], wav0, atol=1e-4)

    import numpy as np
    phase_r = torch.rand(2, 256, 64, device=device) * 4 * np.pi - 2 * np.pi
    phase_g = phase_r + 0.1 * torch.randn(2, 256, 64, device=device)
    ip, gd, iaf = phase_losses(phase_r, phase_g)

    def anti_wrapping(x):
        return torch.abs(x - torch.round(x / (2 * np.pi)) * 2 * np.pi)

    Fb, T = 256, 64
    gd_m = torch.triu(torch.ones(Fb, Fb, device=device), 1) - torch.triu(torch.ones(Fb, Fb, device=device), 2) - torch.eye(Fb, device=device)
    iaf_m = torch.triu(torch.ones(T, T, device=device), 1) - torch.triu(torch.ones(T, T, device=device), 2) - torch.eye(T, device=device)
    ref = (torch.mean(anti_wrapping(phase_r - phase_g)),
           torch.mean(anti_wrapping(torch.matmul(phase_r.transpose(1, 2), gd_m) - torch.matmul(phase_g.transpose(1, 2), gd_m))),
           torch.mean(anti_wrapping(torch.matmul(phase_r, iaf_m) - torch.matmul(phase_g, iaf_m))))
    for a, b in zip((ip, gd, iaf), ref):
        assert torch.allclose(a, b, atol=1e-6), (a, b)
    print("T9 analyze/synthesize + phase losses == upstream implementation OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", help="official RWSA_s.pth for state-dict alignment")
    ap.add_argument("--upstream", help="cloned RWSAMamba-UNet repo dir for numerical equivalence")
    ap.add_argument("--stub", help="CPU stub mamba_ssm dir (no real kernels)")
    args = ap.parse_args()

    device, stub = setup(args)
    print(f"device={device} (mamba stub: {stub})\n")

    t1_padding_helpers()
    t2_pipeline_all_fs(device)
    t3_loss_backward_optimizer(device)
    t4_lsigmoid_slicing(device)
    t5_rwsa_variant(device)
    t6_param_count(device)
    t7_yaml_build(device)
    if args.ckpt:
        t8_state_dict_alignment(args.ckpt)
    if args.upstream:
        t9_upstream_equivalence(args.upstream, device)
    print("\nALL SELF-TESTS PASSED")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Self-test for the SEMamba / BaseUSE integration.

GPU machine (real mamba-ssm kernels, runs everything):
    python scripts/semamba_selftest.py

Login node without CUDA (structure + shapes only, runs S1/S2/S7):
    python scripts/semamba_selftest.py            # auto-detects, skips S3-S6
"""

import argparse
import importlib.util
import os
import sys

import torch
import torch.nn as nn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

FS_LIST = [8000, 16000, 22050, 24000, 32000, 44100, 48000]


def setup(args):
    """Returns (device, cuda_ok). S3-S6 need CUDA; S1/S2/S7 run anywhere."""
    has_mamba = importlib.util.find_spec("mamba_ssm") is not None
    if args.cpu:
        return "cpu", False
    if has_mamba and torch.cuda.is_available():
        return "cuda", True
    print("NOTE: no CUDA mamba kernels here - running the CPU-only subset "
          "(S1/S2/S7). Run the full test on the GPU node.")
    return "cpu", False


class _IdentityTFMamba(nn.Module):
    """CPU stand-in for TFMambaBlock (its selective scan is CUDA-only).

    Shape-transparent on [B, C, T, F], which is all the conv-side shape
    contract of the flat backbone needs to be checked against.
    """

    def __init__(self, cfg, inchannels=None, **kwargs):
        super().__init__()
        self.identity = nn.Identity()

    def forward(self, x):
        return self.identity(x)


def t1_padding_helpers():
    """S1: flat-backbone padding policy."""
    from baseuse.models.rwsa_se import (
        _freq_bins, _pad_bins, _pad_bins_even, _pad_frames)

    for fs in FS_LIST:
        bins = _freq_bins(510, fs, 16000)
        even = _pad_bins_even(bins)
        assert even % 2 == 0 and even - bins <= 1, (fs, bins, even)
        # the U-Net policy is stricter (F % 8) - make sure we did not weaken it
        assert _pad_bins(bins) % 8 == 0
    assert _pad_bins_even(9) == 10 and _pad_bins_even(10) == 10
    # frames: flat backbone needs none, U-Net needs % 4
    assert _pad_frames(5) == 8 and _pad_frames(9) == 12
    print("S1 padding policy OK: even-F rounding (%s) + U-Net F%%8/T%%4 intact"
          % ", ".join(f"{fs}:{_freq_bins(510, fs, 16000)}" for fs in FS_LIST))


def t2_shapes_cpu():
    """S2: CPU shape round-trip of the flat backbone + full SFI wrapper.

    TFMambaBlock is stubbed out, so this validates the conv-side contract that
    actually decides correctness: DenseEncoder halves F, the decoders'
    PixelShuffle(2) must restore it EXACTLY (the mag branch multiplies the mask
    by the noisy magnitude), and T must round-trip for any frame count.
    """
    from baseuse.models.semamba import generator as gen_mod
    from baseuse.models.semamba.generator import SEMamba
    from baseuse.models.semamba_se import SEMamba_SE

    real = gen_mod.TFMambaBlock
    gen_mod.TFMambaBlock = _IdentityTFMamba
    try:
        cfg = {
            'model_cfg': {
                'hid_feature': 8, 'num_tfmamba': 2, 'd_state': 16, 'd_conv': 4,
                'expand': 4, 'norm_epsilon': 1e-5, 'beta': 2.0,
                'compress_factor': 0.3, 'input_channel': 2, 'output_channel': 1,
            },
            # table must be >= the largest even-padded F tested below (the
            # wrapper derives this from max_fs; 766 is the 48 kHz bin count)
            'stft_cfg': {'freq_bins_max': 768},
        }
        net = SEMamba(cfg).eval()

        # even F: exact round-trip; T: arbitrary values must all survive
        for f in (8, 64, 202, 766):
            for t in (1, 5, 17, 64, 133):
                mag = torch.randn(2, f, t).abs()
                pha = torch.randn(2, f, t)
                with torch.no_grad():
                    dm, dp, dc = net(mag, pha)
                assert dm.shape == (2, f, t), (f, t, dm.shape)
                assert dp.shape == (2, f, t), (f, t, dp.shape)
                assert dc.shape == (2, f, t, 2), (f, t, dc.shape)
                assert torch.allclose(dm, dc[..., 0].pow(2).add(dc[..., 1].pow(2)).sqrt(), atol=1e-5)
        print("S2a flat backbone shape round-trip OK (even F x any T)")

        # odd F must fail loudly: that is what makes the even padding load-bearing
        try:
            with torch.no_grad():
                net(torch.randn(1, 9, 4).abs(), torch.randn(1, 9, 4))
        except RuntimeError:
            print("S2a odd F correctly rejected by the mask multiply "
                  "(even padding is load-bearing)")
        else:
            raise AssertionError("odd F did not raise - mask/noisy_mag shape contract changed?")

        # full wrapper (espnet STFT on CPU) at every sampling rate
        se = SEMamba_SE(net_cfg={'hid_feature': 8, 'num_tfmamba': 2}).eval()
        assert se.pad_n_frames(13) == 13, "flat policy must not pad T"
        assert se.pad_freq_bins(201) == 202
        for fs in FS_LIST:
            for secs in (1, 2):
                wav = torch.randn(2, fs * secs)
                with torch.no_grad():
                    out = se.run(wav, fs)
                    est, _ = se(wav.unsqueeze(1), None, fs)
                assert out['est_mag'].size(1) == out['n_bins'], (fs, out['est_mag'].shape, out['n_bins'])
                assert est.shape == wav.shape, (fs, est.shape, wav.shape)
                assert torch.isfinite(est).all(), fs
        print("S2b full SFI wrapper OK at 7 sampling rates x {1s, 2s} (CPU, stubbed Mamba)")
    finally:
        gen_mod.TFMambaBlock = real


def t3_pipeline_all_fs(device, num_tfmamba=1):
    """S3: real Mamba kernels, full pipeline at every sampling rate."""
    from baseuse.models.semamba_se import SEMamba_SE
    torch.manual_seed(0)
    se = SEMamba_SE(net_cfg={'hid_feature': 16, 'num_tfmamba': num_tfmamba}).to(device)
    for fs in FS_LIST:
        noisy = torch.randn(2, fs, device=device)  # 1 second
        out = se.run(noisy, fs)
        assert out['est_wav'].size(0) == 2 and out['est_wav'].size(-1) >= fs
        assert out['est_mag'].shape[1] == out['n_bins']
        for k in ('est_wav', 'est_mag', 'est_pha', 'est_com'):
            assert torch.isfinite(out[k]).all(), (fs, k)
        est, _ = se(noisy.unsqueeze(1), None, fs)
        assert est.shape == noisy.shape and torch.isfinite(est).all()
    print("S3 full pipeline forward at 7 sampling rates OK (real mamba)")


def t4_loss_backward_optimizer(device):
    """S4: loss + backward + optimizer step, including a silent-tail variant."""
    from baseuse.config import Config
    from baseuse.models.semamba_se import SEMambaSEModel

    torch.manual_seed(0)
    cfg = Config()
    cfg.model_configs = {
        'hid_feature': 16, 'num_tfmamba': 1, 'd_state': 16, 'd_conv': 4,
        'expand': 4, 'norm_epsilon': 1e-5, 'beta': 2.0, 'compress_factor': 0.3,
        'grad_checkpoint': True, 'mamba_fp32': True,
        'loss_mag': 0.9, 'loss_pha': 0.3, 'loss_com': 0.1,
        'loss_time': 0.2, 'loss_con': 0.1,
    }
    cfg.learning_rate, cfg.weight_decay, cfg.lr_gamma = 5e-4, 0.01, 0.99
    model = SEMambaSEModel(cfg).to(device)
    model.log = lambda *a, **k: None
    opt, _ = model.configure_optimizers()
    assert isinstance(opt[0], torch.optim.AdamW)
    assert abs(opt[0].defaults['betas'][0] - 0.8) < 1e-9

    def silent_tail(x, frac=0.35):
        n = int(x.size(-1) * frac)
        return torch.cat([x[..., :-n], torch.zeros_like(x[..., -n:])], dim=-1) if n else x

    for i, fs in enumerate(FS_LIST):
        L = fs
        clean = torch.randn(2, 1, L, device=device)
        noisy = clean + 0.1 * torch.randn(2, 1, L, device=device)
        if i % 2 == 1:
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
    print("S4 loss + backward + optimizer step at 7 sampling rates (incl. silence) OK")


def t5_param_count(device):
    """S5: param budget vs the official checkpoint."""
    from baseuse.models.semamba_se import SEMamba_SE
    net64 = SEMamba_SE(net_cfg={'hid_feature': 64, 'num_tfmamba': 4}).net
    n = sum(p.numel() for p in net64.parameters())
    print(f"S5 flat SEMamba backbone (hid=64, 4 blocks): {n:,} params = {n / 1e6:.3f}M")
    lsig = net64.mask_decoder.lsigmoid.slope.numel()
    print(f"    LearnableSigmoid2D table = {lsig} rows (sized for max_fs=48 kHz;"
          f" upstream SEMamba is 16 kHz-only with 201 rows, so the table is"
          f" {lsig - 201} params larger here)")
    print("    NOT compared to ckpts/SEMamba_advanced.pth: its byte size (9127253)"
          " includes serialization overhead, so size/4 is not a param count.")
    print("    NOTE: param-matched vs the RWSA-UNet config is NOT exact - "
          "measured RWSA-UNet(hid=16, pure Mamba) = 1.888M, so hid=64 is ~1.2x "
          "larger. hid ~= 57 would match; 64 is kept because it is upstream's "
          "official setting.")
    assert 2.0e6 < n < 2.6e6, n
    small = SEMamba_SE(net_cfg={'hid_feature': 16, 'num_tfmamba': 1}).net
    assert sum(p.numel() for p in small.parameters()) < n


def t6_yaml_build(device):
    """S6: yaml -> model -> single training step."""
    from baseuse.config import Config, config_parser
    from baseuse.models.semamba_se import SEMambaSEModel

    yaml_path = os.path.join(REPO_ROOT, "conf/exp/semamba_2025_dynamic.yaml")
    old_argv = sys.argv
    sys.argv = ["x", "--config_file", yaml_path]
    args = config_parser()
    sys.argv = old_argv
    cfg = Config(**vars(args))
    cfg.read_yaml()
    assert cfg.se_model == 'semamba' and cfg.model_type == 'discriminative'
    assert cfg.model_configs['hid_feature'] == 64
    assert cfg.data['train']['sampling']['strategy'] == 'balanced'
    model = SEMambaSEModel(cfg).to(device)
    model.log = lambda *a, **k: None
    model.train()
    fs = 16000
    clean = torch.randn(1, 1, fs // 4, device=device)  # 0.25 s: keep it quick
    batch = (clean, clean + 0.1 * torch.randn_like(clean),
             torch.tensor(fs, dtype=torch.int32),
             torch.tensor([clean.size(-1)], dtype=torch.int32))
    loss = model.training_step(batch)
    loss.backward()
    assert torch.isfinite(loss)
    print(f"S6 yaml -> SEMambaSEModel build + train step OK (loss={loss.item():.4f})")


def t8_length_masking():
    """S8: length-masked losses are invariant to zero padding (CPU, no model).

    The property the `length_mask_loss` flag exists to guarantee: appending
    zero padding + excluding it via the mask must leave every term UNCHANGED,
    while the legacy unmasked term gets diluted toward zero by the padding.
    """
    from baseuse.models.rwsa_se import RWSAMambaSEModel as M, phase_losses

    torch.manual_seed(0)
    B, Fb, T0, K = 3, 256, 200, 300
    a = torch.randn(B, Fb, T0)
    b = a + 0.1 * torch.randn(B, Fb, T0)            # a decent estimate
    ones = torch.ones(B, 1, T0)

    # 1) mask of all ones must reproduce the legacy (unmasked) value exactly
    assert torch.isclose(M._masked_mse(a, b, ones), torch.mean((a - b) ** 2), atol=1e-6)
    la, lb = torch.randn(B, 4 * T0), a.new_zeros(B, 4 * T0)
    s_ones = torch.ones(B, 4 * T0)
    assert torch.isclose(M._masked_l1(la, lb, s_ones), (la - lb).abs().mean(), atol=1e-6)
    ip0, gd0, iaf0 = phase_losses(a, b)
    ip1, gd1, iaf1 = phase_losses(a, b, ones)
    for x, y in ((ip0, ip1), (gd0, gd1), (iaf0, iaf1)):
        assert torch.isclose(x, y, atol=1e-6), (x, y)
    print("S8a all-ones mask == legacy unmasked value (mag/time/phase) OK")

    # 2) padding invariance: pad T by K zeros (as collate does for short clips)
    pad_a = torch.nn.functional.pad(a, (0, K))
    pad_b = torch.nn.functional.pad(b, (0, K))
    pad_m = torch.cat([ones, torch.zeros(B, 1, K)], dim=2)

    plain_plain = torch.mean((a - b) ** 2).item()
    plain_pad = torch.mean((pad_a - pad_b) ** 2).item()
    masked_pad = M._masked_mse(pad_a, pad_b, pad_m).item()
    assert abs(masked_pad - plain_plain) < 1e-6, (masked_pad, plain_plain)
    assert plain_pad < plain_plain * 0.5, (plain_pad, plain_plain)   # diluted ~2.5x
    print("S8b mag MSE: padded-plain %.4f (diluted from %.4f) vs masked %.4f == unpadded OK"
          % (plain_pad, plain_plain, masked_pad))

    for name, fn in (("ip", 0), ("gd", 1), ("iaf", 2)):
        legacy = phase_losses(pad_a, pad_b)[fn].item()
        masked = phase_losses(pad_a, pad_b, pad_m)[fn].item()
        ref = phase_losses(a, b)[fn].item()
        assert abs(masked - ref) < 1e-5, (name, masked, ref)
        assert abs(legacy - ref) > 1e-5, (name, legacy, ref)  # legacy IS diluted
        print("S8b phase %-4s masked %.5f == unpadded %.5f (legacy padded %.5f)"
              % (name, masked, ref, legacy))

    # time-domain L1, same property
    n0, nk = 4 * T0, 4 * K
    la = torch.randn(B, n0)
    lb = la + 0.05 * torch.randn(B, n0)
    legacy = torch.nn.functional.l1_loss(
        torch.nn.functional.pad(la, (0, nk)), torch.nn.functional.pad(lb, (0, nk))).item()
    masked = M._masked_l1(torch.nn.functional.pad(la, (0, nk)),
                          torch.nn.functional.pad(lb, (0, nk)),
                          torch.cat([torch.ones(B, n0), torch.zeros(B, nk)], 1)).item()
    ref = (la - lb).abs().mean().item()
    assert abs(masked - ref) < 1e-6 and legacy < ref * 0.5
    print("S8b time L1: padded-plain %.4f (vs true %.4f) vs masked %.4f OK" % (legacy, ref, masked))

    # 3) complex term [B,F,T,2] (loss_com / loss_con): the denominator must use
    # F and the trailing 2, NOT T. F != T here on purpose - that shape mismatch
    # is exactly what made a size(-2) bug invisible when F == T.
    c_a = torch.randn(B, Fb, T0, 2)
    c_b = c_a + 0.1 * torch.randn(B, Fb, T0, 2)
    assert torch.isclose(M._masked_mse(c_a, c_b, ones), (c_a - c_b).pow(2).mean(), atol=1e-6)
    pad_ca = torch.cat([c_a, torch.zeros(B, Fb, K, 2)], dim=2)
    pad_cb = torch.cat([c_b, torch.zeros(B, Fb, K, 2)], dim=2)
    masked_c = M._masked_mse(pad_ca, pad_cb, pad_m).item()
    ref_c = (c_a - c_b).pow(2).mean().item()
    plain_c = (pad_ca - pad_cb).pow(2).mean().item()
    assert abs(masked_c - ref_c) < 1e-6, (masked_c, ref_c)
    assert plain_c < ref_c * 0.5, (plain_c, ref_c)
    print("S8b complex MSE [B,F,T,2]: padded-plain %.4f (diluted from %.4f) vs masked %.4f OK"
          % (plain_c, ref_c, masked_c))

    # 4) the mask builder itself: frame t is real iff t*hop < length
    lens = torch.tensor([1000, 300, 1000])
    sm, fm = M._valid_masks(lens, n_samples=1000, n_frames=10, hop=100,
                            device=torch.device("cpu"), dtype=torch.float32)
    assert sm.sum().item() == 1000 + 300 + 1000
    assert torch.equal(fm[0, 0], torch.tensor([1.] * 10))          # 1000/100 -> 10 frames
    assert fm[1].sum().item() == 3, fm[1]                          # 300/100 -> frames 0,1,2
    print("S8c _valid_masks OK: sample mask %d/%d ones, short clip keeps %d/10 frames"
          % (int(sm.sum().item()), sm.numel(), int(fm[1].sum().item())))
    print("S8 length-masked losses OK: padding-invariant, legacy path unchanged")


def t9_forward_step_integration(device):
    """S9: run forward_step END-TO-END with length_mask_loss on/off (CPU).

    The backbone is stubbed (real Mamba scans are CUDA-only) but everything
    that matters for the masking stays real: espnet STFT/iSTFT, collate-style
    zero padding, per-sample speech_length, the phase losses, and the SI-SNR
    metric. This executes code the helper-level S8 cannot reach - notably that
    the masks are built with the SAME T/L as the tensors they multiply.
    """
    from baseuse.config import Config
    from baseuse.models.semamba_se import SEMambaSEModel
    from baseuse.data.dataset import collate_fn
    from baseuse.models.rwsa_se import phase_losses

    torch.manual_seed(0)

    def build(mask_flag):
        cfg = Config()
        cfg.model_configs = {
            'hid_feature': 8, 'num_tfmamba': 1, 'd_state': 16, 'd_conv': 4,
            'expand': 4, 'norm_epsilon': 1e-5, 'beta': 2.0, 'compress_factor': 0.3,
            'length_mask_loss': mask_flag,
            'loss_mag': 0.9, 'loss_pha': 0.3, 'loss_com': 0.1, 'loss_time': 0.2,
            'loss_con': 0.1,
        }
        cfg.learning_rate, cfg.weight_decay, cfg.lr_gamma = 5e-4, 0.01, 0.99
        m = SEMambaSEModel(cfg)
        m.log = lambda *a, **k: None

        # stub only the Mamba backbone; keep the real STFT front/back-end and
        # the real mask*noisy_mag magnitude path. The learnable scale keeps a
        # gradient path so the masked denominators can be backward-tested.
        class _Stub(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(0.9))

            def forward(self, noisy_mag, noisy_pha):
                mag = noisy_mag * self.scale
                com = torch.stack((mag * torch.cos(noisy_pha),
                                   mag * torch.sin(noisy_pha)), dim=-1)
                return mag, noisy_pha, com

        m.se_model.net = _Stub()
        return m

    fs, n_long, n_short = 16000, 16000, 4000   # 1.0 s and 0.25 s clips

    def clip(n, seed):
        g = torch.Generator().manual_seed(seed)
        x = torch.randn(1, n, generator=g) * 0.05
        return x, x + 0.05 * torch.randn(1, n, generator=g)

    long_c, long_n = clip(n_long, 1)
    short_c, short_n = clip(n_short, 2)

    # (a) unequal lengths -> real zero padding; masked terms must be LARGER
    batch_pad = collate_fn([
        (long_c.numpy(), long_n.numpy(), fs, n_long),
        (short_c.numpy(), short_n.numpy(), fs, n_short),
    ])
    m_on, m_off = build(True), build(False)
    loss_on = m_on.forward_step(batch_pad)
    loss_off = m_off.forward_step(batch_pad)
    assert torch.isfinite(loss_on) and torch.isfinite(loss_off)
    assert loss_on > loss_off, (loss_on.item(), loss_off.item())
    # masked denominators must stay differentiable
    loss_on.backward()
    g = m_on.se_model.net.scale.grad
    assert g is not None and torch.isfinite(g) and g.abs() > 0, g
    print("S9a padded batch (1.0s + 0.25s): masked loss %.4f > unmasked %.4f "
          "(padding no longer dilutes); backward d(loss)/d(scale)=%.3e finite"
          % (loss_on.item(), loss_off.item(), g.item()))

    # (b) equal lengths -> no BATCH padding, but run() still pads the waveform
    # up to a hop multiple (16000 -> 16080), and the frame mask correctly
    # excludes the resulting all-padding tail frame (centre at t*hop == length
    # falls outside the real signal). So the two paths must agree up to ~1
    # frame out of n_frames, NOT exactly. A gross masking error (e.g. the wrong
    # denominator) would show up as tens of percent, not 0.05%.
    c2, n2 = clip(n_long, 3)
    batch_eq = collate_fn([
        (long_c.numpy(), long_n.numpy(), fs, n_long),
        (c2.numpy(), n2.numpy(), fs, n_long),
    ])
    l_on = m_on.forward_step(batch_eq).item()
    l_off = m_off.forward_step(batch_eq).item()
    rel = abs(l_on - l_off) / abs(l_off)
    # authoritative frame count from the real wrapper (no re-implemented STFT math)
    n_frames = m_on.se_model.run(torch.zeros(1, n_long), fs)['n_frames_padded']
    assert rel < 2.0 / n_frames + 1e-3, (l_on, l_off, rel)
    print("S9b equal-length batch: masked %.6f vs unmasked %.6f (rel %.4f%% ~ one "
          "excluded hop-alignment tail frame of %d)" % (l_on, l_off, rel * 100, n_frames))

    # (c) phase_losses default path still matches the upstream 2-arg signature
    a = torch.randn(2, 256, 64)
    b = a + 0.1 * torch.randn(2, 256, 64)
    assert tuple(phase_losses(a, b)) == tuple(phase_losses(a, b, None))
    print("S9c phase_losses(a, b) default f_mask=None is byte-identical (T9 safe)")
    print("S9 forward_step integration OK with length_mask_loss on and off")


def t7_rwsa_regression():
    """S7: the rwsa_se.py delegation refactor must not change RWSA behaviour."""
    from baseuse.models.rwsa_se import (
        RWSAMambaUNet_SE, _freq_bins, _pad_bins, _pad_frames)
    from baseuse.models.semamba_se import SEMamba_SE

    # module-level helpers still exist with unchanged semantics (selftest T1
    # imports them directly)
    for fs in FS_LIST:
        bins = _freq_bins(510, fs, 16000)
        assert _pad_bins(bins) % 8 == 0
        assert _pad_frames(5) == 8 and _pad_frames(8) == 8 and _pad_frames(9) == 12

    # default (U-Net) policy still delegates to the module-level helpers, and
    # SEMamba overrides only its own copies. object.__new__ avoids building the
    # backbone: these three methods never touch instance attributes.
    base = object.__new__(RWSAMambaUNet_SE)
    flat = object.__new__(SEMamba_SE)
    assert SEMamba_SE.__mro__[1] is RWSAMambaUNet_SE
    assert base.pad_freq_bins(201) == _pad_bins(201) == 208
    assert base.pad_n_frames(13) == _pad_frames(13) == 16
    assert flat.pad_freq_bins(201) == 202
    assert flat.pad_n_frames(13) == 13

    from baseuse.models.semamba.generator import SEMamba
    from baseuse.models.rwsamamba.generator import MambAttentionSEUNet
    assert SEMamba_SE.build_net is not RWSAMambaUNet_SE.build_net
    assert SEMamba is not MambAttentionSEUNet
    print("S7 RWSA regression OK: module-level padding helpers intact, "
          "U-Net policy unchanged, SEMamba overrides are isolated")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", action="store_true",
                    help="force the CPU-only subset (S1/S2/S7)")
    args = ap.parse_args()

    device, cuda_ok = setup(args)
    print(f"device={device} (CUDA subset: {cuda_ok})\n")

    t1_padding_helpers()
    t2_shapes_cpu()
    t5_param_count(device)   # construction is CPU-safe; only forward needs CUDA
    t8_length_masking()      # pure-tensor helpers, CPU-safe
    t9_forward_step_integration(device)  # stub backbone + real STFT/collate, CPU
    t7_rwsa_regression()
    if cuda_ok:
        t3_pipeline_all_fs(device)
        t4_loss_backward_optimizer(device)
        t6_yaml_build(device)
    else:
        print("\nS3/S4/S6 skipped (no CUDA). Run on the GPU node to complete.")
    print("\nALL RUNNABLE SELF-TESTS PASSED")


if __name__ == "__main__":
    main()

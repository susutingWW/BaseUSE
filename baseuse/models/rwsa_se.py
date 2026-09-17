"""RWSAMamba-UNet integration for BaseUSE.

Upstream: https://github.com/NikolaiKyhne/RWSAMamba-UNet (ICASSP 2026).
This module provides:

- RWSAMambaUNet_SE: SFI-STFT front/back-end wrapper around the vendored
  MambAttentionSEUNet backbone (baseuse/models/rwsamamba/). The espnet
  STFTEncoder/STFTDecoder reconfigure n_fft/win/hop per input sampling rate
  while keeping their duration fixed (reference: 16 kHz, n_fft=510,
  hop=120 -> 31.875 ms window / 7.5 ms hop / ~31.4 Hz bin spacing at every
  rate), so one set of weights serves 8k-48k. Default config is the
  pure-Mamba variant (`use_rwsa: false`, attention removed).

- RWSAMambaSEModel: LightningModule with the upstream generator loss and no
  GAN parts (no MetricDiscriminator / metric loss / PESQ-in-the-loop):
      L = 0.9 * MSE(mag) + 0.3 * (IP + GD + IAF phase) + 0.1 * 2 * MSE(com)
        + 0.2 * L1(time) + 0.1 * 2 * MSE(com vs re-STFT(com))
   weights from checkpoints/RWSA_MambaUNet_s.yaml (training_cfg.loss).

Both classes are also the base for the flat SEMamba backbone: see
baseuse/models/semamba_se.py, which overrides only the backbone + padding
policy (`build_net` / `pad_freq_bins` / `pad_n_frames` / `se_class`).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as L

from espnet2.enh.encoder.stft_encoder import STFTEncoder
from espnet2.enh.decoder.stft_decoder import STFTDecoder
from espnet2.enh.loss.criterions.time_domain import SISNRLoss

from baseuse.config import Config
from baseuse.models.rwsamamba.generator import MambAttentionSEUNet


# ---------------------------------------------------------------------------
# SFI helpers
# ---------------------------------------------------------------------------

def _freq_bins(n_fft, fs, base_fs):
    """Number of one-sided bins for the SFI-reconfigured STFT at `fs`."""
    return (n_fft * int(fs) // int(base_fs)) // 2 + 1


def _pad_bins(freq_bins):
    """Pad the frequency axis so the U-Net freq path stays integral.

    DenseEncoder halves the freq axis once (stride-2 conv) and Downsample
    PixelUnshuffle halves it twice more, so F/2 must be a multiple of 4
    (F % 8 == 0). The MagDecoder PixelShuffle then restores exactly F bins,
    which the elementwise mask multiplication in the backbone requires.
    """
    f = int(freq_bins)
    return f + (-f) % 8


def _pad_frames(n_frames):
    """Pad the frame axis so the U-Net time path (2x PixelUnshuffle) stays
    integral: T must be a multiple of 4."""
    t = int(n_frames)
    return t + (4 - t % 4) % 4


def _pad_bins_even(freq_bins):
    """Frequency padding for FLAT backbones (no U-Net): only an even axis is
    needed. DenseEncoder halves F once (stride-2 conv) and MagDecoder /
    PhaseDecoder restore it with PixelShuffle(2), so the encoders/decoders
    round-trip exactly - and the `mask * noisy_mag` elementwise product in the
    backbone requires that exact match. The frame axis is untouched: T doubles
    under PixelShuffle(2) and is halved again by the stride-(2,1) conv, which
    is exact for any T."""
    f = int(freq_bins)
    return f + f % 2


# ---------------------------------------------------------------------------
# Anti-wrapping phase losses (MP-SENet style), size taken from the tensors
# so they follow the current sampling rate / padded shapes.
# ---------------------------------------------------------------------------

_2PI = 2 * math.pi
EPS = 1e-8
_DIFF_MATRIX_CACHE = {}


def anti_wrapping_function(x):
    """Anti-wrapping function to adjust phase values within -pi..pi."""
    return torch.abs(x - torch.round(x / _2PI) * _2PI)


def _diff_matrix(size, ref, device, dtype):
    """First-difference matrix (triu(1) - triu(2) - eye), cached per size."""
    key = (size, str(device), dtype)
    m = _DIFF_MATRIX_CACHE.get(key)
    if m is None:
        eye = torch.eye(size, device=device, dtype=dtype)
        ones = torch.ones(size, size, device=device, dtype=dtype)
        m = torch.triu(ones, 1) - torch.triu(ones, 2) - eye
        _DIFF_MATRIX_CACHE[key] = m
    return m


def phase_losses(phase_r, phase_g, f_mask=None):
    """In-phase / gradient-delay / integrated-absolute-frequency losses.

    Args:
    - phase_r, phase_g (torch.Tensor): [B, F, T] reference/generated phase.
    - f_mask (torch.Tensor, optional): [B, 1, T] float 0/1 marking frames that
      belong to real audio rather than to the batch's zero padding. When given,
      each term is the mean over VALID elements only. With f_mask=None (the
      default) this is byte-identical to the upstream MP-SENet implementation,
      which is what rwsamamba_selftest.py T9 asserts against.

    Returns:
    - (ip_loss, gd_loss, iaf_loss)
    """
    dim_freq = phase_r.size(1)
    dim_time = phase_r.size(2)

    gd_matrix = _diff_matrix(dim_freq, phase_r, phase_r.device, phase_r.dtype)
    gd_r = torch.matmul(phase_r.transpose(1, 2), gd_matrix)
    gd_g = torch.matmul(phase_g.transpose(1, 2), gd_matrix)

    iaf_matrix = _diff_matrix(dim_time, phase_r, phase_r.device, phase_r.dtype)
    iaf_r = torch.matmul(phase_r, iaf_matrix)
    iaf_g = torch.matmul(phase_g, iaf_matrix)

    ip_map = anti_wrapping_function(phase_r - phase_g)     # [B, F, T]
    gd_map = anti_wrapping_function(gd_r - gd_g)           # [B, T, F] <- transposed
    iaf_map = anti_wrapping_function(iaf_r - iaf_g)        # [B, F, T]

    if f_mask is None:
        return ip_map.mean(), gd_map.mean(), iaf_map.mean()

    # gd_map is indexed [B, T, F], so its mask runs along dim 1
    valid = f_mask.sum().clamp_min(1.0) * dim_freq
    return ((ip_map * f_mask).sum() / valid,
            (gd_map * f_mask.transpose(1, 2)).sum() / valid,
            (iaf_map * f_mask).sum() / valid)


# ---------------------------------------------------------------------------
# SFI-STFT wrapper
# ---------------------------------------------------------------------------

class RWSAMambaUNet_SE(nn.Module):
    """espnet SFI-STFT front-end + MambAttentionSEUNet + iSTFT back-end.

    forward(speech [B, L] or [B, 1, L], ilens, fs) -> (enhanced [B, L], None),
    mirroring BSRNN_SE's interface.
    """

    def __init__(self,
                 base_fs=16000,
                 n_fft=510,
                 hop_length=120,
                 win_length=510,
                 max_fs=48000,
                 compress_factor=0.3,
                 net_cfg=None):
        super().__init__()
        self.base_fs = int(base_fs)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length) if win_length else int(n_fft)
        self.max_fs = int(max_fs)
        self.compress_factor = float(compress_factor)

        self.encoder = STFTEncoder(
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            use_builtin_complex=True,
            default_fs=self.base_fs,
        )
        self.decoder = STFTDecoder(
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            default_fs=self.base_fs,
        )

        # Frequency table for LearnableSigmoid2D: the net emits
        # 2 * ceil(padded_bins / 2) freq bins at the max sampling rate.
        padded_max = self.pad_freq_bins(_freq_bins(self.n_fft, self.max_fs, self.base_fs))
        freq_bins_max = padded_max + (padded_max % 2)

        mc = dict(net_cfg or {})
        mc.setdefault('hid_feature', 16)
        mc.setdefault('compress_factor', self.compress_factor)
        mc.setdefault('num_tfmamba', 4)
        mc.setdefault('d_state', 16)
        mc.setdefault('d_conv', 4)
        mc.setdefault('expand', 4)
        mc.setdefault('norm_epsilon', 1e-5)
        mc.setdefault('beta', 2.0)
        mc.setdefault('use_rwsa', False)
        mc.setdefault('grad_checkpoint', False)  # recompute TFMamba stacks in backward (train-only)
        mc['input_channel'] = 2
        mc['output_channel'] = 1
        cfg = {'model_cfg': mc, 'stft_cfg': {'freq_bins_max': freq_bins_max}}
        self.net = self.build_net(cfg)

    # -- backbone / padding policy (overridden by flat-backbone subclasses) ---

    def build_net(self, cfg):
        """Instantiate the TF-grid backbone. cfg = {'model_cfg', 'stft_cfg'}."""
        return MambAttentionSEUNet(cfg)

    def pad_freq_bins(self, freq_bins):
        """Frequency-axis padding required by this backbone (U-Net: % 8)."""
        return _pad_bins(freq_bins)

    def pad_n_frames(self, n_frames):
        """Frame-axis padding required by this backbone (U-Net: % 4)."""
        return _pad_frames(n_frames)

    # -- analysis / synthesis ------------------------------------------------

    def stft_params(self, fs):
        """SFI-reconfigured (n_fft, hop) at `fs` (win_length == n_fft here)."""
        fs = int(fs)
        return (self.n_fft * fs // self.base_fs,
                self.hop_length * fs // self.base_fs)

    def analyze(self, wav, fs, addeps=False):
        """wav [B, L] -> (mag, pha, com) with shapes [B, F, T] / [B, F, T, 2].

        Magnitude is power-compressed (compress_factor); matches the upstream
        mag_phase_stft semantics (hann window, center=True, reflect padding).
        With addeps=True a small eps is added inside sqrt/atan2 (upstream's
        addeps variant): required on gradient paths, because real audio has
        exactly-zero STFT bins (silence / zero padding) where the backward of
        abs().pow(c) is 0.3*0^(-0.7)=inf and of angle() is 0/0=NaN.
        """
        ilens = torch.full((wav.size(0),), wav.size(-1),
                           dtype=torch.long, device=wav.device)
        spec, _ = self.encoder(wav.float(), ilens, fs=int(fs))  # [B, T, F]
        spec = spec.transpose(1, 2)                              # [B, F, T]
        if addeps:
            eps = EPS
            real, imag = spec.real, spec.imag
            mag = torch.sqrt(real.pow(2) + imag.pow(2) + eps).pow(self.compress_factor)
            pha = torch.atan2(imag + eps, real + eps)
        else:
            mag = spec.abs().pow(self.compress_factor)
            pha = torch.angle(spec)
        com = torch.stack((mag * torch.cos(pha), mag * torch.sin(pha)), dim=-1)
        return mag, pha, com

    def synthesize(self, mag, pha, out_length, fs):
        """mag/pha [B, F, T] (compressed) -> wav [B, out_length]."""
        mag = mag.pow(1.0 / self.compress_factor)
        com = torch.complex(mag * torch.cos(pha), mag * torch.sin(pha))  # [B, F, T]
        ilens = torch.full((com.size(0),), int(out_length),
                           dtype=torch.long, device=com.device)
        wav, _ = self.decoder(com.transpose(1, 2), ilens, fs=int(fs))
        return wav

    # -- full pipeline ---------------------------------------------------

    def run(self, wav, fs):
        """Enhancement pipeline on wav [B, L].

        Returns a dict with the model outputs plus the shapes needed to build
        matching training targets (see RWSAMambaSEModel.forward_step).
        """
        fs = int(fs)
        _, hop = self.stft_params(fs)
        length = wav.size(-1)
        padded_length = -(-length // hop) * hop  # ceil to a hop multiple
        wav_p = F.pad(wav.float(), (0, padded_length - length))

        noisy_mag, noisy_pha, _ = self.analyze(wav_p, fs)        # [B, F, T]
        n_bins, n_frames = noisy_mag.size(1), noisy_mag.size(2)
        bins_pad = self.pad_freq_bins(n_bins)
        frames_pad = self.pad_n_frames(n_frames)
        noisy_mag = F.pad(noisy_mag, (0, frames_pad - n_frames, 0, bins_pad - n_bins))
        noisy_pha = F.pad(noisy_pha, (0, frames_pad - n_frames, 0, bins_pad - n_bins))

        mag_g, pha_g, com_g = self.net(noisy_mag, noisy_pha)     # [B, >=F, T']

        est_mag = mag_g[:, :n_bins]
        est_pha = pha_g[:, :n_bins]
        est_com = com_g[:, :n_bins]
        est_wav = self.synthesize(est_mag, est_pha, padded_length, fs)

        return {
            'est_wav': est_wav,            # [B, padded_length]
            'est_mag': est_mag,            # [B, F, T']
            'est_pha': est_pha,            # [B, F, T']
            'est_com': est_com,            # [B, F, T', 2]
            'n_bins': n_bins,              # true freq bins at this fs
            'n_frames': n_frames,          # true frame count
            'n_frames_padded': frames_pad, # frame count fed to the net
            'padded_length': padded_length,
        }

    def forward(self, speech, ilens, fs):
        wav = speech.view(speech.size(0), -1).float()
        out = self.run(wav, fs)
        est = out['est_wav']
        return est[:, :wav.size(-1)], None


# ---------------------------------------------------------------------------
# Lightning module (no GAN: generator losses only)
# ---------------------------------------------------------------------------

class RWSAMambaSEModel(L.LightningModule):
    """LightningModule: generator losses only (no GAN).

    `se_class` selects the SFI-STFT wrapper; subclasses (e.g. SEMambaSEModel)
    override ONLY that, inheriting the losses, optimizer, scheduler and NaN
    guards verbatim.
    """

    se_class = RWSAMambaUNet_SE

    def __init__(self, cfg: Config):
        super().__init__()

        self.save_hyperparameters()
        self.cfg = cfg
        mc = dict(cfg.model_configs or {})

        self.se_model = self.se_class(
            base_fs=mc.get('base_fs', 16000),
            n_fft=mc.get('n_fft', 510),
            hop_length=mc.get('hop_length', 120),
            win_length=mc.get('win_length', 510),
            max_fs=mc.get('max_fs', 48000),
            compress_factor=mc.get('compress_factor', 0.3),
            net_cfg=mc,
        )

        # upstream training_cfg.loss weights (metric/GAN term dropped)
        self.loss_weights = {
            'magnitude': mc.get('loss_mag', 0.9),
            'phase': mc.get('loss_pha', 0.3),
            'complex': mc.get('loss_com', 0.1),
            'time': mc.get('loss_time', 0.2),
            'consistancy': mc.get('loss_con', 0.1),
        }
        self.sisnr_loss = SISNRLoss()

        # Length-aware losses (EXPERIMENTAL, default off): collate pads every
        # batch to its longest member, and the original recipe computes all five
        # terms over that padded length, so short clips get their losses diluted
        # by zeros (a 0.5 s clip inside a 15 s val batch is ~96% padding). With
        # this flag on, every term is averaged over VALID samples/frames only,
        # and the SI-SNR metric is trimmed to the real length.
        #
        # Off by default because it CHANGES THE TRAINING OBJECTIVE (loss_time
        # grows, so the effective weight of the time term rises) and therefore
        # breaks metric comparability with runs trained without it. Kept as a
        # flag so one code base can reproduce both exactly.
        self.length_mask_loss = bool(mc.get('length_mask_loss', False))

    # -- length masking helpers ---------------------------------------------

    @staticmethod
    def _valid_masks(lengths, n_samples, n_frames, hop, device, dtype):
        """(sample_mask [B, n_samples], frame_mask [B, 1, n_frames]) of 0/1."""
        lengths = lengths.view(-1, 1).to(device)
        s_idx = torch.arange(n_samples, device=device).unsqueeze(0)
        sample_mask = (s_idx < lengths).to(dtype)
        # espnet's STFT is centered: frame t is real audio iff its centre
        # (t * hop) falls inside the unpadded signal.
        f_idx = torch.arange(n_frames, device=device).unsqueeze(0) * hop
        frame_mask = (f_idx < lengths).to(dtype).unsqueeze(1)
        return sample_mask, frame_mask

    @staticmethod
    def _masked_mse(a, b, frame_mask):
        """MSE over valid frames only. a/b: [B,F,T] or [B,F,T,2]."""
        err = (a - b).pow(2)
        if err.dim() == 4:                       # complex term: [B, F, T, 2]
            m = frame_mask.unsqueeze(-1)         # broadcast over F and re/im
            per_frame = err.size(-3)             # F  (NOT -2, which is T)
            extra = err.size(-1)                 # 2
        else:                                    # [B, F, T]
            m = frame_mask
            per_frame = err.size(-2)             # F
            extra = 1
        valid = frame_mask.sum().clamp_min(1.0) * per_frame * extra
        return (err * m).sum() / valid

    @staticmethod
    def _masked_l1(a, b, sample_mask):
        """L1 over valid samples only. a/b: [B, L]."""
        return ((a - b).abs() * sample_mask).sum() / sample_mask.sum().clamp_min(1.0)

    def forward_step(self, batch, stage='train'):

        clean_speech, noisy_speech, fs, speech_length = batch
        batch_size = len(clean_speech)
        clean = clean_speech.view(clean_speech.size(0), -1).float()
        noisy = noisy_speech.view(noisy_speech.size(0), -1).float()
        fs = int(fs)

        out = self.se_model.run(noisy, fs)

        # targets: same SFI analysis on clean (padded identically)
        clean_p = F.pad(clean, (0, out['padded_length'] - clean.size(-1)))
        clean_mag, clean_pha, clean_com = self.se_model.analyze(clean_p, fs)
        pad_t = (0, out['n_frames_padded'] - out['n_frames'])
        clean_mag = F.pad(clean_mag, pad_t)
        clean_pha = F.pad(clean_pha, pad_t)
        clean_com = F.pad(clean_com, (0, 0, 0, pad_t[1]))

        # L2 Magnitude Loss
        if self.length_mask_loss:
            _, hop = self.se_model.stft_params(fs)
            sample_mask, frame_mask = self._valid_masks(
                speech_length, out['padded_length'], out['n_frames_padded'],
                hop, clean.device, clean.dtype)
        else:
            sample_mask = frame_mask = None

        if frame_mask is None:
            loss_mag = F.mse_loss(clean_mag, out['est_mag'])
        else:
            loss_mag = self._masked_mse(clean_mag, out['est_mag'], frame_mask)
        # Anti-wrapping Phase Loss (f_mask=None keeps the upstream definition)
        loss_ip, loss_gd, loss_iaf = phase_losses(clean_pha, out['est_pha'], frame_mask)
        loss_pha = loss_ip + loss_gd + loss_iaf
        # L2 Complex Loss
        loss_com = (F.mse_loss(clean_com, out['est_com']) if frame_mask is None
                    else self._masked_mse(clean_com, out['est_com'], frame_mask)) * 2
        # Time Loss
        loss_time = (F.l1_loss(clean_p, out['est_wav']) if sample_mask is None
                     else self._masked_l1(clean_p, out['est_wav'], sample_mask))
        # Consistancy Loss (addeps=True: est_wav has exactly-zero STFT bins at
        # silence/padding, where the eps-free backward is inf/NaN)
        _, _, rec_com = self.se_model.analyze(out['est_wav'], fs, addeps=True)
        rec_com = F.pad(rec_com, (0, 0, 0, pad_t[1]))
        loss_con = (F.mse_loss(out['est_com'], rec_com) if frame_mask is None
                    else self._masked_mse(out['est_com'], rec_com, frame_mask)) * 2

        loss = (
            loss_mag * self.loss_weights['magnitude'] +
            loss_pha * self.loss_weights['phase'] +
            loss_com * self.loss_weights['complex'] +
            loss_time * self.loss_weights['time'] +
            loss_con * self.loss_weights['consistancy']
        )

        if torch.isnan(loss):
            print('NaN in loss has been detected, skip')
            # finite zero loss that still carries a graph (backward-able);
            # est_wav itself may contain NaN, hence nan_to_num
            return torch.nan_to_num(out['est_wav']).sum() * 0.0

        with torch.no_grad():
            est = out['est_wav'][:, :clean.size(-1)]
            if self.length_mask_loss:
                # Score each clip on its own valid prefix: over the padded
                # length a short clip is mostly zeros, which flattens SI-SNR
                # (measured +12.0 dB for a 4%-real clip whose true in-band value
                # was ~15-20 dB). B is small and this is a metric, so the per
                # sample loop is cheap.
                lens = speech_length.view(-1).tolist()
                per_utt = torch.cat([
                    -self.sisnr_loss(clean[i:i + 1, :max(int(n), 1)],
                                     est[i:i + 1, :max(int(n), 1)])
                    for i, n in enumerate(lens)
                ])
            else:
                per_utt = -self.sisnr_loss(clean, est)          # [B], dB
            # SI-SNR can be *undefined* rather than merely bad: URGENT's
            # simulation_validation holds ~16/2168 clips whose NOISY track is
            # exactly all-zero (an extreme packet-loss style degradation). Since
            # est_mag = mask * noisy_mag and the mask is strictly positive, a
            # silent input yields an exactly-zero output, so fast_bss_eval's
            # coherence is 0 -> loss +inf -> sisnr -inf. One such sample poisons
            # the batch mean and hence the epoch mean (and would freeze the
            # monitor='val_sisnr' checkpoint on the very first, untrained model).
            # Score only the well-posed samples; genuine divergence still shows
            # up as a non-finite loss, which the NaN guards above already catch.
            good = torch.isfinite(per_utt)
            if bool(good.all()):
                sisnr = per_utt.mean()
            elif bool(good.any()):
                sisnr = per_utt[good].mean()
            else:
                # whole batch silent/undefined: report 0 dB rather than -inf so
                # the metric stays comparable and the checkpoint monitor lives
                sisnr = per_utt.new_zeros(())

        self.log(f'{stage}_loss', loss.detach().item(),
                 on_step=True, prog_bar=True, batch_size=batch_size)
        self.log(f'{stage}_sisnr', sisnr.detach().item(),
                 on_step=True, prog_bar=True, batch_size=batch_size)
        self.log(f'{stage}_sisnr_{fs}', sisnr.detach().item(),
                 on_step=True, batch_size=batch_size)
        self.log(f'{stage}_mag_loss', loss_mag.detach().item(),
                 on_step=True, batch_size=batch_size)
        self.log(f'{stage}_pha_loss', loss_pha.detach().item(),
                 on_step=True, batch_size=batch_size)
        self.log(f'{stage}_com_loss', loss_com.detach().item(),
                 on_step=True, batch_size=batch_size)
        self.log(f'{stage}_time_loss', loss_time.detach().item(),
                 on_step=True, batch_size=batch_size)
        self.log(f'{stage}_con_loss', loss_con.detach().item(),
                 on_step=True, batch_size=batch_size)

        return loss

    def training_step(self, batch):
        return self.forward_step(batch)

    def validation_step(self, batch):
        loss = self.forward_step(batch, stage='val')
        return {'loss': loss.detach()}

    def on_before_optimizer_step(self, optimizer) -> None:
        """Runs after backward, before optimizer.step(): this is where NaN
        grads must be caught so they are never applied to the weights.
        (The optimizer_step override used by SEModel inspects stale grads
        from the previous step - one step too late - so weights could be
        poisoned before the guard ever fires.)"""
        bad = [n for n, p in self.named_parameters()
               if p.grad is not None and torch.isnan(p.grad).any()]
        if bad:
            shown = ', '.join(bad[:10]) + (', ...' if len(bad) > 10 else '')
            print(f'NaN in grad has been detected, reset grad to zero ({len(bad)} params: {shown})')
            optimizer.zero_grad(set_to_none=True)

    def configure_optimizers(self):
        # upstream recipe: AdamW(lr=5e-4, betas=(0.8, 0.99)) + 0.99/epoch decay
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.learning_rate,
            betas=(0.8, 0.99),
            weight_decay=self.cfg.weight_decay,
        )
        # With lr_decay_steps > 0 decay per optimizer step instead of per
        # epoch (epoch length now varies with data.train.sampling), reaching
        # lr_decay_factor x lr at lr_decay_steps.
        total_steps = int(getattr(self.cfg, 'lr_decay_steps', 0) or 0)
        if total_steps > 0:
            end_factor = float(getattr(self.cfg, 'lr_decay_factor', 0.1))
            gamma = end_factor ** (1.0 / total_steps)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
            return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=self.cfg.lr_gamma)
        return [optimizer], [scheduler]

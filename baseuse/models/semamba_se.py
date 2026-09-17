"""SEMamba integration for BaseUSE.

Upstream: https://github.com/roychao19477/semamba (Chao et al., "An
Investigation of Incorporating Mamba for Speech Enhancement", IEEE SLT 2024;
URGENT 2024 challenge 4th place).

This module provides:

- SEMamba_SE: the same SFI-STFT wrapper as RWSAMambaUNet_SE (espnet
  STFTEncoder/Decoder reconfiguring n_fft/win/hop per input sampling rate with
  fixed duration), but around the FLAT SEMamba backbone: DenseEncoder ->
  num_tfmamba x TFMambaBlock -> (MagDecoder, PhaseDecoder). No U-Net, no patch
  embedding, no MB-deform, no mag/pha refinement stacks, and both decoders read
  the same backbone output.

  Consequences for the padding policy (overridden here):
    * frequency axis only needs to be EVEN (not % 8), because DenseEncoder
      halves F once and the decoders' PixelShuffle(2) doubles it back - and the
      magnitude branch multiplies the mask by the noisy magnitude elementwise,
      so the round-trip must be exact;
    * the frame axis needs NO padding: PixelShuffle(2) doubles T and the
      following stride-(2,1) conv halves it exactly for any T (the U-Net needs
      % 4 for its two PixelUnshuffle levels).

- SEMambaSEModel: inherits RWSAMambaSEModel verbatim (losses, AdamW(5e-4,
  betas (0.8, 0.99)), per-step exponential decay, NaN guards) and overrides
  only `se_class`. SEMamba's generator loss weights are identical to the ones
  already configured (mag 0.9 / phase 0.3 / complex 0.1 / time 0.2 /
  consistency 0.1), so there is nothing to re-tune.

Deliberately NOT ported from upstream:
  * MetricDiscriminator + PESQ-in-the-loop (training_cfg.loss.metric = 0.05,
    computed per batch with 30 joblib processes). wb-PESQ is defined for 16 kHz
    only and is not comparable across the SFI sampling rates this repo trains
    on, and the host-side scoring would dominate a loop that is already
    model-bound.
  * PCS400 perceptual-contrast-stretching post-processing (inference-time
    only; adds a dependency and changes the evaluation target).
  * models/loss.py::compute_stft, which carries two bugs inherited from
    MP-SENet: `mag = sqrt(real^2 * imag^2 + eps)` (should be `+`, not `*`) and
    `pha = atan2(real + eps, imag + eps)` (arguments swapped). rwsa_se.analyze()
    is the correct implementation and additionally has the `addeps` guard for
    exactly-zero bins, which prevents the consistency-loss backward from
    producing inf/NaN on silence and padding.
"""

from baseuse.models.rwsa_se import (
    RWSAMambaSEModel,
    RWSAMambaUNet_SE,
    _pad_bins_even,
)
from baseuse.models.semamba.generator import SEMamba


class SEMamba_SE(RWSAMambaUNet_SE):
    """SFI-STFT front/back-end + flat SEMamba backbone."""

    def build_net(self, cfg):
        return SEMamba(cfg)

    def pad_freq_bins(self, freq_bins):
        return _pad_bins_even(freq_bins)

    def pad_n_frames(self, n_frames):
        # flat backbone: T round-trips exactly for any frame count
        return int(n_frames)


class SEMambaSEModel(RWSAMambaSEModel):
    """SEMamba LightningModule: same losses / optimizer / guards, new backbone."""

    se_class = SEMamba_SE

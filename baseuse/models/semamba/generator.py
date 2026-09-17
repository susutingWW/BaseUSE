"""Flat SEMamba backbone (Chao et al., "An Investigation of Incorporating Mamba
for Speech Enhancement", IEEE SLT 2024).

Topology vendored from https://github.com/roychao19477/semamba
=models/generator.py::SEMamba: DenseEncoder -> num_tfmamba x TFMambaBlock ->
(MagDecoder, PhaseDecoder). Unlike MambAttentionSEUNet (RWSAMamba-UNet) there is
no U-Net, no patch embedding, no MB-deform and no separate mag/pha refinement
stacks: both decoders read the same backbone output, so the only shape
constraint is an even frequency axis (see _pad_bins_even in semamba_se.py).

The building blocks are REUSED from baseuse/models/rwsamamba/, which is already
a line-for-line port of SEMamba's models/mamba_block.py and codec_module.py.
Two differences from upstream, both inherited from that port and kept on
purpose:
  - TFMambaBlock takes `inchannels` explicitly (the U-Net needed per-level dims;
    here it is always model_cfg['hid_feature']).
  - MambaBlock carries the `mamba_fp32` island that keeps the selective scan in
    fp32 under bf16 autocast (NaN guard) and honours `grad_checkpoint`.

Upstream's STFT/loss helpers are NOT used: models/loss.py::compute_stft has two
bugs inherited from MP-SENet (`mag = sqrt(real^2 * imag^2 + eps)` instead of
`+`, and `atan2(real + eps, imag + eps)` with swapped arguments). The wrapper
uses rwsa_se.analyze()/synthesize(), which are correct and have the `addeps`
guard for exactly-zero bins.
"""

import torch
import torch.nn as nn
from einops import rearrange
from torch.utils.checkpoint import checkpoint

from ..rwsamamba.codec_module import DenseEncoder, MagDecoder, PhaseDecoder
from ..rwsamamba.mambattention_block import TFMambaBlock


class SEMamba(nn.Module):
    """Dense encoder + flat TFMamba stack + magnitude/phase decoders.

    I/O convention is identical to MambAttentionSEUNet, so it drops straight
    into RWSAMambaUNet_SE: two [B, F, T] inputs (compressed magnitude and
    phase), and (denoised_mag, denoised_pha, denoised_com) outputs.

    Args:
    - cfg: dict with 'model_cfg' and 'stft_cfg' sections (see semamba_se.py).
    """

    def __init__(self, cfg):
        super(SEMamba, self).__init__()
        self.cfg = cfg
        self.num_tscblocks = cfg['model_cfg'].get('num_tfmamba') or 4
        self.grad_checkpoint = bool(cfg['model_cfg'].get('grad_checkpoint', False))
        hid = cfg['model_cfg']['hid_feature']

        self.dense_encoder = DenseEncoder(cfg)

        self.TSMamba = nn.ModuleList([
            TFMambaBlock(cfg, hid) for _ in range(self.num_tscblocks)
        ])

        self.mask_decoder = MagDecoder(cfg)
        self.phase_decoder = PhaseDecoder(cfg)

    def _run_blocks(self, x):
        """Run the TFMamba stack, optionally as one checkpoint segment.

        The whole stack is a single segment (unlike the U-Net, which checkpoints
        per stage): it is the only large activation holder in this backbone.
        """
        def seg(inp):
            for block in self.TSMamba:
                inp = block(inp)
            return inp

        if self.grad_checkpoint and self.training and x.requires_grad:
            return checkpoint(seg, x, use_reentrant=False)
        return seg(x)

    def forward(self, noisy_mag, noisy_pha):
        """
        Args:
        - noisy_mag (torch.Tensor): Compressed noisy magnitude [B, F, T].
        - noisy_pha (torch.Tensor): Noisy phase [B, F, T].

        Returns:
        - denoised_mag (torch.Tensor): [B, F_out, T]
        - denoised_pha (torch.Tensor): [B, F_out, T]
        - denoised_com (torch.Tensor): [B, F_out, T, 2]

        F must be EVEN: DenseEncoder halves the frequency axis (ceil(F/2)) and
        the decoders' PixelShuffle(2) doubles it back, and the magnitude branch
        multiplies the predicted mask by `noisy_mag` elementwise - an odd F
        would give F_out = F + 1 and break that product. The wrapper pads with
        _pad_bins_even() and crops the extra bin afterwards.
        """
        noisy_mag = rearrange(noisy_mag, 'b f t -> b t f').unsqueeze(1)  # [B,1,T,F]
        noisy_pha = rearrange(noisy_pha, 'b f t -> b t f').unsqueeze(1)  # [B,1,T,F]

        x = torch.cat((noisy_mag, noisy_pha), dim=1)                     # [B,2,T,F]

        x = self.dense_encoder(x)        # [B, hid, T, F/2]
        x = self._run_blocks(x)          # same shape

        # mask the noisy magnitude (upstream behaviour); phase is predicted
        denoised_mag = rearrange(
            self.mask_decoder(x) * noisy_mag, 'b c t f -> b f t c').squeeze(-1)
        denoised_pha = rearrange(
            self.phase_decoder(x), 'b c t f -> b f t c').squeeze(-1)

        denoised_com = torch.stack(
            (denoised_mag * torch.cos(denoised_pha),
             denoised_mag * torch.sin(denoised_pha)), dim=-1)

        return denoised_mag, denoised_pha, denoised_com

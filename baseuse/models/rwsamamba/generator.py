# Vendored from https://github.com/NikolaiKyhne/RWSAMamba-UNet (models/generator.py)
# Modification: `use_rwsa` switch (cfg['model_cfg']['use_rwsa']).
# - use_rwsa=False (default here): every U-Net stage uses TFMambaBlock
#   (pure Mamba, no attention). This removes the O(T^2) attention memory so
#   long segments / arbitrary sampling rates fit, at the cost of dropping the
#   paper's resolution-wise shared attention (RWSA) contribution.
# - use_rwsa=True: original RWSAMamba-UNet behavior (shared attention on
#   levels 1/2, private attention in the middle level).
# Modification: `grad_checkpoint` (cfg['model_cfg']['grad_checkpoint']): wrap
# the per-stage TFMamba block sequences and the mag/pha refinement stacks in
# torch.utils.checkpoint when training. Backward recomputes them instead of
# storing their activations (~2-3x activation memory saving, ~20-30% slower).
# No effect in eval mode.

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from einops import rearrange
from .mambattention_block import MambAttentionBlock, TFMambaBlock, AttentionModule
from .codec_module import DenseEncoder, MagDecoder, PhaseDecoder
from torchvision.ops.deform_conv import DeformConv2d
import math

class DWConv2d_BN(nn.Module):

    def __init__(
            self,
            in_ch,
            out_ch,
            kernel_size=1,
            stride=1,
            norm_layer=nn.BatchNorm2d,
            act_layer=nn.Hardswish,
            bn_weight_init=1,
            offset_clamp=(-1, 1)
    ):
        super().__init__()

        self.offset_clamp = offset_clamp
        self.offset_generator = nn.Sequential(nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=3,
                                                        stride=1, padding=1, bias=False, groups=in_ch),
                                              nn.Conv2d(in_channels=in_ch, out_channels=18,
                                                        kernel_size=1,
                                                        stride=1, padding=0, bias=False)
                                              )
        self.dcn = DeformConv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            groups=in_ch
        )
        self.pwconv = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)
        self.act = act_layer() if act_layer is not None else nn.Identity()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        offset = self.offset_generator(x)

        if self.offset_clamp:
            offset = torch.clamp(offset, min=self.offset_clamp[0], max=self.offset_clamp[1])
        x = self.dcn(x, offset)

        x = self.pwconv(x)
        x = self.act(x)
        return x


class MB_Deform_Embedding(nn.Module):

    def __init__(self,
                 in_chans=3,
                 embed_dim=768,
                 patch_size=16,
                 stride=1,
                 act_layer=nn.Hardswish,
                 offset_clamp=(-1, 1)):
        super().__init__()

        self.patch_conv = DWConv2d_BN(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            act_layer=act_layer,
            offset_clamp=offset_clamp
        )

    def forward(self, x):
        """foward function"""
        x = self.patch_conv(x)

        return x


class Patch_Embed_stage(nn.Module):
    """Depthwise Convolutional Patch Embedding stage comprised of
    `DWCPatchEmbed` layers."""

    def __init__(self, in_chans, embed_dim, isPool=False, offset_clamp=(-1, 1)):
        super(Patch_Embed_stage, self).__init__()

        self.patch_embeds = MB_Deform_Embedding(
                in_chans=in_chans,
                embed_dim=embed_dim,
                patch_size=3,
                stride=1,
                offset_clamp=offset_clamp)

    def forward(self, x):
        """foward function"""

        att_inputs = self.patch_embeds(x)

        return att_inputs

#####################################
class Downsample(nn.Module):
    def __init__(self, input_feat, out_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(
            # dw
            nn.Conv2d(input_feat, input_feat, kernel_size=3, stride=1, padding=1, groups=input_feat, bias=False),
            # pw-linear
            nn.Conv2d(input_feat, out_feat // 4, 1, 1, 0, bias=False),
            nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, input_feat, out_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(
            # dw
            nn.Conv2d(input_feat, input_feat, kernel_size=3, stride=1, padding=1, groups=input_feat, bias=False),
            # pw-linear
            nn.Conv2d(input_feat, out_feat * 4, 1, 1, 0, bias=False),
            nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


class MambAttentionSEUNet(nn.Module):
    """
    RWSAMamba-UNet backbone. Operates on [B, F, T] magnitude/phase inputs and
    is agnostic to (F, T); the surrounding wrapper handles SFI-STFT, padding
    and cropping. With use_rwsa=False this is the pure-Mamba ablation variant
    of the paper.
    """
    def __init__(self, cfg):
        """
        Initialize the model.

        Args:
        - cfg: dict with 'model_cfg' and 'stft_cfg' sections (see
          baseuse/models/rwsa_se.py).
        """
        super(MambAttentionSEUNet, self).__init__()
        self.cfg = cfg
        self.num_tscblocks = cfg['model_cfg'].get('num_tfmamba') or 4  # default tfmamba: 4
        self.use_rwsa = bool(cfg['model_cfg'].get('use_rwsa', False))
        self.grad_checkpoint = bool(cfg['model_cfg'].get('grad_checkpoint', False))

        self.dim = [cfg['model_cfg']['hid_feature'], cfg['model_cfg']['hid_feature'] * 2, cfg['model_cfg']['hid_feature'] * 3]
        dim = self.dim

        # Initialize dense encoder
        self.dense_encoder = DenseEncoder(cfg)

        # Initialize U-Net stages (TFMambaBlock, or MambAttentionBlock for RWSA)
        self.patch_embed_encoder_level1 = Patch_Embed_stage(dim[0], dim[0])

        if self.use_rwsa:
            self.shared_attention_lvl1 = nn.ModuleList([AttentionModule(dim=dim[0], n_head=4) for _ in range(self.num_tscblocks)])
            self.TSMamba1_encoder = nn.ModuleList([MambAttentionBlock(cfg, dim[0], shared_attention=self.shared_attention_lvl1[i]) for i in range(self.num_tscblocks)])
        else:
            self.TSMamba1_encoder = nn.ModuleList([TFMambaBlock(cfg, dim[0]) for _ in range(self.num_tscblocks)])

        self.down1_2 = Downsample(dim[0], dim[1])

        self.patch_embed_encoder_level2 = Patch_Embed_stage(dim[1], dim[1])

        if self.use_rwsa:
            self.shared_attention_lvl2 = nn.ModuleList([AttentionModule(dim=dim[1], n_head=4) for _ in range(self.num_tscblocks)])
            self.TSMamba2_encoder = nn.ModuleList([MambAttentionBlock(cfg, dim[1], shared_attention=self.shared_attention_lvl2[i]) for i in range(self.num_tscblocks)])
        else:
            self.TSMamba2_encoder = nn.ModuleList([TFMambaBlock(cfg, dim[1]) for _ in range(self.num_tscblocks)])

        self.down2_3 = Downsample(dim[1], dim[2])

        self.patch_embed_middle = Patch_Embed_stage(dim[2], dim[2])

        if self.use_rwsa:
            self.TSMamba_middle = nn.ModuleList([MambAttentionBlock(cfg, dim[2]) for _ in range(self.num_tscblocks)])
        else:
            self.TSMamba_middle = nn.ModuleList([TFMambaBlock(cfg, dim[2]) for _ in range(self.num_tscblocks)])

        ###########

        self.up3_2 = Upsample(int(dim[2]), dim[1])

        self.concat_level2 = nn.Sequential(
            nn.Conv2d(dim[1] * 2, dim[1], 1, 1, 0, bias=False),
        )

        self.patch_embed_decoder_level2 = Patch_Embed_stage(dim[1], dim[1])

        if self.use_rwsa:
            self.TSMamba2_decoder = nn.ModuleList([MambAttentionBlock(cfg, dim[1], shared_attention=self.shared_attention_lvl2[i]) for i in range(self.num_tscblocks)])
        else:
            self.TSMamba2_decoder = nn.ModuleList([TFMambaBlock(cfg, dim[1]) for _ in range(self.num_tscblocks)])

        self.up2_1 = Upsample(int(dim[1]), dim[0])

        self.concat_level1 = nn.Sequential(
            nn.Conv2d(dim[0] * 2, dim[0], 1, 1, 0, bias=False),
        )

        self.patch_embed_decoder_level1 = Patch_Embed_stage(dim[0], dim[0])

        if self.use_rwsa:
            self.TSMamba1_decoder = nn.ModuleList([MambAttentionBlock(cfg, dim[0], shared_attention=self.shared_attention_lvl1[i]) for i in range(self.num_tscblocks)])
        else:
            self.TSMamba1_decoder = nn.ModuleList([TFMambaBlock(cfg, dim[0]) for _ in range(self.num_tscblocks)])

        # Mag refine
        self.mag_patch_embed_refinement = Patch_Embed_stage(dim[0], dim[0])

        self.mag_refinement = nn.ModuleList([TFMambaBlock(cfg, dim[0]) for _ in range(self.num_tscblocks)])

        self.mag_output = nn.Sequential(
            nn.Conv2d(dim[0], dim[0], kernel_size=3, stride=1, padding=1, bias=False),

        )

        # Phase refine
        self.pha_patch_embed_refinement = Patch_Embed_stage(dim[0], dim[0])

        self.pha_refinement = nn.ModuleList([TFMambaBlock(cfg, dim[0]) for _ in range(self.num_tscblocks)])

        self.pha_output = nn.Sequential(
            nn.Conv2d(dim[0], dim[0], kernel_size=3, stride=1, padding=1, bias=False),

        )

        # Initialize decoders
        self.mask_decoder = MagDecoder(cfg)
        self.phase_decoder = PhaseDecoder(cfg)

    def _run_blocks(self, blocks, x, patch_embed=None, residual=None):
        """Run (optional patch_embed +) a block sequence.

        With grad_checkpoint enabled and training, the whole sequence is one
        checkpoint segment: activations are dropped after forward and
        recomputed during backward. The residual input is passed through the
        checkpoint boundary (it must be part of the segment inputs for the
        graph to stay correct)."""
        if residual is None:
            if patch_embed is not None:
                x = patch_embed(x)
            for block in blocks:
                x = block(x)
            return x
        # residual pattern: y = residual + blocks(patch_embed(x))
        def seg(inp, res):
            h = patch_embed(inp) if patch_embed is not None else inp
            for block in blocks:
                h = block(h)
            return res + h
        if self.grad_checkpoint and self.training and x.requires_grad:
            return checkpoint(seg, x, residual, use_reentrant=False)
        return seg(x, residual)

    def _run_plain(self, blocks, x):
        if self.grad_checkpoint and self.training and x.requires_grad:
            return checkpoint(blocks, x, use_reentrant=False)
        return blocks(x)

    def forward(self, noisy_mag, noisy_pha):
        """
        Forward pass.

        Args:
        - noisy_mag (torch.Tensor): Noisy magnitude input tensor [B, F, T].
        - noisy_pha (torch.Tensor): Noisy phase input tensor [B, F, T].

        Returns:
        - denoised_mag (torch.Tensor): Denoised magnitude tensor [B, F_out, T].
        - denoised_pha (torch.Tensor): Denoised phase tensor [B, F_out, T].
        - denoised_com (torch.Tensor): Denoised complex tensor [B, F_out, T, 2].
          F_out equals F or F+1 (freq padding bin); crop as needed.
        """
        # Reshape inputs
        noisy_mag = rearrange(noisy_mag, 'b f t -> b t f').unsqueeze(1)  # [B, 1, T, F]
        noisy_pha = rearrange(noisy_pha, 'b f t -> b t f').unsqueeze(1)  # [B, 1, T, F]

        # Concatenate magnitude and phase inputs
        x = torch.cat((noisy_mag, noisy_pha), dim=1)  # [B, 2, T, F]

        # Encode input
        x1 = self.dense_encoder(x)
        copy1 = x1  # residual basis for the mag/pha outputs

        # Apply U-Net Mamba blocks
        x1 = self._run_blocks(self.TSMamba1_encoder, x1,
                              patch_embed=self.patch_embed_encoder_level1,
                              residual=x1)

        x2 = self.down1_2(x1)
        x2 = self._run_blocks(self.TSMamba2_encoder, x2,
                              patch_embed=self.patch_embed_encoder_level2,
                              residual=x2)

        x3 = self.down2_3(x2)
        x3 = self._run_blocks(self.TSMamba_middle, x3,
                              patch_embed=self.patch_embed_middle,
                              residual=x3)

        y2 = self.up3_2(x3)
        y2 = torch.cat([y2, x2], 1)
        y2 = self.concat_level2(y2)

        y2 = self._run_blocks(self.TSMamba2_decoder, y2,
                              patch_embed=self.patch_embed_decoder_level2,
                              residual=y2)

        y1 = self.up2_1(y2)
        y1 = torch.cat([y1, x1], 1)
        y1 = self.concat_level1(y1)

        y1 = self._run_blocks(self.TSMamba1_decoder, y1,
                              patch_embed=self.patch_embed_decoder_level1,
                              residual=y1)

        # magnitude
        mag = y1
        mag = self._run_blocks(self.mag_refinement, mag,
                               patch_embed=self.mag_patch_embed_refinement,
                               residual=mag)
        mag = self.mag_output(mag) + copy1

        # phase
        pha = y1
        pha = self._run_blocks(self.pha_refinement, pha,
                               patch_embed=self.pha_patch_embed_refinement,
                               residual=pha)
        pha = self.pha_output(pha) + copy1

        # Decode magnitude and phase
        denoised_mag = rearrange(self.mask_decoder(mag) * noisy_mag, 'b c t f -> b f t c').squeeze(-1)
        denoised_pha = rearrange(self.phase_decoder(pha), 'b c t f -> b f t c').squeeze(-1)

        # Combine denoised magnitude and phase into a complex representation
        denoised_com = torch.stack(
            (denoised_mag * torch.cos(denoised_pha), denoised_mag * torch.sin(denoised_pha)),
            dim=-1
        )

        return denoised_mag, denoised_pha, denoised_com

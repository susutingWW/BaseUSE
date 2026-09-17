"""BaseUSE inference entry point.

Usage:
    python -m baseuse.inference \
        --input_scp /path/to/noisy.scp \
        --output_dir ./exp/enhanced \
        --ckpt_path ./exp/<tag>/<name>/version_0/checkpoints/best_xxx.ckpt
"""

from distutils.util import strtobool
import os

import soundfile as sf
import torch
import tqdm

from baseuse.models.se_model import SEModel
from baseuse.models.flow_se_model import FlowSEModel
from baseuse.models.rwsa_se import RWSAMambaSEModel
from baseuse.models.semamba_se import SEMambaSEModel
from baseuse.utils.compat import register_legacy_baseline_code_shim


def str2bool(value: str) -> bool:
    return bool(strtobool(value))


def main(args):
    device = args.device

    # checkpoints from the original urgent2026 repo pickle baseline_code.config.Config
    register_legacy_baseline_code_shim()

    # pick the LightningModule class from the checkpoint's pickled config
    ckpt = torch.load(args.ckpt_path, map_location=device, weights_only=False)
    ckpt_cfg = ckpt.get('hyper_parameters', {}).get('cfg')
    if getattr(ckpt_cfg, 'se_model', None) == 'semamba':
        model = SEMambaSEModel.load_from_checkpoint(args.ckpt_path, map_location=device)
    elif getattr(ckpt_cfg, 'se_model', None) == 'rwsamamba_unet':
        model = RWSAMambaSEModel.load_from_checkpoint(args.ckpt_path, map_location=device)
    else:
        try:
            model = SEModel.load_from_checkpoint(args.ckpt_path, map_location=device)
        except Exception:
            model = FlowSEModel.load_from_checkpoint(args.ckpt_path, map_location=device)
    model.eval()

    input_audios = {}
    with open(args.input_scp) as f:
        for line in f:
            uid, wav = line.strip().split()
            input_audios[uid] = wav

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.output_dir + "/wav", exist_ok=True)

    with open(args.output_dir + "/inf.scp", "w") as f:

        for uid in tqdm.tqdm(input_audios):
            wav_path = input_audios[uid]
            wav, sr = sf.read(wav_path)
            wav = torch.tensor(wav).float().to(device).view(1, -1)
            length = torch.tensor(wav.shape[-1]).to(device).view(1)

            with torch.no_grad():
                if isinstance(model, (SEModel, RWSAMambaSEModel)):
                    enhanced, _ = model.se_model(wav, length, sr)
                elif isinstance(model, FlowSEModel):
                    enhanced = model.enhance(wav, sr, length)

                enhanced = enhanced / enhanced.abs().max() * 0.9

                sf.write(args.output_dir + f"/wav/{uid}.wav", enhanced.cpu().numpy().flatten(), sr)

            print(f"{uid} {args.output_dir}/wav/{uid}.wav", file=f)

    print("done")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_scp",
        type=str,
        required=True,
        help="Path to the 2-column scp file containing noisy audio (uid wav_path)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        default="./tmp/se",
        help="Path to the output directory for writing enhanced speeches",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to the checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for inference",
    )

    args = parser.parse_args()

    main(args)

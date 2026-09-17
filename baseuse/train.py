"""BaseUSE training entry point.

Usage:
    python -m baseuse.train --config_file conf/exp/bsrnn_2025_dynamic.yaml
    python -m baseuse.train --config_file conf/exp/bsrnn_2025_dynamic.yaml --batch_size 2
"""

import os
import glob

import torch
import torch.multiprocessing as mp
import pytorch_lightning as L
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from baseuse.config import Config, config_parser
from baseuse.models.se_model import SEModel
from baseuse.models.flow_se_model import FlowSEModel
from baseuse.models.rwsa_se import RWSAMambaSEModel
from baseuse.models.semamba_se import SEMambaSEModel
from baseuse.data.datamodule import AudioDataModule
from baseuse.utils.compat import register_legacy_baseline_code_shim


class BatchSamplerEpochCallback(L.Callback):
    """Align the train batch sampler with the trainer epoch counter.

    PL 2.x never dispatches DataModule.on_train_epoch_start (only callback /
    LightningModule hooks), so without this the balanced sampler would fall
    back to its internal auto-advance and desync from trainer.current_epoch
    after a resume.
    """

    def on_train_epoch_start(self, trainer, pl_module):
        sampler = getattr(self, 'sampler', None)
        if sampler is None:
            # train_dataloader() (which creates the sampler) may not have run
            # yet at epoch 0; keep re-fetching until it exists.
            sampler = getattr(trainer.datamodule, 'train_batch_sampler', None)
            if sampler is not None:
                self.sampler = sampler
        if sampler is not None and hasattr(sampler, 'set_epoch'):
            sampler.set_epoch(trainer.current_epoch)


def prepare_call_backs(cfg):
    best_metrics = [
        ('val_loss', 'min'),
        ('val_sisnr', 'max'),
    ]
    call_backs = [LearningRateMonitor(logging_interval='step')]
    for i, (metric, min_or_max) in enumerate(best_metrics):
        call_back = ModelCheckpoint(
            filename="best_{epoch:02d}-{step:06d}-{" + metric + ":.3f}",
            save_top_k=cfg.save_top_k,
            monitor=metric,
            every_n_train_steps=cfg.val_check_interval,
            mode=min_or_max,
            save_weights_only=(metric != "val_loss"),
            save_last=(metric == "val_loss"),
            save_on_train_epoch_end=False,
        )
        call_backs.append(call_back)

    return call_backs


def main():
    mp.set_start_method('spawn')
    torch.set_float32_matmul_precision('medium')

    # checkpoints from the original urgent2026 repo pickle baseline_code.config.Config
    register_legacy_baseline_code_shim()

    args = config_parser()
    cfg = Config(**vars(args))
    cfg.read_yaml()
    print(cfg)
    L.seed_everything(seed=cfg.seed)

    if cfg.train_set_dynamic_mixing:
        os.environ['OMP_NUM_THREADS'] = "1"

    if cfg.model_type == "flowse":
        model = FlowSEModel(cfg=cfg)
    elif cfg.se_model == "rwsamamba_unet":
        model = RWSAMambaSEModel(cfg=cfg)
    elif cfg.se_model == "semamba":
        model = SEMambaSEModel(cfg=cfg)
    else:
        model = SEModel(cfg=cfg)

    if cfg.init_from != 'none':
        state_dict = torch.load(cfg.init_from, map_location="cpu", weights_only=False)
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        model.load_state_dict(state_dict)
        print(f"Init param loaded from {cfg.init_from}")

    print(model)

    logger = TensorBoardLogger(save_dir=f"./exp/{cfg.train_tag}", version=cfg.train_version, name=cfg.train_name)
    call_backs = prepare_call_backs(cfg=cfg)
    call_backs.append(BatchSamplerEpochCallback())

    ckpt_dir = f"./exp/{cfg.train_tag}/{cfg.train_name}/version_{cfg.train_version}/checkpoints"
    ckpts = glob.glob(f"{ckpt_dir}/*-val_loss*.ckpt") + glob.glob(f"{ckpt_dir}/last.ckpt")
    ckpts.sort(key=os.path.getmtime, reverse=True)
    last_ckpt = ckpts[0] if ckpts else None
    last_ckpt = last_ckpt if cfg.resume else None
    if last_ckpt is not None:
        print(f"Resume from {last_ckpt}")

    trainer = L.Trainer(
        max_epochs=cfg.num_train_epochs,
        accelerator=cfg.device,
        devices=cfg.num_gpu,
        # find_unused_parameters only pays off for models with unused
        # branches (flow control); rwsamamba/bsrnn use all params every step
        strategy='ddp_find_unused_parameters_true' if cfg.num_gpu > 1 else 'auto',
        gradient_clip_val=cfg.gradient_clip,
        logger=logger,
        val_check_interval=cfg.val_check_interval,
        callbacks=call_backs,
        precision=getattr(cfg, "precision", "32-true"),
    )
    trainer.fit(model=model, datamodule=AudioDataModule(config=cfg), ckpt_path=last_ckpt)


if __name__ == "__main__":
    main()

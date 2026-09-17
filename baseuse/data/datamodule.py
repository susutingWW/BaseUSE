import torch
import pytorch_lightning
from torch.utils.data import DataLoader

from baseuse.config import Config
from baseuse.data.dataset import (
    SimulationConfig,
    DynamicMixingDataset,
    PreSimulatedDataset,
    GroupedBatchSampler,
    collate_fn,
)


class AudioDataModule(pytorch_lightning.LightningDataModule):
    """Wires train/val datasets and dataloaders.

    Data selection priority: cfg.data (nested yaml section) > legacy flat
    fields (cfg.train_set_path / cfg.train_set_dynamic_mixing / ...).

    cfg.data example::

        data:
          train:
            type: dynamic_mixing        # or pre_simulated
            path: data/train_sources_2025
            max_duration: 96000
            use_high_pass: true
            retry_when_fails: true
            simulation: {...}           # optional, see SimulationConfig
          validation:
            type: pre_simulated
            path: data/val_2025
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.num_worker = config.num_worker
        self.batch_size = config.batch_size

        data_cfg = config.data if isinstance(config.data, dict) else {}
        train_cfg = data_cfg.get("train", {}) or {}
        valid_cfg = data_cfg.get("validation", {}) or {}

        # ---------------- train dataset ----------------
        if "type" in train_cfg:
            dynamic = train_cfg["type"] == "dynamic_mixing"
        else:
            dynamic = config.train_set_dynamic_mixing
        train_dir = train_cfg.get("path", config.train_set_path)

        max_duration = train_cfg.get("max_duration", config.max_duration)
        max_duration_sec = train_cfg.get(
            "max_duration_sec", getattr(config, "max_duration_sec", None))
        use_high_pass = train_cfg.get("use_high_pass", config.use_high_pass)
        retry_when_fails = train_cfg.get("retry_when_fails", True)
        sim_config = SimulationConfig.from_dict(train_cfg.get("simulation"))

        if dynamic:
            self.train_dataset = DynamicMixingDataset(
                speech_source_scp=f'{train_dir}/speech_sources.scp',
                noise_source_scp=f'{train_dir}/noise_scoures.scp',
                speech_length_file=f'{train_dir}/source_length.scp',
                rir_scp=f'{train_dir}/rirs.scp',
                windnoise_scp=f'{train_dir}/wind_noise_scoures.scp',
                retry_when_fails=retry_when_fails,
                max_duration=max_duration,
                max_duration_sec=max_duration_sec,
                use_high_pass=use_high_pass,
                sim_config=sim_config,
            )
        else:
            self.train_dataset = PreSimulatedDataset(
                clean_speech=f'{train_dir}/spk1.scp',
                noisy_speech=f'{train_dir}/wav.scp',
                utt2fs=f'{train_dir}/utt2fs',
                speech_length=f'{train_dir}/speech_length.scp',
                max_duration=max_duration,
            )

        # ---------------- validation dataset ----------------
        valid_dir = valid_cfg.get("path", config.valid_set_path)
        self.val_dataset = PreSimulatedDataset(
            clean_speech=f'{valid_dir}/spk1.scp',
            noisy_speech=f'{valid_dir}/wav.scp',
            utt2fs=f'{valid_dir}/utt2fs',
            speech_length=f'{valid_dir}/speech_length.scp',
            max_duration=valid_cfg.get("max_duration", -1),
            max_duration_sec=valid_cfg.get(
                "max_duration_sec", getattr(config, "max_duration_sec", None)),
        )

        self.train_batch_sampler = None
        self.val_batch_sampler = None

    def on_train_epoch_start(self):
        """Refresh batch sampler state at each epoch start.

        NOTE: PL 2.x only dispatches on_train_epoch_start to callbacks and the
        LightningModule, NOT to DataModules, so this method never fires. The
        balanced GroupedBatchSampler therefore auto-advances + resamples in
        its own __iter__ (called exactly once per epoch by the DataLoader);
        set_epoch() here remains for external/manual control.
        """
        if self.train_batch_sampler is not None:
            self.train_batch_sampler.set_epoch(self.current_epoch)

    def train_dataloader(self):
        # with a non-distributed strategy (single GPU) the process group is
        # never initialized and torch.distributed.get_rank() would raise
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            rank, world_size = 0, 1

        # optional epoch-resampling strategy (GAP-URGENet style):
        #   sampling: {strategy: balanced, num_per_epoch: 40000,
        #              pool_keywords: [...]}   # keywords optional
        data_cfg = self.config.data if isinstance(self.config.data, dict) else {}
        sampling_cfg = (data_cfg.get("train", {}) or {}).get("sampling", {}) or {}
        self.train_batch_sampler = GroupedBatchSampler(
            self.train_dataset,
            batch_size=self.batch_size,
            rank=rank,
            world_size=world_size,
            drop_last=True,
            sampling_strategy=sampling_cfg.get("strategy", "full"),
            num_per_epoch=sampling_cfg.get("num_per_epoch", 0),
            pool_keywords=sampling_cfg.get("pool_keywords", None),
        )
        return DataLoader(
            self.train_dataset,
            batch_sampler=self.train_batch_sampler,
            num_workers=self.num_worker,
            pin_memory=False,
            persistent_workers=True,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        self.val_batch_sampler = GroupedBatchSampler(
            self.val_dataset,
            batch_size=self.batch_size,
            rank=0,
            world_size=1,
            drop_last=True,
        )
        return DataLoader(
            self.val_dataset,
            batch_sampler=self.val_batch_sampler,
            num_workers=self.num_worker,
            pin_memory=False,
            persistent_workers=True,
            collate_fn=collate_fn,
        )

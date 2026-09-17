import copy
import random
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import soundfile
import torch
from torch.utils.data import BatchSampler
from torch.utils.data import DataLoader

from baseuse.simulation.generate_data_param import process_one_sample as get_simu_meta
from baseuse.simulation.simulate_data_from_param import process_one_sample, save_audio, read_audio


# Corpus pools for balanced sampling (GAP-URGENet style): every epoch draws
# `num_per_epoch // n_pools + 1` utterances WITH REPLACEMENT from each pool,
# so rare corpora (EARS 14k, VCTK 84k) get as much exposure as CommonVoice
# (859k). Order matters: first keyword found in the path wins.
DEFAULT_POOL_KEYWORDS = [
    'dns5_fullband', 'vctk', 'libritts', 'commonvoice', 'mls_segments', 'ears',
]


def extract_pool(path, keywords=DEFAULT_POOL_KEYWORDS):
    """Map an audio path to a corpus-pool name via substring match."""
    for kw in keywords:
        if kw in path:
            return kw
    return 'other'


@dataclass
class SimulationConfig:
    """On-the-fly simulation recipe.

    Defaults reproduce the official URGENT 2026 track1 baseline
    (baseline_code/dataset.py::SimulationConfigs). Field names match the
    argument names expected by baseuse.simulation.generate_data_param.
    """

    snr_low_bound: float = -5.0
    snr_high_bound: float = 20.0
    reuse_noise: bool = True

    prob_wind_noise: float = 0.05
    wind_noise_config: dict = field(default_factory=lambda: dict(
        threshold=[0.1, 0.3],
        ratio=[1, 20],
        attack=[5, 100],
        release=[5, 100],
        sc_gain=[0.8, 1.2],
        clipping_threshold=[0.85, 1.0],
        clipping_chance=0.75,
        wind_noise_snr_low_bound=-10.0,
        wind_noise_snr_high_bound=15.0,
    ))

    prob_reverberation: float = 0.5
    reuse_rir: bool = True

    augmentations: dict = field(default_factory=lambda: dict(
        bandwidth_limitation=dict(
            weight=1.0,
            resample_methods='random',
        ),
        clipping=dict(
            weight=1.0,
            clipping_min_quantile=[0.0, 0.1],
            clipping_max_quantile=[0.9, 1.0],
        ),
        codec=dict(
            weight=1.0,
            config=[
                dict(format='mp3', encoder=None, qscale=[1, 10]),
                dict(format='ogg', encoder=['vorbis'], qscale=[1, 10]),
            ]
        ),
        packet_loss=dict(
            weight=1.0,
            packet_duration_ms=20,
            max_continuous_packet_loss=10,
            packet_loss_rate=[0.05, 0.25],
        )
    ))

    num_augmentations: dict = field(default_factory=lambda: {
        0: 0.25,
        1: 0.40,
        2: 0.20,
        3: 0.15,
    })

    @classmethod
    def from_dict(cls, values=None):
        """Build a SimulationConfig from a (possibly partial) yaml dict."""
        config = cls()
        if not values:
            return config
        for key, val in values.items():
            if not hasattr(config, key):
                raise KeyError(f"Unknown simulation config key: {key}")
            if key == "num_augmentations" and isinstance(val, dict):
                val = {int(k): v for k, v in val.items()}
            setattr(config, key, val)
        return config


def read_kv_scp(scp):
    rtv = {}
    with open(scp, "r") as f:
        for line in f:
            uid, value = line.strip().split()
            assert uid not in rtv, (uid)
            rtv[uid] = value
    return rtv


def read_source_scp(scp):
    source_dict = defaultdict(dict)
    source_dict_flatten = {}
    with open(scp, "r") as f:
        for line in f:
            uid, fs, audio_path = line.strip().split()
            assert uid not in source_dict[int(fs)], (uid, fs)
            source_dict[int(fs)][uid] = audio_path
            source_dict_flatten[uid] = audio_path

    source_uids = {k: list(source_dict[k].keys()) for k in source_dict}

    return source_dict, source_uids, source_dict_flatten


class PreSimulatedDataset(torch.utils.data.Dataset):
    """Paired (noisy, clean) audio read from scp manifests produced offline."""

    def __init__(self, clean_speech, noisy_speech, utt2fs, speech_length, max_duration=-1,
                 max_duration_sec=None):

        self.clean_speech = read_kv_scp(clean_speech)
        self.noisy_speech = read_kv_scp(noisy_speech)
        self.utt2fs = {k: int(v) for k, v in read_kv_scp(utt2fs).items()}
        self.speech_length = {k: int(v)
                              for k, v in read_kv_scp(speech_length).items()}

        self.uid = list(self.clean_speech.keys())
        self.max_duration = max_duration
        self.max_duration_sec = max_duration_sec

        assert len(self.clean_speech) == len(self.noisy_speech)
        assert len(self.clean_speech) == len(self.utt2fs)
        assert len(self.clean_speech) == len(self.speech_length)

    def _cap_for_uid(self, uid):
        """Sample-count cap; with max_duration_sec set, capped by seconds at
        the utterance's native fs (a sample-count-only cap lets low-fs
        utterances grow to e.g. 12s @ 8k, which dominates memory)."""
        cap = self.max_duration
        if self.max_duration_sec is not None:
            sec_cap = int(round(self.utt2fs[uid] * self.max_duration_sec))
            cap = sec_cap if cap <= 0 else min(cap, sec_cap)
        return cap

    def get_source_length(self):
        if self.max_duration > 0 or self.max_duration_sec is not None:
            return [min(self.speech_length[k], self._cap_for_uid(k)) for k in self.uid]
        else:
            return [self.speech_length[k] for k in self.uid]

    def get_srs(self):
        return [self.utt2fs[k] for k in self.uid]

    def get_paths(self):
        return [self.clean_speech[k] for k in self.uid]

    def __len__(self):

        return len(self.clean_speech)

    def __getitem__(self, index):

        uid = self.uid[index]
        audio, fs = read_audio(self.clean_speech[uid])

        assert fs == self.utt2fs[uid]

        noisy, fs = read_audio(self.noisy_speech[uid])
        assert fs == self.utt2fs[uid]

        cap = self._cap_for_uid(uid)
        if cap > 0 and audio.shape[1] > cap:
            start = random.randint(0, audio.shape[1] - cap)
            audio = audio[:, start:start+cap]
            noisy = noisy[:, start:start+cap]

        speech_length = audio.shape[1]

        return audio, noisy, fs, speech_length


class DynamicMixingDataset(torch.utils.data.Dataset):
    """On-the-fly simulation dataset.

    Each __getitem__ picks a clean utterance from the speech source pool and
    simulates a noisy version (noise + RIR + bandwidth/clipping/codec/packet
    loss/wind noise) according to the SimulationConfig. No simulated audio is
    ever written to disk.
    """

    def __init__(self, speech_source_scp, noise_source_scp, rir_scp, windnoise_scp, speech_length_file,
                 use_high_pass=True, retry_when_fails=False, max_duration=240000,
                 max_duration_sec=None, sim_config: SimulationConfig = None):
        super().__init__()

        self.sim = sim_config if sim_config is not None else SimulationConfig()

        self.speech_source, self.speech_uids, self.speech_source_flt = read_source_scp(
            speech_source_scp)
        self.noise_source, self.noise_uids, self.noise_source_flt = read_source_scp(
            noise_source_scp)
        self.rirs, self.rir_uids, self.rirs_flt = read_source_scp(rir_scp)
        self.wind_noises, self.wind_noises_uids, self.wind_noises_flt = read_source_scp(
            windnoise_scp)

        self.all_noise_flt = copy.deepcopy(self.noise_source_flt)
        self.all_noise_flt.update(self.wind_noises_flt)

        self.source_length = {k: min(int(v), max_duration)
                              for k, v in read_kv_scp(speech_length_file).items()}
        self.max_duration = max_duration
        self.max_duration_sec = max_duration_sec
        if max_duration_sec is not None:
            # seconds-based cap, per utterance at its native fs (see
            # PreSimulatedDataset._cap_for_uid for the rationale)
            for fs, uids in self.speech_uids.items():
                sec_cap = int(round(fs * max_duration_sec))
                cap = sec_cap if max_duration <= 0 else min(max_duration, sec_cap)
                for uid in uids:
                    if uid in self.source_length:
                        self.source_length[uid] = min(self.source_length[uid], cap)

        self.length = sum([len(self.speech_source[k])
                          for k in self.speech_source])

        self.samplerates = list(self.speech_source.keys())
        self.fs_sub_lengths = [len(self.speech_source[k])
                               for k in self.samplerates]
        self.accum_lengths = [sum(self.fs_sub_lengths[0:i+1])
                              for i in range(len(self.fs_sub_lengths))]

        self.augmentations = list(self.sim.augmentations.keys())
        weight_augmentations = np.array(
            [v["weight"] for v in self.sim.augmentations.values()])
        self.weight_augmentations = weight_augmentations / \
            np.sum(weight_augmentations)
        self.use_high_pass = use_high_pass
        self.retry_when_fails = retry_when_fails

    def get_srs(self, ):

        srs = []
        for i in range(len(self)):
            srs.append(self._get_from_index(i)[0])
        return srs

    def get_paths(self, ):
        """Index-aligned source paths (for corpus-pool extraction)."""
        paths = []
        for i in range(len(self)):
            fs, real_idx = self._get_from_index(i)
            uid = self.speech_uids[fs][real_idx]
            paths.append(self.speech_source[fs][uid])
        return paths

    def get_source_length(self,):

        length = []
        for i in range(len(self)):
            fs, real_idx = self._get_from_index(i)
            uid = self.speech_uids[fs][real_idx]
            length.append(self.source_length[uid])

        return length

    def __len__(self):

        return self.length

    def simulation(self, ):
        pass

    def _get_from_index(self, index):

        real_idx = -1
        speech_fs = -1
        previous = 0

        for i, fs in enumerate(self.samplerates):
            if index >= previous and index < self.accum_lengths[i]:
                speech_fs = fs
                real_idx = index - previous
                break
            previous = self.accum_lengths[i]

        assert real_idx >= 0 and speech_fs > 0

        return speech_fs, real_idx

    def _cap_for_fs(self, fs):
        cap = self.max_duration
        if self.max_duration_sec is not None:
            sec_cap = int(round(fs * self.max_duration_sec))
            cap = sec_cap if cap <= 0 else min(cap, sec_cap)
        return cap

    def run_simulation(self, speech_uid, speech_length, sr):

        use_wind_noise = np.random.random() < self.sim.prob_wind_noise
        num_aug = np.random.choice(
            list(self.sim.num_augmentations.keys()),
            p=list(self.sim.num_augmentations.values()),
        )

        if num_aug == 0:
            aug = "none"
        else:
            aug = np.random.choice(
                self.augmentations,
                p=self.weight_augmentations,
                size=num_aug,
                replace=False,
            )
            # As wind-noise simulation include clipping,
            # we exclude clipping from augmentation list
            while use_wind_noise and "clipping" in aug:
                aug = np.random.choice(
                    self.augmentations,
                    p=self.weight_augmentations,
                    size=num_aug,
                    replace=False,
                )

        info = get_simu_meta(
            self.sim,
            speech_length,
            sr,
            noise_dic=self.noise_source,
            used_noise_dic=None,
            wind_noise_dic=self.wind_noises,
            used_wind_noise_dic=None,
            use_wind_noise=use_wind_noise,
            snr_range=(self.sim.snr_low_bound,
                       self.sim.snr_high_bound),
            wind_noise_snr_range=(self.sim.wind_noise_config['wind_noise_snr_low_bound'],
                                  self.sim.wind_noise_config['wind_noise_snr_high_bound']
                                  ),
            store_noise=False,
            rir_dic=self.rirs,
            used_rir_dic=None,
            augmentations=aug,
            force_1ch=True,
        )

        info['speech_uid'] = speech_uid
        info['id'] = speech_uid
        info['snr_dB'] = info['snr']

        speech, noisy_speech, fs = process_one_sample(
            info,
            store_noise=False,
            speech_dic=self.speech_source_flt,
            noise_dic=self.all_noise_flt,
            rir_dic=self.rirs_flt,
            highpass=self.use_high_pass,
            on_the_fly=True,
            max_duration=self._cap_for_fs(sr),

        )

        return speech, noisy_speech, fs

    def __getitem__(self, index):

        speech_fs, real_idx = self._get_from_index(index)

        speech_uid = self.speech_uids[speech_fs][real_idx]
        speech_path = self.speech_source[speech_fs][speech_uid]

        # Load speech sample (Channel, Time)
        if speech_path.endswith(".wav"):
            with soundfile.SoundFile(speech_path) as af:
                speech_length = af.frames
        else:
            # Sometimes the acutal loaded audio's length differs from af.frames
            speech_length = soundfile.read(speech_path)[0].shape[0]

        speech_length = min(self._cap_for_fs(speech_fs), speech_length)

        if self.retry_when_fails:
            attempts = 0

            while attempts < 3:
                try:
                    speech, noisy_speech, fs = self.run_simulation(
                        speech_uid, speech_length, speech_fs)
                    return speech, noisy_speech, fs, speech_length
                except:
                    attempts += 1

            # if simulation failed, return clean speech
            # (C, T) like read_audio; collate_fn requires 2-D mono
            speech, fs = soundfile.read(speech_path, always_2d=True)
            speech = speech[:, :1].T
            if speech.shape[1] > speech_length:
                speech = speech[:, :speech_length]
            noisy_speech = speech
            print('Simulation Failed after 3 times try, return clean speech')
            return speech, noisy_speech, fs, speech_length

        else:
            speech, noisy_speech, fs = self.run_simulation(
                speech_uid, speech_length, speech_fs)
            return speech, noisy_speech, fs, speech_length


class GroupedBatchSampler(BatchSampler):
    """Batches indices so that every batch shares one sampling rate.

    sampling_strategy:
      - 'full' (default): every epoch iterates ALL dataset indices once
        (legacy behavior; buckets built once at __init__).
      - 'balanced': GAP-URGENet style. At every set_epoch(), each corpus pool
        (path-keyword groups, see DEFAULT_POOL_KEYWORDS) contributes
        `num_per_epoch // n_pools + 1` indices drawn WITH REPLACEMENT; the
        combined list is reshuffled and re-bucketed. An utterance may thus
        appear several times per epoch (fresh simulation each visit) and the
        epoch length becomes num_per_epoch regardless of corpus size.
    """

    def __init__(self, dataset, batch_size, rank, world_size, seed=0, drop_last=False,
                 bucket_size_mult=100, sampler=None, sampling_strategy='full',
                 num_per_epoch=0, pool_keywords=None):
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.bucket_size = batch_size * bucket_size_mult  # bucket size
        self.epoch = 0
        self.world_size = world_size
        self.rank = rank
        self.seed = seed
        self.generator = torch.Generator().manual_seed(seed + rank + self.epoch)

        self.sampling_strategy = sampling_strategy
        self.num_per_epoch = num_per_epoch

        # static per-index metadata (dataset indices never change identity)
        self._sr_list = dataset.get_srs()
        self._len_list = dataset.get_source_length()

        if self.sampling_strategy == 'balanced':
            if num_per_epoch <= 0:
                raise ValueError(
                    "balanced sampling requires data.train.sampling.num_per_epoch > 0")
            keywords = pool_keywords if pool_keywords else DEFAULT_POOL_KEYWORDS
            self._pools = defaultdict(list)
            for idx, path in enumerate(dataset.get_paths()):
                self._pools[extract_pool(path, keywords)].append(idx)
            pool_stats = {p: len(v) for p, v in sorted(self._pools.items())}
            print(f"[GroupedBatchSampler] balanced pools "
                  f"(num_per_epoch={num_per_epoch}): {pool_stats}")
            self.buckets = []
            self._sample_epoch()
            # True = buckets already hold the draw for self.epoch and the next
            # __iter__ must serve them instead of advancing (see __iter__).
            self._pending = True
        else:
            self._pools = None
            self._build_buckets(range(len(self._sr_list)))

    def _sample_epoch(self):
        """GAP-style per-pool sampling with replacement, then re-bucket.

        Deterministic in (seed, epoch) and rank-independent, so every DDP
        rank draws the SAME global epoch list; per-rank splitting happens
        later at the batch level in _epoch_batches.
        """
        rng = random.Random(f"{self.seed}-{self.epoch}")
        n_pools = len(self._pools)
        epoch_indices = []
        for indices in self._pools.values():
            if len(indices) > 0:
                k = self.num_per_epoch // n_pools + 1
                epoch_indices.extend(rng.choices(indices, k=k))
        rng.shuffle(epoch_indices)
        self._build_buckets(epoch_indices)

    def _build_buckets(self, indices):
        """Group by sampling rate, sort by length, bucket (rank-agnostic).

        Sharding happens later at the BATCH level (see _epoch_batches):
        sharding indices per rank + per-rank drop_last makes epoch lengths
        differ across ranks (e.g. 1247 vs 1200 batches), which deadlocks DDP
        when one rank enters a validation broadcast while another is still
        in a training all-reduce.
        """
        sr_groups = defaultdict(list)
        for idx in indices:
            sr_groups[self._sr_list[idx]].append(idx)

        self.buckets = []
        for sr, idxs in sr_groups.items():
            sorted_indices = sorted(idxs, key=lambda x: self._len_list[x])
            for i in range(0, len(sorted_indices), self.bucket_size):
                self.buckets.append(sorted_indices[i:i + self.bucket_size])

    def _epoch_batches(self):
        """Deterministic, rank-independent global batch pool for this epoch.

        The pool is built identically on every rank (local RNG seeded by
        (seed, epoch) only), trimmed to a whole multiple of world_size, and
        sharded round-robin by whole batches, so every rank iterates exactly
        len(pool) // world_size batches per epoch.
        """
        rng = random.Random(f"{self.seed}-pool-{self.epoch}")
        buckets = [list(b) for b in self.buckets]
        rng.shuffle(buckets)
        all_batches = []
        for bucket in buckets:
            # shuffle within bucket
            rng.shuffle(bucket)
            # make batches
            for i in range(0, len(bucket), self.batch_size):
                batch = bucket[i:i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                all_batches.append(batch)
        # shuffle all batches
        rng.shuffle(all_batches)
        # equal per-rank epoch length: drop <world_size trailing batches
        usable = len(all_batches) - len(all_batches) % self.world_size
        return all_batches[:usable]

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.generator.manual_seed(self.seed + self.rank + self.epoch)
        if self.sampling_strategy == 'balanced':
            self._sample_epoch()
            self._pending = True

    def __iter__(self):
        if self.sampling_strategy == 'balanced' and not getattr(self, '_pending', False):
            # Fallback epoch advance when nobody calls set_epoch() (PL 2.x
            # does not dispatch DataModule.on_train_epoch_start; a callback
            # in train.py normally does). DataLoader iterates its
            # batch_sampler exactly once per epoch, so advance + resample here.
            self.epoch += 1
            self._sample_epoch()
        self._pending = False
        return iter(self._epoch_batches()[self.rank::self.world_size])

    def state_dict(self):
        return {'seed': self.seed, 'epoch': self.epoch}

    def __len__(self):
        return len(self._epoch_batches()) // self.world_size


def collate_fn(batch):
    """Pad variable-length audio (right zero padding) and stack.

    Input: batch = [(clean (1,T), noisy (1,T), fs, length), ...]
    Output: (padded_clean (B,1,Tmax), padded_noisy (B,1,Tmax), sr, lengths)
    """
    speechs = [torch.tensor(item[0]) for item in batch]
    noisy_speechs = [torch.tensor(item[1]) for item in batch]
    srs = [item[2] for item in batch]
    lengths = [item[3] for item in batch]

    # all sampling rates within a batch must agree (guaranteed by the sampler)
    assert all(sr == srs[0] for sr in srs), "sampling rate must be consistent within a batch"
    sr = srs[0]

    max_length = max(audio.shape[1] for audio in speechs)

    padded_audios = torch.stack([
        torch.nn.functional.pad(
            audio,
            (0, max_length - audio.shape[1]),  # right padding
            value=0.0
        ) for audio in speechs
    ], dim=0)  # (B, 1, T_max)

    padded_noisy_speech = torch.stack([
        torch.nn.functional.pad(
            audio,
            (0, max_length - audio.size(1)),  # right padding
            value=0.0
        ) for audio in noisy_speechs
    ], dim=0)  # (B, 1, T_max)

    return padded_audios, padded_noisy_speech, torch.tensor(sr, dtype=torch.int32), torch.tensor(lengths, dtype=torch.int32)

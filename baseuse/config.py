import os
import sys
import yaml
import argparse


class Config:
    """Flat experiment config (training + model), plus an optional nested `data` dict.

    Attribute names follow the URGENT 2026 track1 baseline so that checkpoints
    saved by either project remain loadable (see baseuse.utils.compat).

    The nested `data` section (loaded from yaml) is consumed by
    baseuse.data.datamodule.AudioDataModule; the legacy flat fields
    (train_set_path / valid_set_path / ...) are kept as fallback.
    """

    def __init__(self, **kwargs):
        # ---- training ----
        self.learning_rate = 1e-3
        self.batch_size = 4
        self.weight_decay = 1e-6
        self.adam_epsilon = 1e-8
        self.num_worker = 4
        self.num_train_epochs = 150
        self.device = "cuda"
        self.num_gpu = 1
        self.train_version = 0
        self.train_tag = "run_0"
        self.train_name = "baseline"
        self.val_check_interval = 5000
        self.save_top_k = 5
        self.resume = True
        self.seed = 1996
        self.gradient_clip = 0.5
        self.lr_step_size = 1
        self.lr_gamma = 0.85

        # ---- data (legacy flat fields, kept for checkpoint compatibility) ----
        self.train_set_path = "none"
        self.train_set_dynamic_mixing = True
        self.valid_set_path = "none"
        self.init_from = "none"
        self.max_duration = 96000
        self.use_high_pass = True

        # ---- model ----
        self.se_model = "bsrnn"
        self.model_type = "discriminative"
        self.model_configs = None

        # ---- flow (FlowSE) related ----
        self.ema_decay = 0.999
        self.theta = 1.5
        self.sigma_max = 0.5
        self.sigma_min = 0.05
        self.t_eps = 0.03
        self.T_rev = 1.0
        self.loss_type = "mse"
        self.loss_abs_exponent = 0.5

        # ---- STFT (FlowSE) ----
        self.n_fft = 1536
        self.hop_length = 384
        self.spec_transform_type = "exponent"
        self.spec_abs_exponent = 0.667
        self.spec_factor = 0.065
        self.bsrnn_hidden = 384

        # ---- BaseUSE extensions ----
        # nested data config, see conf/exp/*.yaml "data:" section
        self.data = None
        self.config_file = "none"
        # trade ~30% step time for ~6x activation memory in BSRNN (FlowSE)
        self.gradient_checkpointing = False
        # Lightning Trainer precision: "32-true" | "16-mixed" | "bf16-mixed"
        # (bf16-mixed: ~half activation memory + tensor-core speedup on
        #  Ampere+; no loss scaling needed unlike 16-mixed)
        self.precision = "32-true"

        for k, v in kwargs.items():
            self.__setattr__(k, v)

    def read_yaml(self):
        """Apply yaml overrides; train_tag defaults to the yaml file name."""
        if self.config_file == "none":
            return self

        with open(self.config_file, "r", encoding="utf-8") as f:
            values = yaml.safe_load(f) or {}

        for k, v in values.items():
            self.__setattr__(k, v)

        self.train_tag = os.path.basename(self.config_file).replace(".yaml", "")
        return self


def _str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def _cli_type(default):
    """argparse type for scalar fields only; nested configs come from yaml."""
    if isinstance(default, bool):
        return _str2bool
    if isinstance(default, (int, float, str)):
        return type(default)
    return None


def config_parser():
    """Priority: class defaults < experiment yaml < explicit command line flags.

    Fields whose value is a dict/list/None (data, model_configs, ...) are
    yaml-only and are not exposed as command line flags.
    """
    cfg = Config()
    parameters = vars(cfg)

    parser = argparse.ArgumentParser(description="BaseUSE training")
    for par, default in parameters.items():
        arg_type = _cli_type(default)
        if arg_type is None:
            parser.add_argument(f"--{par}", default=default)
        else:
            parser.add_argument(f"--{par}", type=arg_type, default=default)

    args = parser.parse_args()
    values = vars(args)

    # 1) apply yaml
    if values["config_file"] != "none":
        with open(values["config_file"], "r", encoding="utf-8") as f:
            yaml_values = yaml.safe_load(f) or {}
        for k, v in yaml_values.items():
            values[k] = v
        values["train_tag"] = os.path.basename(values["config_file"]).replace(".yaml", "")

    # 2) let explicitly-passed command line flags win over the yaml
    passed = {a.split("=")[0].lstrip("-") for a in sys.argv[1:] if a.startswith("--")}
    for k in passed:
        if k in parameters:
            values[k] = vars(args)[k]

    return args

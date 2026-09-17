import sys
import types


def register_legacy_baseline_code_shim():
    """Make checkpoints saved by the original urgent2026_challenge_track1
    baseline_code package loadable inside BaseUSE.

    The original code calls ``save_hyperparameters()`` on a
    ``baseline_code.config.Config`` object, so the object's class path is
    pickled into every checkpoint ('hyper_parameters'). Unpickling therefore
    requires an importable ``baseline_code.config.Config``. This shim aliases
    that module path to ``baseuse.config.Config`` (structurally identical), so
    both official pretrained checkpoints and our own checkpoints load.
    """
    if "baseline_code.config" in sys.modules:
        return

    from baseuse.config import Config

    pkg = types.ModuleType("baseline_code")
    pkg.__path__ = []  # mark as package
    cfgmod = types.ModuleType("baseline_code.config")
    cfgmod.Config = Config
    pkg.config = cfgmod

    sys.modules.setdefault("baseline_code", pkg)
    sys.modules.setdefault("baseline_code.config", cfgmod)

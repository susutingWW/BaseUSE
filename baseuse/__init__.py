"""BaseUSE: decoupled URGENT-style universal speech enhancement baseline.

Structure
---------
- baseuse.models      : SE models (BSRNN discriminative / BSRNN-Flow generative)
- baseuse.data        : datasets, datamodule, manifest builders
- baseuse.simulation  : on-the-fly / offline noise+augmentation simulation engine
- baseuse.evaluation  : URGENT objective metric scripts
- baseuse.train       : training entry point
- baseuse.inference   : inference entry point
- baseuse.evaluate    : evaluation entry point
"""

__version__ = "0.1"

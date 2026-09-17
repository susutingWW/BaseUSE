#!/usr/bin/env python
from setuptools import setup

install_requires = [
    "matplotlib==3.6.2",
    "numpy==1.23.5",
    "scipy==1.11.1",
    "sounddevice==0.4.6",
    "soundfile==0.12.1",
    "spectrum==0.8.1",
    "torch==2.2.1",
    "torchaudio==2.2.1",
    "torch-ema==0.3",
    "pytorch-lightning==2.5.2",
    "espnet==202412",
    "tensorboard==2.20.0",
    "ffmpeg==1.4",
    "soxr==0.5.0",
    "gdown==5.2.0",
    "tqdm",
    "PyYAML",
]

setup(
    name='baseuse',
    version='0.1',
    description='Decoupled URGENT-style universal speech enhancement baseline '
                '(models / data / simulation / evaluation), trainable on URGENT 2025 data',
    install_requires=install_requires,
    packages=[
        'baseuse',
        'baseuse.models',
        'baseuse.models.rwsamamba',
        'baseuse.sampling',
        'baseuse.data',
        'baseuse.data.utils',
        'baseuse.simulation',
        'baseuse.evaluation',
        'baseuse.utils',
    ],
)

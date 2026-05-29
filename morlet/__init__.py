"""Morlet: MST (Morlet Spectral Transformer) code release for SEED-family EEG emotion decoding."""

from importlib import metadata as _metadata

try:
    __version__ = _metadata.version("morlet")
except _metadata.PackageNotFoundError:
    __version__ = "0.1.0"

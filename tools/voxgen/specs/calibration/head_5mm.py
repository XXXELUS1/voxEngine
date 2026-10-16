"""Calibration: the same head at 5 mm voxels."""
from _human_head import build_head


def build():
    return build_head(0.005, name="head_5mm")

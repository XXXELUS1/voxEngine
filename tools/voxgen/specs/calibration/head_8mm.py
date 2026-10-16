"""Calibration: the same head at 8 mm voxels."""
from _human_head import build_head


def build():
    return build_head(0.008, name="head_8mm")

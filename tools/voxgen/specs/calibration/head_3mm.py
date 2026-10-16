"""Calibration: the same head at 3 mm voxels (compare with 5 mm and 8 mm)."""
from _human_head import build_head


def build():
    return build_head(0.003, name="head_3mm")

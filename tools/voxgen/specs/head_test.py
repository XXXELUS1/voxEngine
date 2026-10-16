"""The head the Godot lab scene loads (godot/assets/models/head_test.vxm): 5 mm voxels."""
from _human_head import build_head


def build():
    return build_head(0.005, name="head_test")

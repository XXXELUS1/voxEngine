"""Hybrid head: the form from the morphed MakeHuman mesh at 1 cm (mild caricature), the hair a
designed mass, the face drawn at 5 mm (voxgen.hybridhead + voxgen.facedetail)."""
from voxgen.hybridhead import HybridHeadParams, build_hybrid_head


def build():
    return build_hybrid_head(HybridHeadParams(), name="hybrid")

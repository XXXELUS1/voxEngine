"""The designed voxel-art head (voxgen.toonhead): both presets build, are mirror-symmetric,
have eyes, hair on skin, honest layers and pass the checks."""
import numpy as np
import pytest

from voxgen import checks
from voxgen.body import exposed
from voxgen.materials import MAT_AIR, MAT_BONE, MAT_EYE, MAT_HAIR, MAT_SKIN
from voxgen.toonhead import PRESETS, ToonHeadParams, build_toon_head


@pytest.fixture(scope="module", params=PRESETS)
def head(request):
    return build_toon_head(ToonHeadParams.preset_params(request.param), name="t_" + request.param)


def test_builds_and_passes_checks(head):
    part = head.part("head")
    hist = part.material_histogram()
    for m in (MAT_SKIN, MAT_BONE, MAT_EYE, MAT_HAIR):
        assert hist.get(m, 0) > 0, m
    assert hist[MAT_EYE] >= 2 * 2 * 2 * 2 * 3                 # two eyes of at least 3 x 2 design voxels
    assert part.voxel_size == 0.005
    assert part.origin[1] == pytest.approx(-ToonHeadParams.preset_params(head.meta["preset"]).neck_rows * 0.01)
    report = checks.run(head)
    assert report["errors"] == [], report["errors"]
    assert report["warnings"] == [], report["warnings"]


def test_symmetric_and_layered(head):
    part = head.part("head")
    mat = part.material
    assert np.array_equal(mat, mat[:, :, ::-1])                # the material field mirrors exactly
    solid = mat != MAT_AIR
    ex = exposed(solid)
    assert not (ex & (mat == MAT_BONE)).any()                  # bone never on the surface
    assert (ex & (mat == MAT_SKIN)).sum() > 0.5 * (ex & (mat != MAT_HAIR)).sum()
    # hair sits on skin or hair, and every design voxel became a 2x2x2 fine block (a hair/eye
    # voxel is never split: steps are two fine voxels)
    hair = mat == MAT_HAIR
    coarse = hair[::2, ::2, ::2]
    assert np.array_equal(hair, coarse.repeat(2, 0).repeat(2, 1).repeat(2, 2))


def test_knobs_and_errors():
    p = ToonHeadParams.preset_params("toon", hair_style="bald", stubble=True, expression="happy")
    m = build_toon_head(p)
    hist = m.part("head").material_histogram()
    assert MAT_HAIR not in hist or hist[MAT_HAIR] < 200       # bald: at most the protruding brows
    assert m.palette.get("stubble") in set(np.unique(m.part("head").color).tolist())
    with pytest.raises(ValueError):
        ToonHeadParams.preset_params("giant")
    with pytest.raises(ValueError):
        build_toon_head(ToonHeadParams(), fine_voxel=0.003)

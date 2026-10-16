import time

import numpy as np
from PIL import Image

from helpers import layered_ball
from voxgen import Part, render
from voxgen.materials import MAT_SKIN


def test_render_views_non_empty_non_uniform():
    model, p = layered_ball(hair=True)
    imgs = render.render_views(model, size=96)
    assert set(imgs) == set(render.VIEW_NAMES)
    for name, im in imgs.items():
        assert isinstance(im, Image.Image) and im.size == (96, 96)
        a = np.asarray(im)
        assert a.std() > 10, name                      # not a flat image
        assert (a != np.asarray(render.BG)).any(axis=2).sum() > 500, name  # something was drawn


def test_iso_views_show_three_face_shades_and_outline():
    model, p = layered_ball(n=24, r=0.05)
    a = np.asarray(render.render_views(model, ["iso"], size=128)["iso"]).astype(int)
    drawn = (a != np.asarray(render.BG)).any(axis=2)
    assert drawn.sum() > 1000
    # the outline colour is present around the silhouette
    assert (a == np.asarray(render.OUTLINE)).all(axis=2).sum() > 50
    # the lit side is brighter than the shadow side
    lum = a.sum(axis=2)
    assert lum[drawn].max() - lum[drawn].min() > 150


def test_state_tint_matches_mesher_rules():
    base = np.array([[200, 100, 50]], np.float32)
    assert np.allclose(render.tint_state(base, np.array([0])), base)
    bruise1 = render.tint_state(base, np.array([1]))[0]
    assert np.allclose(bruise1, base[0] * 0.55 + np.array([217, 89, 89]) * 0.45)
    burn3 = render.tint_state(base, np.array([3 << 2]))[0]
    assert np.allclose(burn3, base[0] * 0.1 + np.array([26, 20, 18]) * 0.9)
    blood = render.tint_state(base, np.array([0x20 | 0x10]))[0]
    assert np.allclose(blood, base[0] * 0.35 + np.array([115, 8, 8]) * 0.65)
    cut = render.tint_state(base, np.array([0x10]))[0]
    assert np.allclose(cut, base[0] * 0.85)
    wet = render.tint_state(base, np.array([0x40]))[0]
    assert np.allclose(wet, base[0])


def test_bare_part_renders_with_material_colours_and_sheet_is_fast():
    p = Part("cube", (16, 16, 16), 0.01)
    p.material[4:12, 4:12, 4:12] = MAT_SKIN
    im = render.render_views(p, ["front"], size=64)["front"]
    assert np.asarray(im).std() > 5
    model, part = layered_ball(n=48, r=0.1, hair=True)
    t0 = time.perf_counter()
    sheet = render.contact_sheet(model, size=256)
    dt = time.perf_counter() - t0
    assert sheet.size[0] > 1000 and sheet.size[1] > 500
    assert dt < 2.0, f"contact sheet took {dt:.2f}s"

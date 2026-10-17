"""Beauty render of a .vxm model in Blender (bpy as a python module). One style x light x view
per run; ``voxgen render`` drives this as a subprocess of the bpy venv:

    $VOXENGINE_BPY/bin/python -m voxgen.blender.render_styles MODEL.vxm --style cubes --light studio \
        --view three_quarter --size 640 --samples 96 --out build/renders/x/cubes_studio_three_quarter.png

Styles (``voxgen.blender.STYLES``): cubes (culled quads, full AO), bevel (cubes + Bevel modifier),
facet (cubes whose one-voxel steps are chamfered to 45 degree slopes: ``mesh.step_edge_weights``
with ``min_run=2`` + a Bevel limited by edge weight, 0.5 voxel, 1 segment; ``--facet-step 2``
for 2-voxel steps and a 1 voxel chamfer), facet_all (every
90 degree edge chamfered), slopes (``mesh_slopes.slope_mesh``: cubes whose regular staircases of
2:1, 1:1 and 1:2 treads:risers become planar facets of 26.6, 45 and 63.4 degrees, lone steps stay
cubes, no Bevel; ``--facet-step 2`` fits 4:2, 2:2, 2:4, a 2-granular part at its coarse scale, the
rest as fine cubes), chamfer (``mesh_slopes.chamfer_mesh``: the facet look built by the mesher, every
step edge cut 45 degrees ``facet_step / 2`` voxels into both faces, watertight, no Bevel seams),
smooth (surface nets, shade smooth, per-face voxel colour, skin material:
roughness 0.6, no bump, weak subsurface 0.1), clay (smooth + subsurface + thumb-mark bumps), toon
(EEVEE, 3-step ramp + Freestyle outline), mix (per part: skin parts smooth, the rest facet, see
``voxgen.blender.resolve_part_styles``; ``--part-styles head=smooth,hair=facet`` overrides any
mapping). A smooth part that carries the ``sdf`` field (``voxgen.sdfield``) is meshed by Hermite
surface nets (the exact zero crossing of the field, normals from its gradient as custom split
normals, no Laplacian pass: ``mesh.surface_nets(hermite=True)``); ``--no-hermite`` forces the
binary path, ``--hermite-passes N`` adds light cell-clamped passes on top of the exact crossings.
For parts without the field ``--smooth-passes N`` and ``--smooth-clamp cell|free``
(``mesh.surface_nets``: cell keeps the vertices in their cell, free restores the volume instead)
and ``--smooth-inflate V`` (outward offset in voxels, closes the groove against cubic hair) tune
the Laplacian smoothing. The json side-car records ``smooth.hermite``. The facet styles use
one flat material per palette colour (constant colour x corner AO) instead of the corner colour
attribute, so the bevel faces inherit a whole colour and never blend two; their colour jitter
defaults to 0.
Lights: studio (warm key 2700K, cool rim 8000K, soft fill) and candle (1850K point + faint 9000K
rim). Backgrounds: dark, grey, and light (warm light-grey world and ground, a cooler softer rig,
exposure -0.7). Views: camera presets around the model bbox (``voxgen.blender.VIEWS``). In the
smooth styles every material is part of the smooth surface; ``--cubic-materials 5,8`` (tooth, hair
ids) keeps those voxels as cubes instead.

Scene: every part of the model at its own origin (parts share the model space), Godot axes
mapped to Blender (Y up -> Z up, -Z forward -> -Y forward), a ground plane at y = min(0, model
bottom) so props built on y = 0 stand on it. Colour management: AgX medium-high contrast; Cycles
CPU with OIDN denoise; toon on EEVEE with samples // 4. Writes the PNG and a ``.json`` side-car
with timings, engine, faces and model info (read by ``voxgen.blender.sheet``).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np

import bpy  # noqa: E402
from mathutils import Vector  # noqa: E402

from voxgen import io_vxm, mesh, mesh_slopes  # noqa: E402
from voxgen.blender import (BACKGROUNDS, LIGHTS, SMOOTH_CLAMPS, STYLES, VIEWS, parse_part_styles,  # noqa: E402
                            resolve_part_styles)

SMOOTH_STYLES = ("smooth", "clay", "toon")
FACET_STYLES = ("facet", "facet_all")
SLOPE_STYLES = ("slopes",)
CHAMFER_STYLES = ("chamfer",)   # the facet chamfers built by mesh_slopes.chamfer_mesh, no Bevel modifier
LOWPOLY_STYLES = ("lowpoly",)   # surface nets + light Laplacian, then quadric collapse to a few flat facets
FLAT_STYLES = FACET_STYLES + SLOPE_STYLES + CHAMFER_STYLES + LOWPOLY_STYLES  # one flat material per palette colour, jitter 0
CHAMFER_FRACTION = float(os.environ.get("VOXGEN_CHAMFER", "0.5"))  # chamfer width per step voxel
LOWPOLY_PASSES = 14             # Laplacian passes before the collapse (free: the 1/2 U design steps melt into a curve)
LOWPOLY_MODE = os.environ.get("VOXGEN_LOWPOLY", "planar")   # planar (angle dissolve) | collapse
LOWPOLY_ANGLE_DEG = float(os.environ.get("VOXGEN_LOWPOLY_ANGLE", "10"))
LOWPOLY_RATIO = 0.06            # Decimate(collapse) face ratio: the low-poly look of the owner's bald reference
SMOOTH_DEFAULTS = {"passes": 2, "clamp": "cell", "inflate": 0.0, "jitter": 0.04,  # jitter: 0 in mix (flat skin)
                   "hermite": None, "hermite_passes": 0}  # hermite None = when the part has the sdf field
# background preset -> (world bottom, world top, world strength), ground colour, exposure
WORLDS = {"dark": ((0.015, 0.015, 0.017), (0.11, 0.11, 0.12), 0.3),
          "grey": ((0.22, 0.22, 0.23), (0.55, 0.55, 0.57), 0.6),
          "light": ((0.55, 0.52, 0.48), (1.0, 0.98, 0.95), 1.0)}
GROUNDS = {"dark": (0.018, 0.018, 0.0189), "grey": (0.30, 0.30, 0.315), "light": (0.62, 0.59, 0.54)}
EXPOSURES = {"dark": -0.25, "grey": -0.25, "light": -0.7}


# ----------------------------------------------------------------------------- helpers
def blackbody_rgb(T: float):
    """Linear sRGB (max channel = 1) of a black body at T kelvin (Kim et al. Planckian locus)."""
    T = max(1667.0, min(25000.0, T))
    if T <= 4000:
        x = -0.2661239e9 / T ** 3 - 0.2343589e6 / T ** 2 + 0.8776956e3 / T + 0.179910
    else:
        x = -3.0258469e9 / T ** 3 + 2.1070379e6 / T ** 2 + 0.2226347e3 / T + 0.240390
    if T <= 2222:
        y = -1.1063814 * x ** 3 - 1.34811020 * x ** 2 + 2.18555832 * x - 0.20219683
    elif T <= 4000:
        y = -0.9549476 * x ** 3 - 1.37418593 * x ** 2 + 2.09137015 * x - 0.16748867
    else:
        y = 3.0817580 * x ** 3 - 5.87338670 * x ** 2 + 3.75112997 * x - 0.37001483
    X, Y, Z = x / y, 1.0, (1 - x - y) / y
    r = 3.2406 * X - 1.5372 * Y - 0.4986 * Z
    g = -0.9689 * X + 1.8758 * Y + 0.0415 * Z
    b = 0.0557 * X - 0.2040 * Y + 1.0570 * Z
    r, g, b = (max(v, 0.0) for v in (r, g, b))
    m = max(r, g, b)
    return (r / m, g / m, b / m)


def srgb_to_linear(c):
    c = np.asarray(c, np.float32) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)


def godot_to_blender(p):
    """(N,3) Godot metres (Y up, -Z forward) -> Blender (Z up, -Y forward). Proper rotation."""
    p = np.asarray(p, np.float32)
    return np.stack([-p[:, 0], p[:, 2], p[:, 1]], axis=1)


def make_mesh_object(name: str, m: mesh.Mesh):
    """Blender object from a voxgen Mesh: CORNER colour (face colour) + CORNER ao, per-face smooth,
    material_index 1 for the cubic faces of a smooth mesh. A quad whose last two indices are equal
    is a triangle (``mesh_slopes``)."""
    me = bpy.data.meshes.new(name)
    q = m.quads
    tri = q[:, 2] == q[:, 3]
    faces = [row[:3] if t else row for row, t in zip(q.tolist(), tri.tolist())] if tri.any() else q.tolist()
    me.from_pydata(godot_to_blender(m.positions).tolist(), [], faces)
    me.update()
    F = m.n_faces
    nl = np.where(tri, 3, 4)
    loop_face = np.repeat(np.arange(F), nl)
    loop_corner = np.arange(len(loop_face)) - np.repeat(np.cumsum(nl) - nl, nl)
    rgb = srgb_to_linear(m.face_colors[:, :3])[loop_face]
    rgba = np.concatenate([rgb, np.ones((len(loop_face), 1), np.float32)], axis=1)
    col = me.color_attributes.new(name="Col", type="FLOAT_COLOR", domain="CORNER")
    col.data.foreach_set("color", rgba.ravel())
    aoa = me.attributes.new(name="ao", type="FLOAT", domain="CORNER")
    aoa.data.foreach_set("value", m.corner_ao[loop_face, loop_corner].astype(np.float32).ravel())
    me.polygons.foreach_set("use_smooth", m.face_smooth.tolist())
    if m.face_smooth.any():
        me.polygons.foreach_set("material_index", (~m.face_smooth).astype(np.int32).tolist())
    me.validate()
    me.update()
    if m.vertex_normals is not None and len(me.loops) == F * 4:
        # Hermite nets: the sdf gradient as custom split normals (smooth faces); flat faces keep theirs
        vn = godot_to_blender(m.vertex_normals)
        loops = vn[m.quads.ravel()].astype(np.float32)
        if not m.face_smooth.all():
            P = godot_to_blender(m.positions)[m.quads]
            fn = np.cross(P[:, 2] - P[:, 0], P[:, 3] - P[:, 1])
            fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
            flat = np.repeat(~m.face_smooth, 4)
            loops[flat] = np.repeat(fn, 4, axis=0)[flat]
        me.normals_split_custom_set(loops.tolist())
        me.update()
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    return ob


# ----------------------------------------------------------------------------- materials
def _base_color_nodes(nt, ao_strength: float, mute: float = 0.0):
    """Attribute Col * (AO mixed by strength) -> output socket."""
    nodes, links = nt.nodes, nt.links
    a_col = nodes.new("ShaderNodeAttribute")
    a_col.attribute_name = "Col"
    a_col.location = (-900, 0)
    a_ao = nodes.new("ShaderNodeAttribute")
    a_ao.attribute_name = "ao"
    a_ao.location = (-900, -250)
    m = nodes.new("ShaderNodeMath")
    m.operation = "MULTIPLY_ADD"  # ao*strength + (1-strength)
    m.inputs[1].default_value = ao_strength
    m.inputs[2].default_value = 1.0 - ao_strength
    m.location = (-650, -250)
    links.new(a_ao.outputs["Fac"], m.inputs[0])
    vm = nodes.new("ShaderNodeVectorMath")
    vm.operation = "SCALE"
    vm.location = (-450, 0)
    links.new(a_col.outputs["Color"], vm.inputs[0])
    links.new(m.outputs[0], vm.inputs["Scale"])
    out = vm.outputs["Vector"]
    if mute > 0:
        grey = nodes.new("ShaderNodeVectorMath")
        grey.operation = "MULTIPLY_ADD"  # v*(1-mute) + mute*0.5
        grey.inputs[1].default_value = (1 - mute,) * 3
        grey.inputs[2].default_value = (mute * 0.5,) * 3
        grey.location = (-250, 0)
        links.new(out, grey.inputs[0])
        out = grey.outputs["Vector"]
    return out


def material_principled(name, roughness, ao_strength, specular=0.5, clay=False, sss=0.0):
    """Principled with the corner colour x AO; ``clay`` adds the muted colour, strong subsurface and
    thumb-mark bumps, ``sss`` alone a weak skin subsurface (weight, no bumps)."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    outn = nt.nodes.new("ShaderNodeOutputMaterial")
    outn.location = (300, 0)
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (0, 0)
    nt.links.new(bsdf.outputs["BSDF"], outn.inputs["Surface"])
    nt.links.new(_base_color_nodes(nt, ao_strength, mute=0.12 if clay else 0.0), bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Specular IOR Level"].default_value = specular
    if sss > 0 and not clay:
        bsdf.inputs["Subsurface Weight"].default_value = sss
        bsdf.inputs["Subsurface Radius"].default_value = (1.0, 0.4, 0.25)
        bsdf.inputs["Subsurface Scale"].default_value = 0.004
        bsdf.subsurface_method = "RANDOM_WALK_SKIN"
    if clay:
        bsdf.inputs["Subsurface Weight"].default_value = 0.35
        bsdf.inputs["Subsurface Radius"].default_value = (1.0, 0.5, 0.3)
        bsdf.inputs["Subsurface Scale"].default_value = 0.006
        bsdf.subsurface_method = "RANDOM_WALK"
        texco = nt.nodes.new("ShaderNodeTexCoord")
        texco.location = (-900, -600)
        noise = nt.nodes.new("ShaderNodeTexNoise")
        noise.location = (-650, -600)
        noise.inputs["Scale"].default_value = 70.0  # object metres -> ~1.4 cm thumb marks
        noise.inputs["Detail"].default_value = 2.0
        noise.inputs["Roughness"].default_value = 0.5
        nt.links.new(texco.outputs["Object"], noise.inputs["Vector"])
        bump = nt.nodes.new("ShaderNodeBump")
        bump.location = (-300, -600)
        bump.inputs["Strength"].default_value = 1.0
        bump.inputs["Distance"].default_value = 0.008
        nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])
        nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def material_toon(name, ao_strength=0.4):
    """EEVEE toon: Diffuse -> Shader to RGB -> 3-step ramp, times vertex colour -> Emission."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    outn = nt.nodes.new("ShaderNodeOutputMaterial")
    outn.location = (600, 0)
    diff = nt.nodes.new("ShaderNodeBsdfDiffuse")
    diff.location = (-600, 300)
    diff.inputs["Color"].default_value = (1, 1, 1, 1)
    s2rgb = nt.nodes.new("ShaderNodeShaderToRGB")
    s2rgb.location = (-400, 300)
    nt.links.new(diff.outputs["BSDF"], s2rgb.inputs["Shader"])
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.location = (-200, 300)
    cr = ramp.color_ramp
    cr.interpolation = "CONSTANT"
    cr.elements[0].position = 0.0
    cr.elements[0].color = (0.30, 0.27, 0.30, 1)
    cr.elements[1].position = 0.22
    cr.elements[1].color = (0.62, 0.60, 0.58, 1)
    e = cr.elements.new(0.55)
    e.color = (1.0, 1.0, 1.0, 1)
    nt.links.new(s2rgb.outputs["Color"], ramp.inputs["Fac"])
    mul = nt.nodes.new("ShaderNodeVectorMath")
    mul.operation = "MULTIPLY"
    mul.location = (100, 0)
    nt.links.new(_base_color_nodes(nt, ao_strength), mul.inputs[0])
    nt.links.new(ramp.outputs["Color"], mul.inputs[1])
    em = nt.nodes.new("ShaderNodeEmission")
    em.location = (350, 0)
    em.inputs["Strength"].default_value = 1.0
    nt.links.new(mul.outputs["Vector"], em.inputs["Color"])
    nt.links.new(em.outputs["Emission"], outn.inputs["Surface"])
    return mat


def material_flat(name, rgb_u8, roughness=0.8, ao_strength=1.0, specular=0.5):
    """Principled with a constant palette colour times the corner "ao" attribute. No colour
    attribute, so a bevel face (which inherits one neighbour's material) keeps a whole palette
    colour: crisp colour borders across the chamfers."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    outn = nt.nodes.new("ShaderNodeOutputMaterial")
    outn.location = (300, 0)
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    nt.links.new(bsdf.outputs["BSDF"], outn.inputs["Surface"])
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Specular IOR Level"].default_value = specular
    a_ao = nt.nodes.new("ShaderNodeAttribute")
    a_ao.attribute_name = "ao"
    a_ao.location = (-900, -250)
    m = nt.nodes.new("ShaderNodeMath")
    m.operation = "MULTIPLY_ADD"
    m.inputs[1].default_value = ao_strength
    m.inputs[2].default_value = 1.0 - ao_strength
    m.location = (-650, -250)
    nt.links.new(a_ao.outputs["Fac"], m.inputs[0])
    vm = nt.nodes.new("ShaderNodeVectorMath")
    vm.operation = "SCALE"
    vm.location = (-450, 0)
    lin = srgb_to_linear(np.asarray(rgb_u8, np.float32))
    vm.inputs[0].default_value = (float(lin[0]), float(lin[1]), float(lin[2]))
    nt.links.new(m.outputs[0], vm.inputs["Scale"])
    nt.links.new(vm.outputs["Vector"], bsdf.inputs["Base Color"])
    return mat


def assign_palette_materials(ob, m: mesh.Mesh, cache: dict):
    """One flat material per distinct face colour of ``m`` (shared across parts through ``cache``);
    material_index of every face = its colour's slot."""
    rgb = m.face_colors[:, :3].astype(np.int64)
    keys = rgb[:, 0] * 65536 + rgb[:, 1] * 256 + rgb[:, 2]
    uniq, inv = np.unique(keys, return_inverse=True)
    for k in uniq:
        k = int(k)
        if k not in cache:
            cache[k] = material_flat(f"pal_{k:06x}", (k >> 16, (k >> 8) & 255, k & 255))
        ob.data.materials.append(cache[k])
    ob.data.polygons.foreach_set("material_index", inv.astype(np.int32).ravel())
    ob.data.update()


def add_chamfer(ob, ew: mesh.EdgeWeights, width_m: float):
    """Bevel modifier limited by the per-edge weights of ``mesh.step_edge_weights``: one flat segment
    of ``width_m`` (0.5 voxel per step voxel) turns every weighted 90 degree edge into an exact 45
    degree slope."""
    me = ob.data
    ev = np.zeros(len(me.edges) * 2, np.int64)
    me.edges.foreach_get("vertices", ev)
    ev = ev.reshape(-1, 2)
    nv = len(me.vertices)
    bk = np.minimum(ev[:, 0], ev[:, 1]) * nv + np.maximum(ev[:, 0], ev[:, 1])
    keys = ew.keys(nv)
    pos = np.clip(np.searchsorted(keys, bk), 0, max(len(keys) - 1, 0))
    ok = (keys[pos] == bk) if len(keys) else np.zeros(len(bk), bool)
    w = np.where(ok, ew.weights[pos], 0.0).astype(np.float32)
    attr = me.attributes.new(name="bevel_weight_edge", type="FLOAT", domain="EDGE")
    attr.data.foreach_set("value", w)
    me.update()
    mod = ob.modifiers.new("Chamfer", "BEVEL")
    mod.affect = "EDGES"
    mod.offset_type = "OFFSET"
    mod.width = width_m
    mod.segments = 1
    mod.limit_method = "WEIGHT"
    mod.harden_normals = False
    mod.use_clamp_overlap = False
    mod.loop_slide = True
    mod.miter_outer = "MITER_SHARP"
    mod.miter_inner = "MITER_SHARP"
    return mod


def material_ground(rgb):
    mat = bpy.data.materials.new("Ground")
    mat.use_nodes = True
    b = mat.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*rgb, 1)
    b.inputs["Roughness"].default_value = 0.95
    b.inputs["Specular IOR Level"].default_value = 0.15 if rgb[0] > 0.5 else 0.2
    return mat


# ----------------------------------------------------------------------------- scene
def build_world(bottom, top, strength):
    w = bpy.data.worlds.new("World")
    w.use_nodes = True
    nt = w.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputWorld")
    bg = nt.nodes.new("ShaderNodeBackground")
    bg.inputs["Strength"].default_value = strength
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Generated"], sep.inputs[0])
    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.inputs["From Min"].default_value = -1.0
    mr.inputs["From Max"].default_value = 1.0
    nt.links.new(sep.outputs["Z"], mr.inputs["Value"])
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.interpolation = "EASE"
    ramp.color_ramp.elements[0].color = (*bottom, 1)
    ramp.color_ramp.elements[1].color = (*top, 1)
    nt.links.new(mr.outputs[0], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bg.inputs["Color"])
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
    bpy.context.scene.world = w
    return w


def add_light(name, kind, loc, target, power, kelvin, size=0.5, radius=0.02):
    ld = bpy.data.lights.new(name, kind)
    ld.energy = power
    ld.color = blackbody_rgb(kelvin)
    if kind == "AREA":
        ld.shape = "DISK"
        ld.size = size
    elif kind == "POINT":
        ld.shadow_soft_size = radius
    ob = bpy.data.objects.new(name, ld)
    bpy.context.scene.collection.objects.link(ob)
    ob.location = Vector(loc)
    ob.rotation_euler = (Vector(target) - ob.location).to_track_quat("-Z", "Y").to_euler()
    return ob


def setup_lights(preset: str, bg: str, center, size_m):
    """center / size_m in Blender space; the camera is on the -Y side."""
    c = Vector(center)
    s = max(size_m)
    k = (s / 0.28) ** 2  # powers were tuned on the 0.28 m head; lights sit at distances ~ s, so scale by s^2
    world = WORLDS[bg]
    if preset == "studio" and bg == "light":
        # the bright world carries the fill: a cooler, weaker key and rim so the chamfers read as planes
        build_world(*world)
        add_light("Key", "AREA", c + Vector((-2.6 * s, -2.4 * s, 2.8 * s)), c, 16 * k, 4300, size=1.8 * s)
        add_light("Fill", "AREA", c + Vector((2.8 * s, -3.2 * s, 0.8 * s)), c, 4 * k, 6500, size=3.5 * s)
        add_light("Rim", "AREA", c + Vector((2.4 * s, 2.6 * s, 2.0 * s)), c, 20 * k, 8000, size=1.0 * s)
    elif preset == "studio":
        build_world(*world)
        add_light("Key", "AREA", c + Vector((-2.6 * s, -2.4 * s, 2.6 * s)), c, 40 * k, 2700, size=1.6 * s)
        add_light("Fill", "AREA", c + Vector((2.8 * s, -3.2 * s, 0.8 * s)), c, 7 * k, 5500, size=3.5 * s)
        add_light("Rim", "AREA", c + Vector((2.4 * s, 2.6 * s, 2.0 * s)), c, 90 * k, 8000, size=0.9 * s)
    elif preset == "candle":
        if bg == "dark":
            build_world((0.004, 0.005, 0.012), (0.012, 0.014, 0.03), 0.5)
        else:
            build_world(*world)
        face = c + Vector((0, -0.55 * s, -0.15 * s))
        add_light("Candle", "POINT", face + Vector((-0.75 * s, -0.75 * s, -0.35 * s)), c, 12.0 * k,
                  1850, radius=0.012)
        add_light("Rim", "AREA", c + Vector((2.2 * s, 2.6 * s, 2.2 * s)), c, 40 * k, 9000, size=1.0 * s)
    else:
        raise ValueError(preset)


def setup_camera(bbox_min, bbox_max, az_deg, el_deg, lens, margin):
    """Camera on a sphere around the bbox centre, pulled back until all 8 corners fit the square frame."""
    scene = bpy.context.scene
    cam_d = bpy.data.cameras.new("Cam")
    cam_d.lens = lens
    cam_d.sensor_width = 36.0
    cam_d.sensor_fit = "AUTO"
    cam = bpy.data.objects.new("Cam", cam_d)
    scene.collection.objects.link(cam)
    scene.camera = cam
    mn, mx = Vector(bbox_min), Vector(bbox_max)
    pivot = (mn + mx) / 2
    az, el = math.radians(az_deg), math.radians(el_deg)
    d = Vector((math.sin(az) * math.cos(el), -math.cos(az) * math.cos(el), math.sin(el)))
    rot = (-d).to_track_quat("-Z", "Y")
    t = 18.0 / lens  # tan(half fov), 36 mm sensor, square frame
    inv = rot.inverted()
    need = 0.0
    for i in range(8):
        corner = Vector((mx.x if i & 1 else mn.x, mx.y if i & 2 else mn.y, mx.z if i & 4 else mn.z)) - pivot
        lc = inv @ corner
        along = corner.dot(d)
        for v in (lc.x, lc.y):
            need = max(need, abs(v) / t + along)
    cam.location = pivot + d * need * margin
    cam.rotation_euler = rot.to_euler()
    return cam


def setup_render(engine, size, samples, out_path, freestyle=False, exposure=-0.25):
    scene = bpy.context.scene
    r = scene.render
    r.resolution_x = r.resolution_y = size
    r.resolution_percentage = 100
    r.image_settings.file_format = "PNG"
    r.image_settings.color_mode = "RGB"
    r.filepath = out_path
    r.film_transparent = False
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "AgX - Medium High Contrast"
    scene.view_settings.exposure = exposure
    if engine == "CYCLES":
        r.engine = "CYCLES"
        cy = scene.cycles
        cy.device = "CPU"
        cy.samples = samples
        cy.use_adaptive_sampling = True
        cy.use_denoising = True
        cy.denoiser = "OPENIMAGEDENOISE"
        cy.denoising_input_passes = "RGB_ALBEDO_NORMAL"
        cy.max_bounces = 6
        cy.diffuse_bounces = 3
        cy.glossy_bounces = 3
        cy.transmission_bounces = 2
        cy.volume_bounces = 0
        cy.caustics_reflective = False
        cy.caustics_refractive = False
    else:
        r.engine = "BLENDER_EEVEE_NEXT" if hasattr(bpy.types, "SceneEEVEE") else "BLENDER_EEVEE"
        ee = scene.eevee
        ee.taa_render_samples = samples
        try:
            ee.use_shadows = True
            ee.use_raytracing = False
        except AttributeError:
            pass
    r.use_freestyle = freestyle
    if freestyle:
        r.line_thickness_mode = "ABSOLUTE"
        r.line_thickness = 1.0
        vl = scene.view_layers[0]
        vl.use_freestyle = True
        fs = vl.freestyle_settings
        fs.crease_angle = math.radians(120.0)
        fs.use_culling = True
        fs.use_smoothness = True
        # bpy module mode: the empty scene has a lineset whose linestyle is None (Freestyle crashes on it)
        for extra in list(fs.linesets)[1:]:
            fs.linesets.remove(extra)
        ls = fs.linesets[0] if len(fs.linesets) else fs.linesets.new("outline")
        ls.name = "outline"
        if ls.linestyle is None:
            ls.linestyle = bpy.data.linestyles.new("Outline")
        ls.select_silhouette = True
        ls.select_border = True
        ls.select_crease = True
        ls.select_by_visibility = True
        ls.select_by_edge_types = True
        ls.linestyle.color = (0.02, 0.015, 0.02)
        ls.linestyle.thickness = 1.5
        ls.linestyle.caps = "ROUND"


# ----------------------------------------------------------------------------- main
def build_geometry(model, part_styles: dict, cubic_materials=(), jitter: float | None = None, smooth=None,
                   facet_step: int = 1):
    """One mesh object per part (each at its own origin), meshed by its own style (``part_styles``:
    part name -> style). Returns (objects, styles per object, bbox_min, bbox_max, faces, chamfer
    stats). ``cubic_materials``: material ids rendered as cubes inside the smooth styles; ``jitter``:
    the per-voxel colour jitter of the cube mesher (None = the style's default: 0 for the facet
    styles, 0.06 otherwise; 0 = flat palette colours, the stylized look); ``smooth``: the
    ``SMOOTH_DEFAULTS`` dict (passes, clamp, inflate, hermite, hermite_passes) of
    ``mesh.surface_nets``; ``facet_step``: the tread depth in voxels that counts as a step for the
    facet styles (``mesh.step_edge_weights``), the chamfer being 0.5 * facet_step voxel wide. The
    facet parts get their per-colour materials and the weighted Chamfer bevel here. Returns (objects, styles per object, bbox_min, bbox_max, faces, chamfer stats,
    hermite part names)."""
    smooth = {**SMOOTH_DEFAULTS, **(smooth or {})}
    objs, styles, faces = [], [], 0
    lo, hi = None, None
    mat_cache, chamfer, hermite_parts = {}, {}, []
    for i, part in enumerate(model.parts):
        style = part_styles[part.name or f"part{i}"]
        if style in SMOOTH_STYLES:
            hermite = smooth["hermite"]
            if hermite is None:
                hermite = part.sdf is not None
            m = mesh.surface_nets(part, model.palette, smooth_passes=int(smooth["passes"]),
                                  keep_cubic_materials=tuple(cubic_materials), clamp=smooth["clamp"],
                                  inflate=float(smooth["inflate"]), jitter=smooth["jitter"],
                                  hermite=bool(hermite), hermite_passes=int(smooth["hermite_passes"]))
            if m.hermite:
                hermite_parts.append(part.name or f"part{i}")
        elif style in LOWPOLY_STYLES:
            # the low-poly look of the owner's bald-head reference: nets vertices sit at the cell
            # centres of the binary field (no Laplacian, no Hermite), flat shading, then Blender's
            # planar dissolve merges the near-coplanar strips of a staircase into one facet
            hermite = part.sdf is not None      # the exact surface of the form when the sdf field is there
            m = mesh.surface_nets(part, model.palette, smooth_passes=0 if hermite else LOWPOLY_PASSES,
                                  keep_cubic_materials=tuple(cubic_materials), clamp="free", inflate=0.0, jitter=0.0,
                                  hermite=hermite, hermite_passes=0)
            m.face_smooth[:] = False
        elif style in SLOPE_STYLES:
            j = jitter if jitter is not None else 0.0
            st = {}
            m = mesh_slopes.slope_mesh_part(part, model.palette, step=int(facet_step), min_run=2, jitter=j, stats=st)
            for key in ("faces_in", "faces_sloped", "risers_dropped", "triangles", "faces_out", "mixed_blocks"):
                chamfer["slopes_" + key] = chamfer.get("slopes_" + key, 0) + int(st.get(key, 0))
        elif style in CHAMFER_STYLES:
            j = jitter if jitter is not None else 0.0
            st = {}
            m = mesh_slopes.chamfer_mesh_part(part, model.palette, step=int(facet_step), min_run=2, jitter=j, stats=st)
            for key in ("chamfer_edges", "faces_in", "faces_sloped", "risers_dropped", "triangles", "faces_out",
                        "checkerboards_filled"):
                chamfer["chamfer_" + key] = chamfer.get("chamfer_" + key, 0) + int(st.get(key, 0))
        else:
            j = jitter if jitter is not None else (0.0 if style in FACET_STYLES else 0.06)
            m = mesh.culled_quads(part, model.palette, jitter=j, ao=style not in FLAT_STYLES)   # baked AO would stripe the chamfers
        if m.n_faces == 0:
            continue
        ob = make_mesh_object(part.name or "part", m)
        objs.append(ob)
        styles.append(style)
        faces += m.n_faces
        if style in SLOPE_STYLES or style in CHAMFER_STYLES:
            assign_palette_materials(ob, m, mat_cache)
        if style in LOWPOLY_STYLES:
            assign_palette_materials(ob, m, mat_cache)
            if LOWPOLY_MODE == "planar":
                # merge near-coplanar faces into big flat facets (the low-poly look of the sheet:
                # large planes on the cheeks and forehead, small ones where the surface curves)
                mod = ob.modifiers.new("LowPoly", "DECIMATE")
                mod.decimate_type = "DISSOLVE"
                mod.angle_limit = math.radians(LOWPOLY_ANGLE_DEG)
                mod.use_dissolve_boundaries = False
                mod.delimit = {"MATERIAL"}
                tri = ob.modifiers.new("Tri", "TRIANGULATE")
                tri.quad_method = "BEAUTY"
                tri.ngon_method = "BEAUTY"
            else:
                mod = ob.modifiers.new("LowPoly", "DECIMATE")
                mod.decimate_type = "COLLAPSE"
                mod.ratio = LOWPOLY_RATIO
                mod.use_collapse_triangulate = True
        if style in FACET_STYLES:
            assign_palette_materials(ob, m, mat_cache)
            ew = mesh.step_edge_weights(m, part, mode="all" if style == "facet_all" else "steps", min_run=2,
                                        step=int(facet_step))
            add_chamfer(ob, ew, CHAMFER_FRACTION * facet_step * part.voxel_size)
            for key, val in ew.stats.items():
                chamfer[key] = chamfer.get(key, 0) + val
        p = godot_to_blender(m.positions)
        lo = p.min(0) if lo is None else np.minimum(lo, p.min(0))
        hi = p.max(0) if hi is None else np.maximum(hi, p.max(0))
    if not objs:
        raise ValueError(f"model {model.name!r} has no solid voxels")
    return objs, styles, lo, hi, faces, chamfer, hermite_parts


def style_materials(style: str, cache: dict):
    """(main, cubic-slot) materials of a part style, made once per style. ``smooth`` is the skin look
    (roughness 0.6, half AO, weak subsurface, no bumps); ``cubic`` is slot 1 of a smooth mesh (the
    hair/teeth cubes of ``--cubic-materials``), None where the style has no second slot."""
    if style in cache:
        return cache[style]
    cubic = None
    if style == "cubes":
        m = material_principled("Cubes", roughness=0.8, ao_strength=1.0)
    elif style == "bevel":
        m = material_principled("Bevel", roughness=0.75, ao_strength=0.85)
    elif style == "smooth":
        m = material_principled("Smooth", roughness=0.6, ao_strength=0.5, sss=0.1)
    elif style == "clay":
        m = material_principled("Clay", roughness=0.55, ao_strength=0.4, specular=0.35, clay=True)
    elif style == "toon":
        m = material_toon("Toon", ao_strength=0.4)
        cubic = m  # toon keeps one look
    else:
        raise ValueError(style)
    if style in ("smooth", "clay"):
        cubic = cache.get("_cubic") or material_principled("HairCubes", roughness=0.85, ao_strength=0.9, specular=0.3)
        cache["_cubic"] = cubic
    cache[style] = (m, cubic)
    return cache[style]


def assign_materials(objs, styles, voxel_size: float):
    """Materials (and the Bevel modifier of ``bevel``) per object by its style; the facet objects
    already carry their per-colour flat materials from ``build_geometry``."""
    cache = {}
    for ob, style in zip(objs, styles):
        if style in FLAT_STYLES:
            continue
        m, cubic = style_materials(style, cache)
        ob.data.materials.append(m)
        if cubic is not None:
            ob.data.materials.append(cubic)
        if style == "bevel":
            mod = ob.modifiers.new("Bevel", "BEVEL")
            mod.width = 0.25 * voxel_size
            mod.segments = 2
            mod.limit_method = "ANGLE"
            mod.angle_limit = math.radians(30)
            mod.harden_normals = True
            mod.miter_outer = "MITER_ARC"
            mod.use_clamp_overlap = True


def render_one(model_path, style, light, view, size, samples, out, bg="dark", engine=None, save_blend=None,
               index=None, cubic_materials=(), jitter: float | None = None, part_styles=None, smooth=None,
               facet_step: int = 1) -> dict:
    """``part_styles``: {part name: style} overrides on top of ``style`` (``mix`` maps skin parts to
    smooth and the rest to facet); ``smooth``: surface-nets options, see ``SMOOTH_DEFAULTS``;
    ``facet_step``: step depth in voxels of the facet chamfer (see ``build_geometry``)."""
    model = io_vxm.read(model_path)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"

    t0 = time.time()
    if jitter is None:
        jitter = 0.0 if style in FLAT_STYLES or style == "mix" else 0.06
    smooth = {**SMOOTH_DEFAULTS, **(smooth or {})}
    if style == "mix" or (jitter == 0.0 and style in FLAT_STYLES):
        smooth["jitter"] = jitter  # flat palette colours on the smooth skin too (the stylized look)
    mapping = resolve_part_styles(model, style, part_styles)
    objs, styles, bmin, bmax, faces, chamfer, hermite_parts = build_geometry(model, mapping, cubic_materials, jitter,
                                                                             smooth, facet_step)
    size_m = bmax - bmin
    center = (bmin + bmax) / 2
    assign_materials(objs, styles, model.parts[0].voxel_size)

    # ground: y = 0 of the model space (props stand on it), or lower when the model goes below 0
    ground_z = min(0.0, float(bmin[2])) - 0.0005
    bpy.ops.mesh.primitive_plane_add(size=max(6.0, 12.0 * float(size_m.max())),
                                     location=(float(center[0]), float(center[1]), ground_z))
    ground = bpy.context.active_object
    ground.name = "Ground"
    ground.data.materials.append(material_ground(GROUNDS[bg]))

    setup_lights(light, bg, center, size_m)
    az, el, lens, margin = VIEWS[view]
    cam_min = np.minimum(bmin, [bmin[0], bmin[1], ground_z]) if view == "full" else bmin
    setup_camera(cam_min, bmax, az, el, lens, margin)

    engine = engine or ("EEVEE" if style == "toon" else "CYCLES")
    engine = "CYCLES" if engine.upper() == "CYCLES" else "EEVEE"
    spl = samples if engine == "CYCLES" else max(16, samples // 4)
    out = os.path.abspath(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    setup_render(engine, size, spl, out, freestyle=(style == "toon"), exposure=EXPOSURES[bg])
    t_setup = time.time() - t0
    if save_blend:
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(save_blend))

    t1 = time.time()
    bpy.ops.render.render(write_still=True)
    dt = time.time() - t1
    p0 = model.parts[0]
    info = {
        "style": style, "light": light, "view": view, "bg": bg, "engine": engine, "samples": spl, "size": size,
        "render_seconds": round(dt, 1), "setup_seconds": round(t_setup, 2),
        "verts": int(sum(len(o.data.vertices) for o in objs)), "faces": int(faces),
        "model": model.name, "parts": [p.name for p in model.parts], "part_styles": mapping,
        "smooth": {"passes": int(smooth["passes"]), "clamp": smooth["clamp"], "inflate": float(smooth["inflate"]),
                   "hermite": bool(hermite_parts), "hermite_passes": int(smooth["hermite_passes"]),
                   "hermite_parts": hermite_parts},
        "hermite": bool(hermite_parts),
        "dims": "x".join(str(v) for v in (p0.nx, p0.ny, p0.nz)) + ("+" if len(model.parts) > 1 else ""),
        "voxel_mm": round(p0.voxel_size * 1000, 2), "blender": bpy.app.version_string, "out": out,
        "cubic_materials": [int(v) for v in cubic_materials], "jitter": jitter,
        "exposure": EXPOSURES[bg], "chamfer": chamfer, "facet_step": int(facet_step),
    }
    if index is not None:
        info["index"] = index
    with open(os.path.splitext(out)[0] + ".json", "w") as f:
        json.dump(info, f, indent=1)
    return info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model", help=".vxm file")
    ap.add_argument("--style", default="cubes", choices=STYLES)
    ap.add_argument("--light", default="studio", choices=LIGHTS)
    ap.add_argument("--view", default="three_quarter", choices=list(VIEWS))
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--samples", type=int, default=96)
    ap.add_argument("--out", required=True, help="output png (a .json side-car is written next to it)")
    ap.add_argument("--bg", default="dark", choices=BACKGROUNDS)
    ap.add_argument("--engine", default=None, help="override: CYCLES or EEVEE")
    ap.add_argument("--save-blend", default=None, help="also save the scene as this .blend")
    ap.add_argument("--index", type=int, default=None, help="tile order in the contact sheet")
    ap.add_argument("--cubic-materials", default="",
                    help="comma list of material ids kept as cubes in smooth/clay/toon (e.g. 5,8 = tooth, hair)")
    ap.add_argument("--jitter", type=float, default=None,
                    help="per-voxel colour jitter of the cube mesher (default 0.06, facet/mix styles 0; 0 = none)")
    ap.add_argument("--part-styles", default="", help="per-part overrides, e.g. head=smooth,hair=facet")
    ap.add_argument("--facet-step", type=int, default=1,
                    help="tread depth in voxels that counts as a step for facet (chamfer 0.5*N voxel) and slopes "
                         "(patterns scaled by N); default 1")
    ap.add_argument("--smooth-passes", type=int, default=SMOOTH_DEFAULTS["passes"],
                    help="Laplacian passes of the smooth parts (default 2)")
    ap.add_argument("--smooth-clamp", default=SMOOTH_DEFAULTS["clamp"], choices=SMOOTH_CLAMPS,
                    help="cell (vertices stay in their cell) or free (no clamp, volume restored)")
    ap.add_argument("--smooth-inflate", type=float, default=SMOOTH_DEFAULTS["inflate"],
                    help="offset of the smooth surface along the normals, in voxels (0..0.5)")
    ap.add_argument("--no-hermite", action="store_true",
                    help="binary surface nets even for parts that carry the sdf field")
    ap.add_argument("--hermite-passes", type=int, default=SMOOTH_DEFAULTS["hermite_passes"],
                    help="light cell-clamped Laplacian passes on top of the Hermite crossings (default 0)")
    args = ap.parse_args(argv)
    cubic = [int(v) for v in args.cubic_materials.split(",") if v.strip()]
    smooth = {"passes": args.smooth_passes, "clamp": args.smooth_clamp, "inflate": args.smooth_inflate,
              "hermite": False if args.no_hermite else None, "hermite_passes": args.hermite_passes}
    info = render_one(args.model, args.style, args.light, args.view, args.size, args.samples, args.out,
                      bg=args.bg, engine=args.engine, save_blend=args.save_blend, index=args.index,
                      cubic_materials=cubic, jitter=args.jitter, part_styles=parse_part_styles(args.part_styles),
                      smooth=smooth, facet_step=args.facet_step)
    print("RENDER_INFO " + json.dumps(info), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

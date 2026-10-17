"""Model sanity checks. ``run(model)`` returns ``{"errors": [...], "warnings": [...], "stats": {...}}``.

Checks (all numpy, no scipy):
  floating      solid 6-connected components not attached to the largest one nor touching another part (same frame)
                (error if a component has more than ``floating_error_vox`` voxels, else warning)
  budget        solid voxels per part vs ``part.budget`` (or ``default_budget``)
  palette_size  palette <= 256, colour indices in range, solid voxels with colour 0
  symmetry_x    for parts with ``symmetric=True``: percent of voxels whose material differs
                from the mirror voxel about local x=0 (hair ignored); warning / error thresholds
  layer_order   hair must be 6-adjacent to skin or hair; skin must not be buried in bone;
                bone / fat / muscle exposed to air -> warning (unless ``part.allow_exposed_bone``)
  hollow        enclosed air pockets (air components not reaching the grid border, hair
                counts as air) -> warning
"""

from __future__ import annotations

import numpy as np

from . import materials as M
from .body import exposed, shift
from .model import Model, Part

DEFAULT_BUDGET = 250_000
FLOATING_ERROR_VOX = 2
SYMMETRY_WARN_PCT = 0.5
SYMMETRY_ERROR_PCT = 10.0


# ---------------------------------------------------------------------------
# connected components (6-connectivity) by iterative label propagation
# ---------------------------------------------------------------------------

def label_components(mask: np.ndarray, max_iter: int = 100_000) -> tuple[np.ndarray, int]:
    """Label 6-connected components of a bool volume.

    Returns ``(labels, count)`` where labels is int32 with 0 for background and 1..count
    for components (ordered by size, largest first). Pure numpy: minimum-label propagation
    to the 6 neighbours plus pointer jumping, iterated until stable.
    """
    mask = np.asarray(mask, bool)
    n = mask.size
    if n == 0 or not mask.any():
        return np.zeros(mask.shape, np.int32), 0
    BIG = np.int64(n)
    idx = np.arange(n, dtype=np.int64).reshape(mask.shape)
    lab = np.where(mask, idx, BIG)
    for _ in range(max_iter):
        new = lab.copy()
        for dx, dy, dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            np.minimum(new, shift(lab, dx, dy, dz, fill=BIG), out=new)
        new = np.where(mask, new, BIG)
        # pointer jumping: follow labels to their current root a few times
        flat = new.ravel()
        valid = flat < BIG
        roots = flat.copy()
        for _ in range(3):
            r = roots[valid]
            roots[valid] = np.minimum(r, lab.ravel()[r])
            roots[valid] = np.minimum(roots[valid], new.ravel()[roots[valid]])
        new = roots.reshape(mask.shape)
        if np.array_equal(new, lab):
            break
        lab = new
    vals = lab[mask]
    uniq, inv, counts = np.unique(vals, return_inverse=True, return_counts=True)
    order = np.argsort(-counts, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(1, order.size + 1)
    out = np.zeros(mask.shape, np.int32)
    out[mask] = rank[inv]
    return out, int(uniq.size)


def component_sizes(labels: np.ndarray, count: int) -> np.ndarray:
    """Voxel count per component label 1..count (index 0 = background)."""
    return np.bincount(labels.ravel(), minlength=count + 1)


# ---------------------------------------------------------------------------
# individual checks: each appends to errors/warnings and returns stats
# ---------------------------------------------------------------------------

def _touches(mask: np.ndarray, other: np.ndarray) -> bool:
    """True if any voxel of ``mask`` shares a face with a voxel of ``other`` (same shape)."""
    for ax in range(3):
        for sh in (1, -1):
            a = np.roll(mask, sh, axis=ax)
            # roll wraps around; clear the wrapped slice
            idx = [slice(None)] * 3
            idx[ax] = slice(0, 1) if sh == 1 else slice(-1, None)
            a[tuple(idx)] = False
            if (a & other).any():
                return True
    return False


def other_parts_anchor(model: Model, part: Part) -> np.ndarray | None:
    """Solid voxels of the other parts resampled into ``part``'s voxel frame (same voxel size, any
    origin/dims: parts are cropped individually), or None when no other part overlaps it."""
    nz, ny, nx = part.solid().shape
    anchor = np.zeros((nz, ny, nx), dtype=bool)
    hit = False
    for q in model.parts:
        if q is part or q.count_solid() == 0 or abs(q.voxel_size - part.voxel_size) > 1e-9:
            continue
        # offset of q's voxel (0,0,0) inside part's frame, in voxels
        off = [int(round((q.origin[i] - part.origin[i]) / part.voxel_size)) for i in range(3)]
        qs = q.solid()
        qz, qy, qx = qs.shape
        # overlap in part coordinates
        x0, y0, z0 = max(off[0], 0), max(off[1], 0), max(off[2], 0)
        x1, y1, z1 = min(off[0] + qx, nx), min(off[1] + qy, ny), min(off[2] + qz, nz)
        if x0 >= x1 or y0 >= y1 or z0 >= z1:
            continue
        sub = qs[z0 - off[2]:z1 - off[2], y0 - off[1]:y1 - off[1], x0 - off[0]:x1 - off[0]]
        anchor[z0:z1, y0:y1, x0:x1] |= sub
        hit = hit or bool(sub.any())
    return anchor if hit else None


def check_floating(part: Part, errors: list, warnings: list, floating_error_vox: int = FLOATING_ERROR_VOX,
                   anchor: np.ndarray | None = None) -> dict:
    """``anchor``: solid voxels of the *other* parts in the same frame (same dims/origin/voxel size);
    a component of this part that shares a face with them is attached (hair on skin, a hat on
    hair), not floating, and is counted in ``attached`` instead."""
    solid = part.solid()
    labels, count = label_components(solid)
    sizes = component_sizes(labels, count)
    floating = []
    attached = 0
    for lab in range(2, count + 1):
        comp = labels == lab
        if anchor is not None and _touches(comp, anchor):
            attached += 1
            continue
        sz = int(sizes[lab])
        zz, yy, xx = np.nonzero(comp)
        at = (int(xx.min()), int(yy.min()), int(zz.min()))
        floating.append((sz, at))
    n_err = 0
    for sz, at in floating:
        msg = f"{part.name}: floating component of {sz} voxels near voxel {at}"
        if sz > floating_error_vox:
            errors.append(msg)
            n_err += 1
        else:
            warnings.append(msg)
    return {"components": count, "floating": len(floating), "floating_voxels": int(sum(s for s, _ in floating)),
            "attached": attached, "largest": int(sizes[1]) if count else 0}


def check_budget(part: Part, errors: list, warnings: list, default_budget: int = DEFAULT_BUDGET) -> dict:
    budget = part.budget if part.budget is not None else default_budget
    n = part.count_solid()
    if n > budget:
        errors.append(f"{part.name}: {n:,} solid voxels exceed the budget of {budget:,}")
    elif n > 0.9 * budget:
        warnings.append(f"{part.name}: {n:,} solid voxels, close to the budget of {budget:,}")
    return {"solid": n, "budget": int(budget), "cells": part.nx * part.ny * part.nz}


def check_palette(model: Model, errors: list, warnings: list) -> dict:
    n = len(model.palette)
    if n > 256:
        errors.append(f"palette has {n} colours (max 256)")
    used = set()
    for p in model.parts:
        solid = p.solid()
        cols = np.unique(p.color[solid])
        used.update(int(c) for c in cols)
        if (cols >= n).any():
            errors.append(f"{p.name}: colour indices {sorted(int(c) for c in cols[cols >= n])} outside the palette ({n} entries)")
        zero = int(np.count_nonzero(solid & (p.color == 0)))
        if zero:
            warnings.append(f"{p.name}: {zero} solid voxels have colour index 0 (air/black)")
    return {"palette_size": n, "colours_used": len(used)}


def _mirror_index_x(part: Part):
    """Index map x -> x' mirroring about local x = 0, or None if the plane is not aligned."""
    c = -part.origin[0] / part.voxel_size  # voxel coordinate of the x=0 plane
    k = round(c)
    if abs(c - k) > 1e-3:
        return None
    x = np.arange(part.nx)
    xm = 2 * k - 1 - x
    return xm


def check_symmetry_x(part: Part, errors: list, warnings: list, warn_pct: float = SYMMETRY_WARN_PCT,
                     error_pct: float = SYMMETRY_ERROR_PCT) -> dict:
    if not part.symmetric:
        return {"symmetric": False}
    mat = part.material.copy()
    mat[mat == M.MAT_HAIR] = 0  # hair is grown with randomness and is not required to be symmetric
    xm = _mirror_index_x(part)
    note = ""
    if xm is None:
        xm = part.nx - 1 - np.arange(part.nx)
        note = " (mirror plane not on a voxel boundary; mirrored about the grid centre)"
    valid = (xm >= 0) & (xm < part.nx)
    a = mat[:, :, valid]
    b = mat[:, :, xm[valid]]
    solid_any = (a != 0) | (b != 0)
    diff = int(np.count_nonzero((a != b) & solid_any))
    total = int(np.count_nonzero(solid_any))
    outside = int(np.count_nonzero(mat[:, :, ~valid]))
    pct = 100.0 * (diff + outside) / max(total + outside, 1)
    msg = f"{part.name}: {pct:.2f}% of voxels are not mirror-symmetric about x=0{note}"
    if pct > error_pct:
        errors.append(msg)
    elif pct > warn_pct:
        warnings.append(msg)
    return {"symmetric": True, "asymmetric_pct": round(pct, 3)}


def check_layer_order(part: Part, errors: list, warnings: list) -> dict:
    mat = part.material
    solid = mat != 0
    hist = part.material_histogram()
    stats = {}
    is_body = any(M.is_body(m) for m in hist)
    if not is_body:
        return {"body": False}
    air = ~solid
    hair = mat == M.MAT_HAIR
    skin = mat == M.MAT_SKIN
    # hair must touch skin or hair (6-connectivity)
    if hair.any():
        support = (skin | hair).astype(np.uint8)
        touch = np.zeros(mat.shape, np.uint8)
        for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            touch |= shift(support, *d)
        loose = hair & (touch == 0)
        n = int(np.count_nonzero(loose))
        stats["hair_unattached"] = n
        if n:
            errors.append(f"{part.name}: {n} hair voxels are not 6-adjacent to skin or hair")
    # skin buried in bone: all 6 neighbours bone
    if skin.any():
        bone = (mat == M.MAT_BONE).astype(np.uint8)
        nb = np.zeros(mat.shape, np.uint8)
        for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            nb += shift(bone, *d)
        buried = skin & (nb == 6)
        n = int(np.count_nonzero(buried))
        stats["skin_in_bone"] = n
        if n:
            errors.append(f"{part.name}: {n} skin voxels are fully enclosed by bone")
    # exposed inner tissues
    ex = exposed(solid, air)
    for mid, label in ((M.MAT_BONE, "bone"), (M.MAT_FAT, "fat"), (M.MAT_MUSCLE, "muscle"), (M.MAT_VESSEL, "vessel")):
        n = int(np.count_nonzero(ex & (mat == mid)))
        stats[f"{label}_exposed"] = n
        if n and not (mid == M.MAT_BONE and part.allow_exposed_bone):
            warnings.append(f"{part.name}: {n} {label} voxels are exposed to air")
    stats["body"] = True
    return stats


def check_hollow(part: Part, errors: list, warnings: list) -> dict:
    # hair columns are not body volume: gaps between them are not pockets
    air = (part.material == M.MAT_AIR) | (part.material == M.MAT_HAIR)
    labels, count = label_components(air)
    if count == 0:
        return {"pockets": 0, "pocket_voxels": 0}
    border = np.zeros(air.shape, bool)
    border[0, :, :] = border[-1, :, :] = True
    border[:, 0, :] = border[:, -1, :] = True
    border[:, :, 0] = border[:, :, -1] = True
    outside_labels = np.unique(labels[border & air])
    sizes = component_sizes(labels, count)
    pockets = [(lab, int(sizes[lab])) for lab in range(1, count + 1) if lab not in set(outside_labels.tolist())]
    n_vox = sum(s for _, s in pockets)
    if pockets:
        warnings.append(f"{part.name}: {len(pockets)} enclosed air pocket(s), {n_vox} voxels")
    return {"pockets": len(pockets), "pocket_voxels": int(n_vox)}


# ---------------------------------------------------------------------------

def run(model: Model, floating_error_vox: int = FLOATING_ERROR_VOX, default_budget: int = DEFAULT_BUDGET,
        which=None) -> dict:
    """Run all checks on a model. ``which`` limits to a subset of names
    (floating, budget, palette_size, symmetry_x, layer_order, hollow)."""
    errors: list[str] = []
    warnings: list[str] = []
    stats: dict = {"model": model.name, "parts": {}, "solid_total": model.count_solid()}
    which = set(which) if which else None

    def want(name):
        return which is None or name in which

    if want("palette_size"):
        stats["palette"] = check_palette(model, errors, warnings)
    for part in model.parts:
        ps: dict = {"dims": list(part.dims), "voxel_size": part.voxel_size, "origin": list(part.origin),
                    "bone": part.bone, "solid": part.count_solid(),
                    "materials": {M.name_of(k): v for k, v in part.material_histogram().items()}}
        if part.count_solid() == 0:
            errors.append(f"{part.name}: part is empty")
            stats["parts"][part.name] = ps
            continue
        if want("budget"):
            ps["budget"] = check_budget(part, errors, warnings, default_budget)
        if want("floating"):
            ps["floating"] = check_floating(part, errors, warnings, floating_error_vox, other_parts_anchor(model, part))
        if want("symmetry_x"):
            ps["symmetry_x"] = check_symmetry_x(part, errors, warnings)
        if want("layer_order"):
            ps["layer_order"] = check_layer_order(part, errors, warnings)
        if want("hollow"):
            ps["hollow"] = check_hollow(part, errors, warnings)
        stats["parts"][part.name] = ps
    if not model.parts:
        errors.append("model has no parts")
    return {"errors": errors, "warnings": warnings, "stats": stats}


def summary(report: dict) -> str:
    """One-line human summary of a report."""
    return f"{len(report['errors'])} error(s), {len(report['warnings'])} warning(s)"

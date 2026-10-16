"""Command line interface (docs/spec/voxgen-cli.md).

    voxgen build <spec.py> [--out DIR] [--preview DIR] [--vox DIR] [--report DIR] [--no-check] [--size N]
    voxgen build-all [specs-dir] [same options]
    voxgen preview <file.vxm> [--out x.png] [--views a,b,c] [--size 256]
    voxgen check <file.vxm>
    voxgen import-vox <file.vox> --spec-json <file.vox.json> --out <file.vxm>
    voxgen info <file.vxm>

Exit code 1 when checks report errors (build, build-all, check) or on failure.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path

from . import __version__, checks, io_vox, io_vxm
from . import materials as M
from .model import Model
from .render import SLICE_VIEWS, VIEW_NAMES, contact_sheet, render_views

DEFAULT_OUT = "godot/assets/models"
DEFAULT_PREVIEW = "build/previews"
DEFAULT_VOX = "build/vox"
DEFAULT_REPORT = "build/reports"


# ---------------------------------------------------------------------------
# spec loading
# ---------------------------------------------------------------------------

def specs_root(path: Path) -> Path | None:
    """Nearest ancestor directory named ``specs`` (so ``calibration/head_3mm.py`` can
    ``from _human_head import build_head``)."""
    for parent in [path] + list(path.parents):
        if parent.name == "specs" and parent.is_dir():
            return parent
    return None


def load_spec(path) -> Model:
    """Import a spec module from a file path and call its ``build()``."""
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    roots = [path.parent]
    root = specs_root(path.parent)
    if root is not None and root not in roots:
        roots.append(root)
    for r in reversed(roots):
        rs = str(r)
        if rs not in sys.path:
            sys.path.insert(0, rs)
    mod_name = "voxgen_spec_" + path.stem.replace("-", "_")
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "build"):
        raise AttributeError(f"{path} has no build() function")
    model = module.build()
    if not isinstance(model, Model):
        raise TypeError(f"{path}: build() must return voxgen.Model, got {type(model).__name__}")
    return model


def find_specs(root) -> list[Path]:
    """All ``*.py`` under ``root`` whose file name (and no parent dir) starts with ``_``."""
    root = Path(root)
    out = []
    for p in sorted(root.rglob("*.py")):
        rel = p.relative_to(root)
        if any(part.startswith("_") for part in rel.parts):
            continue
        out.append(p)
    return out


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def build_one(spec_path, out_dir, preview_dir, vox_dir, report_dir, check: bool = True, size: int = 256,
              quiet: bool = False) -> tuple[Model, dict]:
    """Build one spec: write .vxm, .vox(+.json), preview .png and report .json.
    Returns ``(model, report)``; report["errors"] is empty when the model is fine."""
    spec_path = Path(spec_path)
    t0 = time.perf_counter()
    model = load_spec(spec_path)
    t_build = time.perf_counter() - t0
    try:
        spec_rel = str(spec_path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        spec_rel = str(spec_path)
    report = {"errors": [], "warnings": [], "stats": {}}
    if check:
        t1 = time.perf_counter()
        report = checks.run(model)
        report["stats"]["check_seconds"] = round(time.perf_counter() - t1, 3)
    report["stats"]["build_seconds"] = round(t_build, 3)
    report["spec"] = spec_rel
    report["model"] = model.name
    outputs = {}
    if out_dir:
        outputs["vxm"] = str(io_vxm.write(model, Path(out_dir) / f"{model.name}.vxm", spec=spec_rel))
    if vox_dir:
        vp, jp = io_vox.write(model, Path(vox_dir) / f"{model.name}.vox")
        outputs["vox"] = str(vp)
        outputs["vox_json"] = str(jp)
    if preview_dir:
        notes = [f"checks: {checks.summary(report)}" if check else "checks: skipped"]
        notes += [f"  ERROR: {e}" for e in report["errors"][:4]]
        notes += [f"  warning: {w}" for w in report["warnings"][:3]]
        sheet = contact_sheet(model, size=size, notes=notes)
        pp = Path(preview_dir) / f"{model.name}.png"
        pp.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(pp)
        outputs["preview"] = str(pp)
    if report_dir:
        rp = Path(report_dir) / f"{model.name}.json"
        rp.parent.mkdir(parents=True, exist_ok=True)
        report["outputs"] = outputs
        rp.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
        outputs["report"] = str(rp)
    report["outputs"] = outputs
    if not quiet:
        parts = ", ".join(f"{p.name} {p.nx}x{p.ny}x{p.nz}@{p.voxel_size * 1000:g}mm" for p in model.parts)
        print(f"[{model.name}] {parts}; {model.count_solid():,} voxels; build {t_build:.2f}s; "
              f"{checks.summary(report) if check else 'checks skipped'}")
        for e in report["errors"]:
            print(f"  ERROR: {e}")
        for w in report["warnings"]:
            print(f"  warning: {w}")
        for k, v in outputs.items():
            print(f"  {k}: {v}")
    return model, report


def cmd_build(args) -> int:
    try:
        _, report = build_one(args.spec, args.out, args.preview, args.vox, args.report,
                              check=not args.no_check, size=args.size)
    except Exception:
        traceback.print_exc()
        return 1
    return 1 if report["errors"] else 0


def cmd_build_all(args) -> int:
    root = Path(args.specs_dir)
    if not root.is_dir():
        print(f"specs dir not found: {root}", file=sys.stderr)
        return 1
    specs = find_specs(root)
    if not specs:
        print(f"no specs under {root}", file=sys.stderr)
        return 1
    failed = []
    rows = []
    for sp in specs:
        try:
            model, report = build_one(sp, args.out, args.preview, args.vox, args.report,
                                      check=not args.no_check, size=args.size)
            rows.append((sp, model.name, len(report["errors"]), len(report["warnings"])))
            if report["errors"]:
                failed.append(sp)
        except Exception:
            traceback.print_exc()
            rows.append((sp, "?", -1, 0))
            failed.append(sp)
    print()
    print(f"{'spec':40} {'model':16} {'errors':>6} {'warnings':>8}")
    for sp, name, ne, nw in rows:
        rel = str(sp.relative_to(root))
        print(f"{rel:40} {name:16} {('FAIL' if ne < 0 else ne):>6} {nw:>8}")
    print(f"\n{len(specs) - len(failed)}/{len(specs)} specs ok")
    return 1 if failed else 0


# ---------------------------------------------------------------------------
# preview / check / info / import-vox
# ---------------------------------------------------------------------------

def _load_any(path) -> Model:
    path = Path(path)
    if path.suffix == ".py":
        return load_spec(path)
    if path.suffix == ".vox":
        return io_vox.import_vox(path)
    return io_vxm.read(path)


def cmd_preview(args) -> int:
    model = _load_any(args.file)
    views = [v.strip() for v in args.views.split(",")] if args.views else None
    if views:
        bad = [v for v in views if v not in VIEW_NAMES + SLICE_VIEWS]
        if bad:
            print(f"unknown views {bad}; known: {VIEW_NAMES + SLICE_VIEWS}", file=sys.stderr)
            return 1
    out = Path(args.out) if args.out else Path(DEFAULT_PREVIEW) / f"{model.name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.part:
        obj = model.part(args.part)
    else:
        obj = model
    if views and len(views) == 1:
        img = render_views(obj, views, size=args.size)[views[0]]
        img.save(out)
    else:
        contact_sheet(obj, size=args.size, views=views).save(out)
    print(out)
    return 0


def cmd_check(args) -> int:
    model = _load_any(args.file)
    report = checks.run(model)
    print(json.dumps(report, indent=1, ensure_ascii=False))
    return 1 if report["errors"] else 0


def cmd_info(args) -> int:
    model = _load_any(args.file)
    print(f"model: {model.name}")
    print(f"voxel_size: {model.voxel_size} m")
    print(f"palette: {len(model.palette)} colours")
    if model.meta:
        print(f"meta: {json.dumps(model.meta, ensure_ascii=False)}")
    total = 0
    for p in model.parts:
        bb = p.bbox()
        print(f"part {p.name!r} (bone {p.bone!r}): dims {p.nx}x{p.ny}x{p.nz}, voxel {p.voxel_size * 1000:g} mm, "
              f"origin {tuple(round(v, 4) for v in p.origin)}, solid {p.count_solid():,} / {p.nx * p.ny * p.nz:,}"
              + (f", bbox {bb[0]}..{bb[1]}" if bb else ""))
        for mid, c in sorted(p.material_histogram().items()):
            print(f"    {M.name_of(mid):10} ({mid:2d}): {c:,}")
        total += p.count_solid()
    print(f"total solid voxels: {total:,}")
    return 0


def cmd_import_vox(args) -> int:
    model = io_vox.import_vox(args.file, args.spec_json, name=args.name)
    out = Path(args.out) if args.out else Path(args.file).with_suffix(".vxm")
    io_vxm.write(model, out, spec=f"import-vox {args.file}")
    print(f"{out}: {model.count_solid():,} voxels in {len(model.parts)} part(s)")
    return 0


# ---------------------------------------------------------------------------

def _add_build_opts(p):
    p.add_argument("--out", default=DEFAULT_OUT, help=f".vxm output dir (default {DEFAULT_OUT})")
    p.add_argument("--preview", default=DEFAULT_PREVIEW, help=f"preview png dir (default {DEFAULT_PREVIEW})")
    p.add_argument("--vox", default=DEFAULT_VOX, help=f".vox output dir (default {DEFAULT_VOX})")
    p.add_argument("--report", default=DEFAULT_REPORT, help=f"check report dir (default {DEFAULT_REPORT})")
    p.add_argument("--no-check", action="store_true", help="skip checks")
    p.add_argument("--size", type=int, default=256, help="preview size per view in px (default 256)")


def make_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="voxgen", description=f"voxgen {__version__}: voxel model generator")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("build", help="build one spec")
    p.add_argument("spec")
    _add_build_opts(p)
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("build-all", help="build every spec under a directory (files starting with _ are skipped)")
    p.add_argument("specs_dir", nargs="?", default="tools/voxgen/specs")
    _add_build_opts(p)
    p.set_defaults(func=cmd_build_all)

    p = sub.add_parser("preview", help="render a contact sheet of a .vxm (or .vox / spec .py)")
    p.add_argument("file")
    p.add_argument("--out", default=None, help="output png (default build/previews/<name>.png)")
    p.add_argument("--views", default=None, help="comma list: " + ",".join(VIEW_NAMES + SLICE_VIEWS))
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--part", default=None, help="render only this part")
    p.set_defaults(func=cmd_preview)

    p = sub.add_parser("check", help="run checks on a .vxm and print the json report")
    p.add_argument("file")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("import-vox", help="convert a MagicaVoxel .vox (+ .vox.json) to .vxm")
    p.add_argument("file")
    p.add_argument("--spec-json", default=None, help="side-car json written by voxgen (default: <file>.json)")
    p.add_argument("--out", default=None, help="output .vxm (default: next to the .vox)")
    p.add_argument("--name", default=None, help="override the model name")
    p.set_defaults(func=cmd_import_vox)

    p = sub.add_parser("info", help="print parts, dims, voxel counts and materials of a .vxm")
    p.add_argument("file")
    p.set_defaults(func=cmd_info)
    return ap


def main(argv=None) -> int:
    args = make_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except BrokenPipeError:
        # `voxgen info x | head` closed the pipe early: not an error, exit quietly.
        try:
            sys.stdout.close()
        except Exception:
            pass
        return 0
    except Exception as e:  # keep the exit code meaningful for CI
        if args.cmd in ("build", "build-all"):
            raise
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

import json
import subprocess
import sys
from pathlib import Path

from voxgen import cli, io_vxm

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "specs" / "head_test.py"


def test_build_head_test_into_tmp(tmp_path):
    out, prev, vox, rep = (tmp_path / d for d in ("models", "previews", "vox", "reports"))
    code = cli.main(["build", str(SPEC), "--out", str(out), "--preview", str(prev), "--vox", str(vox),
                     "--report", str(rep)])
    assert code == 0
    assert (out / "head_test.vxm").exists()
    assert (vox / "head_test.vox").exists() and (vox / "head_test.vox.json").exists()
    assert (prev / "head_test.png").exists()
    report = json.loads((rep / "head_test.json").read_text())
    assert report["errors"] == []
    model = io_vxm.read(out / "head_test.vxm")
    assert [p.name for p in model.parts] == ["head"]
    head = model.parts[0]
    assert head.bone == "head" and head.voxel_size == 0.005
    assert 40 <= head.nx <= 64 and 40 <= head.ny <= 64 and 40 <= head.nz <= 64
    hist = head.material_histogram()
    for mid in (1, 2, 3, 4, 5, 6, 7, 8, 9):  # skin fat muscle bone tooth cartilage vessel hair eye
        assert hist.get(mid, 0) > 0, mid
    # the chin is near y = 0 and the head is centred on x/z
    assert head.origin[0] == -head.nx * head.voxel_size / 2
    assert head.origin[2] == -head.nz * head.voxel_size / 2
    assert -0.06 <= head.origin[1] <= 0.0


def test_cli_subcommands_via_module(tmp_path):
    out = tmp_path / "models"
    r = subprocess.run([sys.executable, "-m", "voxgen", "build", str(SPEC), "--out", str(out), "--preview",
                        str(tmp_path / "p"), "--vox", str(tmp_path / "v"), "--report", str(tmp_path / "r"),
                        "--size", "64"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    vxm = out / "head_test.vxm"
    r = subprocess.run([sys.executable, "-m", "voxgen", "info", str(vxm)], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0 and "part 'head'" in r.stdout and "hair" in r.stdout
    r = subprocess.run([sys.executable, "-m", "voxgen", "check", str(vxm)], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0 and json.loads(r.stdout)["errors"] == []
    png = tmp_path / "x.png"
    r = subprocess.run([sys.executable, "-m", "voxgen", "preview", str(vxm), "--out", str(png), "--views",
                        "front,iso", "--size", "64"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0 and png.exists()
    back = tmp_path / "back.vxm"
    r = subprocess.run([sys.executable, "-m", "voxgen", "import-vox", str(tmp_path / "v" / "head_test.vox"),
                        "--spec-json", str(tmp_path / "v" / "head_test.vox.json"), "--out", str(back)],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert io_vxm.read(back).parts[0].count_solid() == io_vxm.read(vxm).parts[0].count_solid()


def test_sdf_command_and_build_sdf(tmp_path):
    from voxgen import sdfield
    out = tmp_path / "models"
    assert cli.main(["build", str(SPEC), "--out", str(out), "--preview", "", "--vox", "", "--report", "",
                     "--no-check"]) == 0
    vxm = out / "head_test.vxm"
    assert io_vxm.read(vxm).parts[0].sdf is None
    with_sdf = tmp_path / "head_sdf.vxm"
    assert cli.main(["sdf", str(vxm), "--out", str(with_sdf)]) == 0
    head = io_vxm.read(with_sdf).parts[0]
    assert head.fields == ["material", "color", "state", "sdf"] and sdfield.is_consistent(head.sdf, head.solid())
    assert io_vxm.read(vxm).parts[0].sdf is None                  # the input was left alone
    assert cli.main(["build", str(SPEC), "--out", str(out), "--preview", "", "--vox", "", "--report", "",
                     "--no-check", "--sdf"]) == 0
    assert io_vxm.read(vxm).parts[0].sdf is not None
    r = subprocess.run([sys.executable, "-m", "voxgen", "info", str(vxm)], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0 and "fields material,color,state,sdf" in r.stdout


def test_build_all_skips_underscore_specs():
    specs = cli.find_specs(ROOT / "specs")
    names = [p.name for p in specs]
    assert "head_test.py" in names and "_human_head.py" not in names
    assert any(p.parent.name == "calibration" for p in specs) and any(p.parent.name == "props" for p in specs)


def test_build_exit_code_on_errors(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text(
        "from voxgen import Model, Part, Palette\n"
        "from voxgen.materials import MAT_SKIN\n"
        "def build():\n"
        "    pal = Palette(); c = pal.add('skin', (200,150,120))\n"
        "    m = Model('bad', 0.01, pal); p = m.add_part(Part('p', (10,10,10), 0.01))\n"
        "    p.material[1:4,1:4,1:4] = MAT_SKIN; p.color[1:4,1:4,1:4] = c\n"
        "    p.material[8,8,6:9] = MAT_SKIN; p.color[8,8,6:9] = c   # 3 loose voxels -> error\n"
        "    return m\n")
    code = cli.main(["build", str(bad), "--out", str(tmp_path / "o"), "--preview", str(tmp_path / "p"),
                     "--vox", str(tmp_path / "v"), "--report", str(tmp_path / "r"), "--size", "32"])
    assert code == 1
    assert cli.main(["build", str(bad), "--out", str(tmp_path / "o"), "--preview", str(tmp_path / "p"),
                     "--vox", str(tmp_path / "v"), "--report", str(tmp_path / "r"), "--size", "32",
                     "--no-check"]) == 0

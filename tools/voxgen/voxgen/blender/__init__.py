"""Blender (bpy) beauty renders of .vxm models.

``render_styles`` needs the bpy venv (``scripts/install_blender.sh``) and is run as a subprocess
by ``voxgen render``; ``sheet`` only needs Pillow. The names below are shared by the CLI
and the renderer so the CLI can validate arguments without importing bpy.
"""

STYLES = ("cubes", "bevel", "smooth", "clay", "toon")
LIGHTS = ("studio", "candle")
BACKGROUNDS = ("dark", "grey")
# view name -> (azimuth deg, elevation deg, lens mm, frame margin)
VIEWS = {
    "three_quarter": (33.0, 11.0, 50.0, 1.0),
    "front": (0.0, 4.0, 60.0, 1.0),
    "profile": (90.0, 4.0, 60.0, 1.0),
    "full": (30.0, 18.0, 35.0, 1.25),
}

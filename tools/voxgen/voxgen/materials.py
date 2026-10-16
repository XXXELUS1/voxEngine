"""Material ids. The single source of truth is docs/spec/vxm-format.md; this table is
mirrored by ``enum Material`` in src/voxbody/src/voxel_volume.h. Change both together.

Body materials use ids 1..31, props 32..62, 63 is generic. 0 is air.
"""

MAT_AIR = 0
MAT_SKIN = 1
MAT_FAT = 2
MAT_MUSCLE = 3
MAT_BONE = 4
MAT_TOOTH = 5
MAT_CARTILAGE = 6
MAT_VESSEL = 7
MAT_HAIR = 8
MAT_EYE = 9
MAT_NAIL = 10

MAT_WOOD = 32
MAT_METAL = 33
MAT_STONE = 34
MAT_CLOTH = 35
MAT_LEATHER = 36
MAT_GLASS = 37
MAT_CERAMIC = 38
MAT_LIQUID = 39
MAT_PLANT = 40

MAT_GENERIC = 63

NAMES = {
    MAT_AIR: "air",
    MAT_SKIN: "skin",
    MAT_FAT: "fat",
    MAT_MUSCLE: "muscle",
    MAT_BONE: "bone",
    MAT_TOOTH: "tooth",
    MAT_CARTILAGE: "cartilage",
    MAT_VESSEL: "vessel",
    MAT_HAIR: "hair",
    MAT_EYE: "eye",
    MAT_NAIL: "nail",
    MAT_WOOD: "wood",
    MAT_METAL: "metal",
    MAT_STONE: "stone",
    MAT_CLOTH: "cloth",
    MAT_LEATHER: "leather",
    MAT_GLASS: "glass",
    MAT_CERAMIC: "ceramic",
    MAT_LIQUID: "liquid",
    MAT_PLANT: "plant",
    MAT_GENERIC: "generic",
}

IDS = {name: mid for mid, name in NAMES.items()}

#: materials that bleed when cut (mirrors VoxelVolume::material_bleeds)
BLEEDS = frozenset({MAT_SKIN, MAT_FAT, MAT_MUSCLE, MAT_VESSEL})
#: materials that can take a bruise (mirrors VoxelVolume::material_bruises)
BRUISES = frozenset({MAT_SKIN})
#: "hard" materials: bone and teeth
HARD = frozenset({MAT_BONE, MAT_TOOTH})

#: Fallback sRGB colours per material. Only used when a Part is rendered or exported
#: without a palette (e.g. a synthetic test volume). Real models define their own palette.
DEFAULT_COLORS = {
    MAT_AIR: (0, 0, 0),
    MAT_SKIN: (224, 180, 150),
    MAT_FAT: (238, 214, 140),
    MAT_MUSCLE: (150, 40, 45),
    MAT_BONE: (235, 225, 200),
    MAT_TOOTH: (245, 240, 220),
    MAT_CARTILAGE: (225, 185, 160),
    MAT_VESSEL: (120, 20, 30),
    MAT_HAIR: (60, 40, 30),
    MAT_EYE: (240, 240, 235),
    MAT_NAIL: (230, 210, 190),
    MAT_WOOD: (150, 105, 60),
    MAT_METAL: (170, 175, 180),
    MAT_STONE: (120, 118, 110),
    MAT_CLOTH: (120, 90, 70),
    MAT_LEATHER: (100, 65, 40),
    MAT_GLASS: (180, 210, 220),
    MAT_CERAMIC: (205, 195, 180),
    MAT_LIQUID: (80, 120, 160),
    MAT_PLANT: (80, 130, 60),
    MAT_GENERIC: (160, 160, 160),
}


def name_of(material_id: int) -> str:
    """Human-readable name for a material id ("mat37" for unknown ids)."""
    return NAMES.get(int(material_id), f"mat{int(material_id)}")


def id_of(name: str) -> int:
    """Material id by name (case-insensitive). Raises KeyError for unknown names."""
    return IDS[name.lower()]


def is_body(material_id: int) -> bool:
    """True for body-tissue materials (ids 1..31)."""
    return 1 <= int(material_id) <= 31


def is_prop(material_id: int) -> bool:
    """True for prop/environment materials (ids 32..62)."""
    return 32 <= int(material_id) <= 62

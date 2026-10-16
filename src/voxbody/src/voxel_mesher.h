#pragma once
// VoxelMesher: turns a VoxelVolume into an ArrayMesh.
//
// Greedy meshing on all six face directions, one surface, PRIMITIVE_TRIANGLES
// with indices. Per-vertex ambient occlusion uses the classic "0fps" rule
// (three neighbours per corner, four levels). Quads are merged only when
// (color, material, state, ao0..ao3) are equal, so the AO stays exact.
//
// Vertex layout (see docs/spec/voxbody-api.md):
//   ARRAY_VERTEX : local metres, origin + voxel * voxel_size
//   ARRAY_NORMAL : face normal
//   ARRAY_COLOR  : rgb = sRGB albedo after the state tint, a = AO factor
//   ARRAY_TEX_UV : (material / 255, state / 255)
//   ARRAY_INDEX  : triangles, Godot winding (clockwise front faces)
//
// The mesher is a plain static function on a RefCounted so GDScript can call
// VoxelMesher.build_mesh(...) without instancing anything.

#include <godot_cpp/classes/array_mesh.hpp>
#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/variant/color.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/packed_color_array.hpp>

#include "voxel_volume.h"

namespace godot {

class VoxelMesher : public RefCounted {
	GDCLASS(VoxelMesher, RefCounted)

protected:
	static void _bind_methods();

public:
	// options: { "ao": bool = true, "ao_strength": float = 0.45 }
	// Returns an ArrayMesh with no surfaces for an empty or null volume.
	static Ref<ArrayMesh> build_mesh(const Ref<VoxelVolume> &volume, const PackedColorArray &palette, const Dictionary &options = Dictionary());

	// sRGB albedo of a voxel after applying the state tint table from the spec.
	// Exposed so GDScript/UI and tests can reproduce the mesher's colours.
	static Color tint_color(const Color &base, int state);

	// AO factor for a corner level 0..3 (0 = fully occluded) and a strength.
	static float ao_factor(int level, float ao_strength);
};

} // namespace godot

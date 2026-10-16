#pragma once
// VoxelVolume: a dense block of voxels for one body part or one prop.
//
// Storage is three byte fields of size nx*ny*nz, x fastest:
//   material - what the voxel is made of (0 = air). See voxgen/materials.py for ids.
//   color    - palette index (0..255) used for rendering.
//   state    - damage flags, see STATE_* below.
//
// Coordinates: "voxel space" is integer/float coordinates in voxels, with
// (0,0,0) at the corner of voxel 0. "local space" (metres) is handled by VoxelPart:
//   local = origin + voxel * voxel_size.
//
// All operations here are integer-deterministic: the same call on the same
// volume gives the same result on every machine. That is what lets networking
// send operations instead of voxels.

#include <godot_cpp/classes/resource.hpp>
#include <godot_cpp/variant/packed_byte_array.hpp>
#include <godot_cpp/variant/vector3.hpp>
#include <godot_cpp/variant/vector3i.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/typed_array.hpp>

#include <cstdint>
#include <vector>

namespace godot {

class VoxelVolume : public Resource {
	GDCLASS(VoxelVolume, Resource)

public:
	// Material ids shared with tools/voxgen/voxgen/materials.py. Keep in sync.
	enum Material {
		MAT_AIR = 0,
		MAT_SKIN = 1,
		MAT_FAT = 2,
		MAT_MUSCLE = 3,
		MAT_BONE = 4,
		MAT_TOOTH = 5,
		MAT_CARTILAGE = 6,
		MAT_VESSEL = 7,
		MAT_HAIR = 8,
		MAT_EYE = 9,
		MAT_NAIL = 10,
		// 32+ : props / environment
		MAT_WOOD = 32,
		MAT_METAL = 33,
		MAT_STONE = 34,
		MAT_CLOTH = 35,
		MAT_LEATHER = 36,
		MAT_GLASS = 37,
		MAT_CERAMIC = 38,
		MAT_LIQUID = 39,
		MAT_PLANT = 40,
		MAT_GENERIC = 63,
	};

	// state byte layout
	enum State {
		STATE_BRUISE_MASK = 0x03, // 0..3, bruise level (red -> purple -> yellow-green)
		STATE_BURN_SHIFT = 2,
		STATE_BURN_MASK = 0x0C, // 0..3, burn level (red -> blister -> char)
		STATE_CUT = 0x10, // freshly exposed surface (wound)
		STATE_BLOOD = 0x20, // blood on the surface
		STATE_WET = 0x40,
	};

private:
	int nx = 0, ny = 0, nz = 0;
	float voxel_size = 0.005f;
	Vector3 origin; // local-space (metres) position of the corner of voxel (0,0,0)
	String part_name;
	String bone;
	PackedByteArray material;
	PackedByteArray color;
	PackedByteArray state;

	inline int idx(int x, int y, int z) const { return x + nx * (y + ny * z); }
	inline bool in_bounds(int x, int y, int z) const {
		return x >= 0 && y >= 0 && z >= 0 && x < nx && y < ny && z < nz;
	}
	inline bool solid_at(int x, int y, int z) const {
		return in_bounds(x, y, z) && material[idx(x, y, z)] != 0;
	}
	static bool material_bleeds(int m);
	static bool material_bruises(int m);
	int mark_exposed(const std::vector<int> &removed_indices, bool bleed);

protected:
	static void _bind_methods();

public:
	VoxelVolume();

	void create(Vector3i p_dims, float p_voxel_size);
	Ref<VoxelVolume> duplicate_volume() const;

	Vector3i get_dims() const { return Vector3i(nx, ny, nz); }
	// Re-creates the volume (clears all voxels) when the size changes. Used by
	// the storage property so a saved resource restores dims before its data.
	void set_dims(Vector3i p_dims);
	int get_voxel_count() const { return nx * ny * nz; }
	void set_voxel_size(float v) { voxel_size = v; }
	float get_voxel_size() const { return voxel_size; }
	void set_origin(Vector3 v) { origin = v; }
	Vector3 get_origin() const { return origin; }
	void set_part_name(const String &v) { part_name = v; }
	String get_part_name() const { return part_name; }
	void set_bone(const String &v) { bone = v; }
	String get_bone() const { return bone; }

	bool is_inside(Vector3i v) const { return in_bounds(v.x, v.y, v.z); }
	int get_material(Vector3i v) const { return in_bounds(v.x, v.y, v.z) ? material[idx(v.x, v.y, v.z)] : 0; }
	int get_color(Vector3i v) const { return in_bounds(v.x, v.y, v.z) ? color[idx(v.x, v.y, v.z)] : 0; }
	int get_state(Vector3i v) const { return in_bounds(v.x, v.y, v.z) ? state[idx(v.x, v.y, v.z)] : 0; }
	void set_voxel(Vector3i v, int p_material, int p_color);
	void set_state(Vector3i v, int p_state);

	// Raw field access (copies). Dims must match when setting.
	PackedByteArray get_material_data() const { return material; }
	void set_material_data(const PackedByteArray &d);
	PackedByteArray get_color_data() const { return color; }
	void set_color_data(const PackedByteArray &d);
	PackedByteArray get_state_data() const { return state; }
	void set_state_data(const PackedByteArray &d);
	// Direct pointers for the mesher (no copy).
	const uint8_t *material_ptr() const { return material.ptr(); }
	const uint8_t *color_ptr() const { return color.ptr(); }
	const uint8_t *state_ptr() const { return state.ptr(); }
	int width() const { return nx; }
	int height() const { return ny; }
	int depth() const { return nz; }

	int count_solid() const;
	// Bounding box of solid voxels (inclusive). For an empty volume
	// get_solid_min() returns dims and get_solid_max() returns (-1,-1,-1).
	Vector3i get_solid_min() const;
	Vector3i get_solid_max() const;

	// --- damage operations (voxel space) --------------------------------
	// Remove every solid voxel whose centre is within `radius` of the segment a-b.
	// Newly exposed voxels get STATE_CUT (+STATE_BLOOD if the material bleeds).
	int carve_capsule(Vector3 a, Vector3 b, float radius, bool mark_wound = true);
	int carve_sphere(Vector3 center, float radius, bool mark_wound = true);
	// Raise bruise level to at least `level` for bruisable surface voxels within radius.
	int bruise_sphere(Vector3 center, float radius, int level);
	// Raise burn level; level >= 2 also burns hair away (see vxm-format.md).
	int burn_sphere(Vector3 center, float radius, int level);
	// Set/clear state bits in a sphere, on solid voxels only.
	int stamp_state_sphere(Vector3 center, float radius, int set_bits, int clear_bits);
	// Remove hair voxels within radius whose "height above skin" exceeds keep_length.
	// (hair columns are MAT_HAIR voxels stacked outward from skin; this is a haircut)
	int trim_hair_capsule(Vector3 a, Vector3 b, float radius, int keep_length);

	// DDA ray cast in voxel space. Returns {hit, voxel: Vector3i, normal: Vector3i,
	// distance: float (voxel units), material: int, position: Vector3 (voxel units)}.
	// The origin may lie outside the volume; the ray is clipped to the box and
	// the entry face becomes the normal. normal is (0,0,0) only when the ray
	// starts inside a solid voxel.
	Dictionary raycast(Vector3 origin_vox, Vector3 dir, float max_dist) const;

	// Flood fill (6-connectivity) from `anchor`. Everything NOT connected to the
	// anchor component is cut out of this volume and returned as new volumes
	// (one per connected piece), each with its own origin so that
	// origin + voxel*size stays in the same local space. If `anchor` is empty,
	// the largest component is kept instead.
	TypedArray<VoxelVolume> detach_disconnected(Vector3i anchor);
};

} // namespace godot

VARIANT_ENUM_CAST(godot::VoxelVolume::Material);
VARIANT_ENUM_CAST(godot::VoxelVolume::State);

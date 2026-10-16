#pragma once
// VoxelPart: one body part / prop on the scene. Owns a VoxelVolume, keeps an
// internal MeshInstance3D child and rebuilds its mesh whenever the volume is
// marked dirty (in _process when auto_rebuild is on, or via rebuild()).
//
// All damage methods take local-space metres and convert to voxel space
// (local = origin + voxel * voxel_size) before calling the volume, so
// GDScript never has to know about voxel units.
//
// Material resolution (see docs/spec/voxbody-api.md):
//   material_override if set, else res://addons/voxbody/voxel_material.tres,
//   else a StandardMaterial3D with vertex colours as sRGB albedo.
//   If the chosen material is a ShaderMaterial with a "voxel_size" uniform,
//   the uniform must equal the volume's voxel_size; when it differs the
//   material is duplicated for this part so other parts keep their own value.
//
// Decisions where the spec is silent:
//   * the MeshInstance3D child is created lazily (first rebuild / ready /
//     get_mesh_instance) as an INTERNAL_MODE_BACK child named "VoxelMesh",
//   * raycast_global accounts for node scale: distances come back in metres
//     along the global ray,
//   * local_to_voxel / voxel_to_local are identity when no volume is set.

#include <godot_cpp/classes/material.hpp>
#include <godot_cpp/classes/mesh_instance3d.hpp>
#include <godot_cpp/classes/node3d.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/packed_color_array.hpp>
#include <godot_cpp/variant/typed_array.hpp>

#include "voxel_volume.h"

namespace godot {

class VoxelPart : public Node3D {
	GDCLASS(VoxelPart, Node3D)

private:
	Ref<VoxelVolume> volume;
	PackedColorArray palette;
	Ref<Material> material_override;
	bool auto_rebuild = true;
	bool ao_enabled = true;

	MeshInstance3D *mesh_instance = nullptr;
	uint64_t mesh_instance_id = 0; // to notice if someone freed the child behind our back
	bool dirty = true;
	// Material actually assigned to the mesh instance (may be a per-part duplicate).
	Ref<Material> active_material;
	// What active_material was derived from, to detect when it must be refreshed.
	Ref<Material> active_source;
	float active_voxel_size = -1.0f;
	// StandardMaterial3D used when voxel_material.tres is missing; created once per part.
	Ref<Material> fallback_material;

	void ensure_mesh_instance();
	void apply_material();
	Ref<Material> load_default_material();
	float get_vs() const;

protected:
	static void _bind_methods();
	void _notification(int p_what);

public:
	VoxelPart();

	void _process(double p_delta) override;

	void set_volume(const Ref<VoxelVolume> &v);
	Ref<VoxelVolume> get_volume() const { return volume; }
	void set_palette(const PackedColorArray &v);
	PackedColorArray get_palette() const { return palette; }
	void set_material_override(const Ref<Material> &m);
	Ref<Material> get_material_override() const { return material_override; }
	void set_auto_rebuild(bool v);
	bool get_auto_rebuild() const { return auto_rebuild; }
	void set_ao_enabled(bool v);
	bool get_ao_enabled() const { return ao_enabled; }

	void rebuild();
	void mark_dirty() { dirty = true; }
	bool is_dirty() const { return dirty; }
	MeshInstance3D *get_mesh_instance();

	Vector3 local_to_voxel(Vector3 p) const;
	Vector3 voxel_to_local(Vector3 v) const;

	int cut(Vector3 a_local, Vector3 b_local, float radius_m);
	int carve_sphere_local(Vector3 center_local, float radius_m);
	int bruise(Vector3 center_local, float radius_m, int level);
	int burn(Vector3 center_local, float radius_m, int level);
	int trim_hair(Vector3 a_local, Vector3 b_local, float radius_m, int keep_length_vox);
	Dictionary raycast_local(Vector3 origin_local, Vector3 dir_local, float max_dist_m) const;
	Dictionary raycast_global(Vector3 origin, Vector3 dir, float max_dist_m) const;
	TypedArray<VoxelVolume> detach_loose(Vector3i anchor_voxel = Vector3i(-1, -1, -1));
	int get_solid_count() const;
};

} // namespace godot

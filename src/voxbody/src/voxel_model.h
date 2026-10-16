#pragma once
// VoxelModel: one .vxm file in memory — a named set of VoxelVolume parts that
// share a palette (see docs/spec/vxm-format.md).
//
// Responsibilities:
//   * load/save the VXM1 binary format (header JSON + raw byte fields),
//   * keep the 256-entry sRGB palette as a PackedColorArray,
//   * keep the material id -> name lookup table (int keys) and header meta.
// Everything else (nodes, meshes) lives in VoxelModelNode / VoxelPart.
//
// Decisions where the spec is silent:
//   * palette is always normalised to exactly 256 entries (missing = black),
//   * materials keys are converted to int when the JSON key is an integer
//     string, so GDScript can write model.materials[VoxelVolume.MAT_SKIN],
//   * save_to_file writes all three fields for every part and the full
//     256-entry palette; header "format" is 1, unknown formats are rejected,
//   * the extra properties voxel_size (top-level default) and meta are kept
//     so a round trip preserves the header.

#include <godot_cpp/classes/resource.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/packed_byte_array.hpp>
#include <godot_cpp/variant/packed_color_array.hpp>
#include <godot_cpp/variant/typed_array.hpp>

#include "voxel_volume.h"

namespace godot {

class VoxelModel : public Resource {
	GDCLASS(VoxelModel, Resource)

public:
	static constexpr int PALETTE_SIZE = 256;

private:
	String model_name;
	float voxel_size = 0.005f;
	PackedColorArray palette;
	Dictionary materials;
	Dictionary meta;
	TypedArray<VoxelVolume> parts;

	static PackedColorArray make_default_palette();
	Dictionary build_header() const;

protected:
	static void _bind_methods();

public:
	VoxelModel();

	// Returns null (and pushes an error) when the file cannot be read or parsed.
	static Ref<VoxelModel> load_from_file(const String &path);
	bool load_from_buffer(const PackedByteArray &bytes);
	bool save_to_file(const String &path) const;
	PackedByteArray save_to_buffer() const;

	void set_model_name(const String &v) { model_name = v; }
	String get_model_name() const { return model_name; }
	void set_voxel_size(float v) { voxel_size = v; }
	float get_voxel_size() const { return voxel_size; }
	// Always 256 entries after set (padded with black / truncated).
	void set_palette(const PackedColorArray &v);
	PackedColorArray get_palette() const { return palette; }
	void set_materials(const Dictionary &v) { materials = v; }
	Dictionary get_materials() const { return materials; }
	void set_meta_info(const Dictionary &v) { meta = v; }
	Dictionary get_meta_info() const { return meta; }

	void set_parts(const TypedArray<VoxelVolume> &v) { parts = v; }
	TypedArray<VoxelVolume> get_parts() const { return parts; }
	int get_part_count() const { return (int)parts.size(); }
	Ref<VoxelVolume> get_part(int i) const;
	Ref<VoxelVolume> find_part(const String &name) const;
	void add_part(const Ref<VoxelVolume> &v);
	void clear_parts() { parts.clear(); }
};

} // namespace godot

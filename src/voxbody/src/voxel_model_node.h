#pragma once
// VoxelModelNode: builds one VoxelPart child per part of a VoxelModel.
//
// Set model_path (res://assets/models/head_test.vxm) or call set_model();
// children are named after part.name and get the model's palette. Bones are
// not used yet (the next iteration moves parts under BoneAttachment3D).
//
// Decisions where the spec is silent:
//   * part children are regular (non-internal) children so GDScript can find
//     them with get_node("head") and iterate them; they have no owner, so
//     they are never saved into a scene,
//   * set_model() removes the previous part children first; a part without a
//     name is called "part_<index>", duplicate names are made unique by Godot,
//   * model_path set before the node enters the tree is loaded on ENTER_TREE.

#include <godot_cpp/classes/node3d.hpp>
#include <godot_cpp/variant/typed_array.hpp>

#include "voxel_model.h"
#include "voxel_part.h"

namespace godot {

class VoxelModelNode : public Node3D {
	GDCLASS(VoxelModelNode, Node3D)

private:
	String model_path;
	Ref<VoxelModel> model;

	void clear_part_nodes();
	void build_part_nodes();

protected:
	static void _bind_methods();
	void _notification(int p_what);

public:
	VoxelModelNode();

	void set_model_path(const String &p);
	String get_model_path() const { return model_path; }
	void set_model(const Ref<VoxelModel> &m);
	Ref<VoxelModel> get_model() const { return model; }

	bool load_model(const String &path);
	VoxelPart *get_part_node(const String &name) const;
	TypedArray<VoxelPart> get_part_nodes() const;
	void rebuild_all();
};

} // namespace godot

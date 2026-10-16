// VoxelModelNode implementation. See voxel_model_node.h.

#include "voxel_model_node.h"

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

using namespace godot;

void VoxelModelNode::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_model_path", "path"), &VoxelModelNode::set_model_path);
	ClassDB::bind_method(D_METHOD("get_model_path"), &VoxelModelNode::get_model_path);
	ClassDB::bind_method(D_METHOD("set_model", "model"), &VoxelModelNode::set_model);
	ClassDB::bind_method(D_METHOD("get_model"), &VoxelModelNode::get_model);
	ClassDB::bind_method(D_METHOD("load_model", "path"), &VoxelModelNode::load_model);
	ClassDB::bind_method(D_METHOD("get_part_node", "name"), &VoxelModelNode::get_part_node);
	ClassDB::bind_method(D_METHOD("get_part_nodes"), &VoxelModelNode::get_part_nodes);
	ClassDB::bind_method(D_METHOD("rebuild_all"), &VoxelModelNode::rebuild_all);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "model_path", PROPERTY_HINT_FILE, "*.vxm"), "set_model_path", "get_model_path");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "model", PROPERTY_HINT_RESOURCE_TYPE, "VoxelModel", PROPERTY_USAGE_EDITOR), "set_model", "get_model");
}

VoxelModelNode::VoxelModelNode() {}

void VoxelModelNode::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			if (model.is_null() && !model_path.is_empty()) {
				load_model(model_path);
			}
		} break;
		default:
			break;
	}
}

void VoxelModelNode::set_model_path(const String &p) {
	model_path = p;
	if (is_inside_tree()) {
		if (model_path.is_empty()) {
			set_model(Ref<VoxelModel>());
		} else {
			load_model(model_path);
		}
	}
}

bool VoxelModelNode::load_model(const String &path) {
	Ref<VoxelModel> m = VoxelModel::load_from_file(path);
	if (m.is_null()) {
		return false;
	}
	model_path = path;
	set_model(m);
	return true;
}

void VoxelModelNode::clear_part_nodes() {
	// Collect first: removing while iterating children invalidates indices.
	TypedArray<VoxelPart> parts = get_part_nodes();
	for (int64_t i = 0; i < parts.size(); i++) {
		VoxelPart *p = Object::cast_to<VoxelPart>(parts[i]);
		if (p) {
			remove_child(p);
			p->queue_free();
		}
	}
}

void VoxelModelNode::build_part_nodes() {
	if (model.is_null()) {
		return;
	}
	const PackedColorArray palette = model->get_palette();
	for (int i = 0; i < model->get_part_count(); i++) {
		Ref<VoxelVolume> vol = model->get_part(i);
		if (vol.is_null()) {
			continue;
		}
		VoxelPart *part = memnew(VoxelPart);
		String name = vol->get_part_name();
		if (name.is_empty()) {
			name = "part_" + String::num_int64(i);
		}
		part->set_name(name);
		part->set_volume(vol);
		part->set_palette(palette);
		add_child(part);
	}
}

void VoxelModelNode::set_model(const Ref<VoxelModel> &m) {
	clear_part_nodes();
	model = m;
	build_part_nodes();
}

VoxelPart *VoxelModelNode::get_part_node(const String &name) const {
	const int n = get_child_count();
	for (int i = 0; i < n; i++) {
		VoxelPart *p = Object::cast_to<VoxelPart>(get_child(i));
		if (p && p->get_name() == StringName(name)) {
			return p;
		}
	}
	return nullptr;
}

TypedArray<VoxelPart> VoxelModelNode::get_part_nodes() const {
	TypedArray<VoxelPart> out;
	const int n = get_child_count();
	for (int i = 0; i < n; i++) {
		VoxelPart *p = Object::cast_to<VoxelPart>(get_child(i));
		if (p) {
			out.push_back(p);
		}
	}
	return out;
}

void VoxelModelNode::rebuild_all() {
	const int n = get_child_count();
	for (int i = 0; i < n; i++) {
		VoxelPart *p = Object::cast_to<VoxelPart>(get_child(i));
		if (p) {
			p->rebuild();
		}
	}
}

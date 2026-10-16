// VoxelPart implementation. See voxel_part.h.

#include "voxel_part.h"

#include <godot_cpp/classes/rendering_server.hpp>
#include <godot_cpp/classes/resource_loader.hpp>
#include <godot_cpp/classes/shader.hpp>
#include <godot_cpp/classes/shader_material.hpp>
#include <godot_cpp/classes/standard_material3d.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/object.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

#include "voxel_mesher.h"

#include <cmath>

using namespace godot;

static const char *DEFAULT_MATERIAL_PATH = "res://addons/voxbody/voxel_material.tres";
static const char *VOXEL_SIZE_PARAM = "voxel_size";

void VoxelPart::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_volume", "volume"), &VoxelPart::set_volume);
	ClassDB::bind_method(D_METHOD("get_volume"), &VoxelPart::get_volume);
	ClassDB::bind_method(D_METHOD("set_palette", "palette"), &VoxelPart::set_palette);
	ClassDB::bind_method(D_METHOD("get_palette"), &VoxelPart::get_palette);
	ClassDB::bind_method(D_METHOD("set_material_override", "material"), &VoxelPart::set_material_override);
	ClassDB::bind_method(D_METHOD("get_material_override"), &VoxelPart::get_material_override);
	ClassDB::bind_method(D_METHOD("set_auto_rebuild", "enabled"), &VoxelPart::set_auto_rebuild);
	ClassDB::bind_method(D_METHOD("get_auto_rebuild"), &VoxelPart::get_auto_rebuild);
	ClassDB::bind_method(D_METHOD("set_ao_enabled", "enabled"), &VoxelPart::set_ao_enabled);
	ClassDB::bind_method(D_METHOD("get_ao_enabled"), &VoxelPart::get_ao_enabled);

	ClassDB::bind_method(D_METHOD("rebuild"), &VoxelPart::rebuild);
	ClassDB::bind_method(D_METHOD("mark_dirty"), &VoxelPart::mark_dirty);
	ClassDB::bind_method(D_METHOD("is_dirty"), &VoxelPart::is_dirty);
	ClassDB::bind_method(D_METHOD("get_mesh_instance"), &VoxelPart::get_mesh_instance);

	ClassDB::bind_method(D_METHOD("local_to_voxel", "p"), &VoxelPart::local_to_voxel);
	ClassDB::bind_method(D_METHOD("voxel_to_local", "v"), &VoxelPart::voxel_to_local);

	ClassDB::bind_method(D_METHOD("cut", "a_local", "b_local", "radius_m"), &VoxelPart::cut);
	ClassDB::bind_method(D_METHOD("carve_sphere_local", "center_local", "radius_m"), &VoxelPart::carve_sphere_local);
	ClassDB::bind_method(D_METHOD("bruise", "center_local", "radius_m", "level"), &VoxelPart::bruise);
	ClassDB::bind_method(D_METHOD("burn", "center_local", "radius_m", "level"), &VoxelPart::burn);
	ClassDB::bind_method(D_METHOD("trim_hair", "a_local", "b_local", "radius_m", "keep_length_vox"), &VoxelPart::trim_hair);
	ClassDB::bind_method(D_METHOD("raycast_local", "origin_local", "dir_local", "max_dist_m"), &VoxelPart::raycast_local);
	ClassDB::bind_method(D_METHOD("raycast_global", "origin", "dir", "max_dist_m"), &VoxelPart::raycast_global);
	ClassDB::bind_method(D_METHOD("detach_loose", "anchor_voxel"), &VoxelPart::detach_loose, DEFVAL(Vector3i(-1, -1, -1)));
	ClassDB::bind_method(D_METHOD("get_solid_count"), &VoxelPart::get_solid_count);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "volume", PROPERTY_HINT_RESOURCE_TYPE, "VoxelVolume"), "set_volume", "get_volume");
	ADD_PROPERTY(PropertyInfo(Variant::PACKED_COLOR_ARRAY, "palette"), "set_palette", "get_palette");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "material_override", PROPERTY_HINT_RESOURCE_TYPE, "Material"), "set_material_override", "get_material_override");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "auto_rebuild"), "set_auto_rebuild", "get_auto_rebuild");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "ao_enabled"), "set_ao_enabled", "get_ao_enabled");

	ADD_SIGNAL(MethodInfo("mesh_rebuilt"));
	ADD_SIGNAL(MethodInfo("pieces_detached", PropertyInfo(Variant::ARRAY, "pieces", PROPERTY_HINT_ARRAY_TYPE, "24/17:VoxelVolume")));
}

VoxelPart::VoxelPart() {}

void VoxelPart::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_READY: {
			ensure_mesh_instance();
			set_process(true);
		} break;
		default:
			break;
	}
}

void VoxelPart::_process(double p_delta) {
	if (auto_rebuild && dirty) {
		rebuild();
	}
}

// ---------------------------------------------------------------------------
// properties

void VoxelPart::set_volume(const Ref<VoxelVolume> &v) {
	volume = v;
	dirty = true;
}

void VoxelPart::set_palette(const PackedColorArray &v) {
	palette = v;
	dirty = true;
}

void VoxelPart::set_material_override(const Ref<Material> &m) {
	material_override = m;
	if (mesh_instance) {
		apply_material();
	}
}

void VoxelPart::set_auto_rebuild(bool v) {
	auto_rebuild = v;
}

void VoxelPart::set_ao_enabled(bool v) {
	if (ao_enabled != v) {
		ao_enabled = v;
		dirty = true;
	}
}

// ---------------------------------------------------------------------------
// mesh / material

void VoxelPart::ensure_mesh_instance() {
	if (mesh_instance) {
		if (ObjectDB::get_instance(mesh_instance_id) == mesh_instance) {
			return;
		}
		// The child was freed externally: forget it and build a fresh one.
		mesh_instance = nullptr;
		active_material.unref();
		active_source.unref();
	}
	mesh_instance = memnew(MeshInstance3D);
	mesh_instance->set_name("VoxelMesh");
	mesh_instance_id = mesh_instance->get_instance_id();
	add_child(mesh_instance, false, INTERNAL_MODE_BACK);
}

MeshInstance3D *VoxelPart::get_mesh_instance() {
	ensure_mesh_instance();
	return mesh_instance;
}

Ref<Material> VoxelPart::load_default_material() {
	// The shared shader material is cached by ResourceLoader, so asking for it
	// on every rebuild is cheap and picks it up as soon as it exists.
	ResourceLoader *rl = ResourceLoader::get_singleton();
	if (rl && rl->exists(DEFAULT_MATERIAL_PATH)) {
		Ref<Material> m = rl->load(DEFAULT_MATERIAL_PATH, "Material");
		if (m.is_valid()) {
			return m;
		}
	}
	// Fallback: plain vertex-colour material. The mesher writes sRGB colours.
	if (fallback_material.is_null()) {
		Ref<StandardMaterial3D> sm;
		sm.instantiate();
		sm->set_flag(BaseMaterial3D::FLAG_ALBEDO_FROM_VERTEX_COLOR, true);
		sm->set_flag(BaseMaterial3D::FLAG_SRGB_VERTEX_COLOR, true);
		sm->set_roughness(0.85f);
		fallback_material = sm;
	}
	return fallback_material;
}

float VoxelPart::get_vs() const {
	if (volume.is_null()) {
		return 1.0f;
	}
	const float vs = volume->get_voxel_size();
	return vs > 0.0f ? vs : 1.0f;
}

void VoxelPart::apply_material() {
	if (!mesh_instance) {
		return;
	}
	Ref<Material> source = material_override;
	if (source.is_null()) {
		source = load_default_material();
	}
	const float vs = volume.is_valid() ? volume->get_voxel_size() : -1.0f;
	if (source == active_source && active_material.is_valid() && vs == active_voxel_size) {
		return; // nothing changed
	}

	Ref<Material> chosen = source;
	Ref<ShaderMaterial> shader_mat = source;
	if (shader_mat.is_valid() && vs > 0.0f) {
		// Only touch the uniform when the shader actually declares it. The current
		// value is the material's explicit value, else the default written in the
		// shader code (asked from the RenderingServer), else unknown. The dummy
		// renderer in headless mode parses no shader code, so the explicit value
		// is the only thing it can see.
		Variant cur = shader_mat->get_shader_parameter(VOXEL_SIZE_PARAM);
		bool has_param = cur.get_type() != Variant::NIL;
		Ref<Shader> shader = shader_mat->get_shader();
		if (!has_param && shader.is_valid()) {
			RenderingServer *rs = RenderingServer::get_singleton();
			if (rs) {
				cur = rs->shader_get_parameter_default(shader->get_rid(), VOXEL_SIZE_PARAM);
				has_param = cur.get_type() != Variant::NIL;
			}
		}
		if (!has_param && shader.is_valid()) {
			const Array uniforms = shader->get_shader_uniform_list();
			for (int64_t i = 0; i < uniforms.size(); i++) {
				const Dictionary u = uniforms[i];
				if ((String)u.get("name", "") == VOXEL_SIZE_PARAM) {
					has_param = true;
					break;
				}
			}
		}
		if (has_param) {
			const float cur_vs = (cur.get_type() == Variant::NIL) ? -1.0f : (float)(double)cur;
			if (std::fabs(cur_vs - vs) > 1e-7f) {
				// A per-part copy so that parts with another voxel size keep theirs.
				Ref<ShaderMaterial> copy = shader_mat->duplicate();
				copy->set_shader_parameter(VOXEL_SIZE_PARAM, vs);
				chosen = copy;
			}
		}
	}
	active_material = chosen;
	active_source = source;
	active_voxel_size = vs;
	mesh_instance->set_material_override(active_material);
}

void VoxelPart::rebuild() {
	ensure_mesh_instance();
	if (volume.is_null()) {
		mesh_instance->set_mesh(Ref<Mesh>());
	} else {
		Dictionary options;
		options["ao"] = ao_enabled;
		Ref<ArrayMesh> mesh = VoxelMesher::build_mesh(volume, palette, options);
		mesh_instance->set_mesh(mesh);
	}
	apply_material();
	dirty = false;
	emit_signal("mesh_rebuilt");
}

// ---------------------------------------------------------------------------
// coordinate conversion

Vector3 VoxelPart::local_to_voxel(Vector3 p) const {
	if (volume.is_null()) {
		return p;
	}
	return (p - volume->get_origin()) / get_vs();
}

Vector3 VoxelPart::voxel_to_local(Vector3 v) const {
	if (volume.is_null()) {
		return v;
	}
	return volume->get_origin() + v * get_vs();
}

// ---------------------------------------------------------------------------
// damage API (metres in, voxel units to the volume)

int VoxelPart::cut(Vector3 a_local, Vector3 b_local, float radius_m) {
	if (volume.is_null()) {
		return 0;
	}
	const int n = volume->carve_capsule(local_to_voxel(a_local), local_to_voxel(b_local), radius_m / get_vs(), true);
	if (n > 0) {
		dirty = true;
	}
	return n;
}

int VoxelPart::carve_sphere_local(Vector3 center_local, float radius_m) {
	if (volume.is_null()) {
		return 0;
	}
	const int n = volume->carve_sphere(local_to_voxel(center_local), radius_m / get_vs(), true);
	if (n > 0) {
		dirty = true;
	}
	return n;
}

int VoxelPart::bruise(Vector3 center_local, float radius_m, int level) {
	if (volume.is_null()) {
		return 0;
	}
	const int n = volume->bruise_sphere(local_to_voxel(center_local), radius_m / get_vs(), level);
	if (n > 0) {
		dirty = true;
	}
	return n;
}

int VoxelPart::burn(Vector3 center_local, float radius_m, int level) {
	if (volume.is_null()) {
		return 0;
	}
	const int n = volume->burn_sphere(local_to_voxel(center_local), radius_m / get_vs(), level);
	if (n > 0) {
		dirty = true;
	}
	return n;
}

int VoxelPart::trim_hair(Vector3 a_local, Vector3 b_local, float radius_m, int keep_length_vox) {
	if (volume.is_null()) {
		return 0;
	}
	const int n = volume->trim_hair_capsule(local_to_voxel(a_local), local_to_voxel(b_local), radius_m / get_vs(), keep_length_vox);
	if (n > 0) {
		dirty = true;
	}
	return n;
}

Dictionary VoxelPart::raycast_local(Vector3 origin_local, Vector3 dir_local, float max_dist_m) const {
	Dictionary r;
	r["hit"] = false;
	if (volume.is_null()) {
		return r;
	}
	const float vs = get_vs();
	Dictionary hit = volume->raycast(local_to_voxel(origin_local), dir_local, max_dist_m / vs);
	if (!(bool)hit.get("hit", false)) {
		return r;
	}
	r["hit"] = true;
	r["voxel"] = hit["voxel"];
	r["normal"] = Vector3((Vector3i)hit["normal"]);
	r["distance"] = (float)(double)hit["distance"] * vs;
	r["material"] = hit["material"];
	r["position"] = voxel_to_local((Vector3)hit["position"]);
	return r;
}

Dictionary VoxelPart::raycast_global(Vector3 origin, Vector3 dir, float max_dist_m) const {
	Dictionary r;
	r["hit"] = false;
	if (volume.is_null() || dir.length_squared() < 1e-12f) {
		return r;
	}
	const Transform3D xf = get_global_transform();
	const Transform3D inv = xf.affine_inverse();
	const Vector3 o_local = inv.xform(origin);
	const Vector3 d_local_raw = inv.basis.xform(dir.normalized());
	const float scale = d_local_raw.length(); // local metres per global metre along the ray
	if (scale < 1e-12f) {
		return r;
	}
	Dictionary hit = raycast_local(o_local, d_local_raw / scale, max_dist_m * scale);
	if (!(bool)hit.get("hit", false)) {
		return r;
	}
	const Vector3 pos_global = xf.xform((Vector3)hit["position"]);
	r["hit"] = true;
	r["voxel"] = hit["voxel"];
	r["normal"] = xf.basis.xform((Vector3)hit["normal"]).normalized();
	r["distance"] = (float)(pos_global - origin).length();
	r["material"] = hit["material"];
	r["position"] = pos_global;
	return r;
}

TypedArray<VoxelVolume> VoxelPart::detach_loose(Vector3i anchor_voxel) {
	TypedArray<VoxelVolume> pieces;
	if (volume.is_null()) {
		return pieces;
	}
	pieces = volume->detach_disconnected(anchor_voxel);
	if (pieces.size() > 0) {
		dirty = true;
		emit_signal("pieces_detached", pieces);
	}
	return pieces;
}

int VoxelPart::get_solid_count() const {
	return volume.is_valid() ? volume->count_solid() : 0;
}

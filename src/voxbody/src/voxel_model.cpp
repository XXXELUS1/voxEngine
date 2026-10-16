// VoxelModel implementation: VXM1 reader/writer. See voxel_model.h.

#include "voxel_model.h"

#include <godot_cpp/classes/file_access.hpp>
#include <godot_cpp/classes/json.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

#include <cstring>

using namespace godot;

static const char VXM_MAGIC[4] = { 'V', 'X', 'M', '1' };

void VoxelModel::_bind_methods() {
	ClassDB::bind_static_method("VoxelModel", D_METHOD("load_from_file", "path"), &VoxelModel::load_from_file);
	ClassDB::bind_method(D_METHOD("load_from_buffer", "bytes"), &VoxelModel::load_from_buffer);
	ClassDB::bind_method(D_METHOD("save_to_file", "path"), &VoxelModel::save_to_file);
	ClassDB::bind_method(D_METHOD("save_to_buffer"), &VoxelModel::save_to_buffer);

	ClassDB::bind_method(D_METHOD("set_model_name", "v"), &VoxelModel::set_model_name);
	ClassDB::bind_method(D_METHOD("get_model_name"), &VoxelModel::get_model_name);
	ClassDB::bind_method(D_METHOD("set_voxel_size", "v"), &VoxelModel::set_voxel_size);
	ClassDB::bind_method(D_METHOD("get_voxel_size"), &VoxelModel::get_voxel_size);
	ClassDB::bind_method(D_METHOD("set_palette", "v"), &VoxelModel::set_palette);
	ClassDB::bind_method(D_METHOD("get_palette"), &VoxelModel::get_palette);
	ClassDB::bind_method(D_METHOD("set_materials", "v"), &VoxelModel::set_materials);
	ClassDB::bind_method(D_METHOD("get_materials"), &VoxelModel::get_materials);
	ClassDB::bind_method(D_METHOD("set_meta_info", "v"), &VoxelModel::set_meta_info);
	ClassDB::bind_method(D_METHOD("get_meta_info"), &VoxelModel::get_meta_info);
	ClassDB::bind_method(D_METHOD("set_parts", "v"), &VoxelModel::set_parts);
	ClassDB::bind_method(D_METHOD("get_parts"), &VoxelModel::get_parts);

	ClassDB::bind_method(D_METHOD("get_part_count"), &VoxelModel::get_part_count);
	ClassDB::bind_method(D_METHOD("get_part", "i"), &VoxelModel::get_part);
	ClassDB::bind_method(D_METHOD("find_part", "name"), &VoxelModel::find_part);
	ClassDB::bind_method(D_METHOD("add_part", "v"), &VoxelModel::add_part);
	ClassDB::bind_method(D_METHOD("clear_parts"), &VoxelModel::clear_parts);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "model_name"), "set_model_name", "get_model_name");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "voxel_size"), "set_voxel_size", "get_voxel_size");
	ADD_PROPERTY(PropertyInfo(Variant::PACKED_COLOR_ARRAY, "palette"), "set_palette", "get_palette");
	ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "materials"), "set_materials", "get_materials");
	ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "meta_info"), "set_meta_info", "get_meta_info");
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "parts", PROPERTY_HINT_ARRAY_TYPE, "24/17:VoxelVolume"), "set_parts", "get_parts");

	BIND_CONSTANT(PALETTE_SIZE);
}

VoxelModel::VoxelModel() {
	palette = make_default_palette();
}

PackedColorArray VoxelModel::make_default_palette() {
	PackedColorArray p;
	p.resize(PALETTE_SIZE);
	p.fill(Color(0, 0, 0, 1));
	return p;
}

void VoxelModel::set_palette(const PackedColorArray &v) {
	palette = make_default_palette();
	Color *dst = palette.ptrw();
	const int64_t n = v.size() < PALETTE_SIZE ? v.size() : PALETTE_SIZE;
	const Color *src = v.ptr();
	for (int64_t i = 0; i < n; i++) {
		dst[i] = Color(src[i].r, src[i].g, src[i].b, 1.0f);
	}
}

Ref<VoxelVolume> VoxelModel::get_part(int i) const {
	if (i < 0 || i >= (int)parts.size()) {
		return Ref<VoxelVolume>();
	}
	return parts[i];
}

Ref<VoxelVolume> VoxelModel::find_part(const String &name) const {
	for (int i = 0; i < (int)parts.size(); i++) {
		Ref<VoxelVolume> v = parts[i];
		if (v.is_valid() && v->get_part_name() == name) {
			return v;
		}
	}
	return Ref<VoxelVolume>();
}

void VoxelModel::add_part(const Ref<VoxelVolume> &v) {
	ERR_FAIL_COND_MSG(v.is_null(), "VoxelModel.add_part: volume is null");
	parts.push_back(v);
}

// ---------------------------------------------------------------------------
// loading

static bool read_vec3(const Variant &v, Vector3 &out) {
	if (v.get_type() != Variant::ARRAY) {
		return false;
	}
	const Array a = v;
	if (a.size() < 3) {
		return false;
	}
	out = Vector3((float)(double)a[0], (float)(double)a[1], (float)(double)a[2]);
	return true;
}

static bool read_vec3i(const Variant &v, Vector3i &out) {
	if (v.get_type() != Variant::ARRAY) {
		return false;
	}
	const Array a = v;
	if (a.size() < 3) {
		return false;
	}
	out = Vector3i((int)(int64_t)a[0], (int)(int64_t)a[1], (int)(int64_t)a[2]);
	return true;
}

Ref<VoxelModel> VoxelModel::load_from_file(const String &path) {
	Ref<FileAccess> f = FileAccess::open(path, FileAccess::READ);
	if (f.is_null()) {
		ERR_PRINT("VoxelModel.load_from_file: cannot open '" + path + "' (error " + String::num_int64((int64_t)FileAccess::get_open_error()) + ")");
		return Ref<VoxelModel>();
	}
	const PackedByteArray bytes = f->get_buffer((int64_t)f->get_length());
	f->close();
	Ref<VoxelModel> model;
	model.instantiate();
	if (!model->load_from_buffer(bytes)) {
		ERR_PRINT("VoxelModel.load_from_file: failed to parse '" + path + "'");
		return Ref<VoxelModel>();
	}
	return model;
}

bool VoxelModel::load_from_buffer(const PackedByteArray &bytes) {
	const int64_t size = bytes.size();
	const uint8_t *data = bytes.ptr();
	ERR_FAIL_COND_V_MSG(size < 8, false, "VXM: buffer too short");
	ERR_FAIL_COND_V_MSG(std::memcmp(data, VXM_MAGIC, 4) != 0, false, "VXM: bad magic (expected 'VXM1')");
	const uint32_t header_len = (uint32_t)data[4] | ((uint32_t)data[5] << 8) | ((uint32_t)data[6] << 16) | ((uint32_t)data[7] << 24);
	ERR_FAIL_COND_V_MSG((int64_t)8 + header_len > size, false, "VXM: header length exceeds buffer");

	const String header_text = String::utf8((const char *)data + 8, (int64_t)header_len);
	const Variant parsed = JSON::parse_string(header_text);
	ERR_FAIL_COND_V_MSG(parsed.get_type() != Variant::DICTIONARY, false, "VXM: header is not a JSON object");
	const Dictionary header = parsed;

	const int format = (int)(int64_t)header.get("format", 1);
	ERR_FAIL_COND_V_MSG(format != 1, false, "VXM: unsupported format " + String::num_int64(format));

	model_name = (String)header.get("name", "");
	voxel_size = (float)(double)header.get("voxel_size", 0.005);
	ERR_FAIL_COND_V_MSG(!(voxel_size > 0.0f), false, "VXM: voxel_size must be positive");

	// palette
	palette = make_default_palette();
	{
		const Variant pv = header.get("palette", Variant());
		if (pv.get_type() == Variant::ARRAY) {
			const Array pa = pv;
			Color *dst = palette.ptrw();
			const int64_t n = pa.size() < PALETTE_SIZE ? pa.size() : PALETTE_SIZE;
			for (int64_t i = 0; i < n; i++) {
				const Variant ev = pa[i];
				if (ev.get_type() != Variant::ARRAY) {
					continue;
				}
				const Array e = ev;
				if (e.size() < 3) {
					continue;
				}
				dst[i] = Color((float)(double)e[0] / 255.0f, (float)(double)e[1] / 255.0f, (float)(double)e[2] / 255.0f, 1.0f);
			}
		}
	}

	// materials: "1": "skin" -> {1: "skin"}
	materials = Dictionary();
	{
		const Variant mv = header.get("materials", Variant());
		if (mv.get_type() == Variant::DICTIONARY) {
			const Dictionary md = mv;
			const Array keys = md.keys();
			for (int64_t i = 0; i < keys.size(); i++) {
				const Variant k = keys[i];
				Variant nk = k;
				if (k.get_type() == Variant::STRING) {
					const String ks = k;
					if (ks.is_valid_int()) {
						nk = ks.to_int();
					}
				}
				materials[nk] = md[k];
			}
		}
	}

	// meta: kept verbatim
	{
		const Variant mv = header.get("meta", Variant());
		meta = (mv.get_type() == Variant::DICTIONARY) ? (Dictionary)mv : Dictionary();
	}

	// parts
	parts.clear();
	const Variant partsv = header.get("parts", Variant());
	ERR_FAIL_COND_V_MSG(partsv.get_type() != Variant::ARRAY, false, "VXM: header has no 'parts' array");
	const Array parts_arr = partsv;
	int64_t offset = (int64_t)8 + header_len;
	for (int64_t pi = 0; pi < parts_arr.size(); pi++) {
		const Variant pv = parts_arr[pi];
		ERR_FAIL_COND_V_MSG(pv.get_type() != Variant::DICTIONARY, false, "VXM: part " + String::num_int64(pi) + " is not an object");
		const Dictionary pd = pv;

		Vector3i dims;
		ERR_FAIL_COND_V_MSG(!read_vec3i(pd.get("dims", Variant()), dims), false, "VXM: part " + String::num_int64(pi) + " has no valid 'dims'");
		ERR_FAIL_COND_V_MSG(dims.x < 0 || dims.y < 0 || dims.z < 0, false, "VXM: negative dims");
		const int64_t nvox = (int64_t)dims.x * dims.y * dims.z;
		ERR_FAIL_COND_V_MSG(nvox > (int64_t)1 << 30, false, "VXM: part too large");

		const Variant fv = pd.get("fields", Variant());
		ERR_FAIL_COND_V_MSG(fv.get_type() != Variant::ARRAY, false, "VXM: part " + String::num_int64(pi) + " has no 'fields' array");
		const Array fields = fv;

		Ref<VoxelVolume> vol;
		vol.instantiate();
		const float part_vs = (float)(double)pd.get("voxel_size", (double)voxel_size);
		ERR_FAIL_COND_V_MSG(!(part_vs > 0.0f), false, "VXM: part voxel_size must be positive");
		vol->create(dims, part_vs);
		vol->set_part_name((String)pd.get("name", "part" + String::num_int64(pi)));
		vol->set_bone((String)pd.get("bone", ""));
		Vector3 origin;
		if (read_vec3(pd.get("origin", Variant()), origin)) {
			vol->set_origin(origin);
		}

		bool has_material = false;
		for (int64_t fi = 0; fi < fields.size(); fi++) {
			const String field = fields[fi];
			ERR_FAIL_COND_V_MSG(offset + nvox > size, false, "VXM: truncated data in part '" + vol->get_part_name() + "' field '" + field + "'");
			if (field == "material") {
				vol->set_material_data(bytes.slice(offset, offset + nvox));
				has_material = true;
			} else if (field == "color") {
				vol->set_color_data(bytes.slice(offset, offset + nvox));
			} else if (field == "state") {
				vol->set_state_data(bytes.slice(offset, offset + nvox));
			}
			// unknown fields are skipped
			offset += nvox;
		}
		ERR_FAIL_COND_V_MSG(!has_material, false, "VXM: part '" + vol->get_part_name() + "' has no 'material' field");
		parts.push_back(vol);
	}
	return true;
}

// ---------------------------------------------------------------------------
// saving

Dictionary VoxelModel::build_header() const {
	Dictionary h;
	h["format"] = 1;
	h["name"] = model_name;
	h["voxel_size"] = (double)voxel_size;

	Array pal;
	const Color *pc = palette.ptr();
	for (int64_t i = 0; i < palette.size(); i++) {
		Array e;
		e.push_back((int64_t)Math::round(Math::clamp(pc[i].r, 0.0f, 1.0f) * 255.0f));
		e.push_back((int64_t)Math::round(Math::clamp(pc[i].g, 0.0f, 1.0f) * 255.0f));
		e.push_back((int64_t)Math::round(Math::clamp(pc[i].b, 0.0f, 1.0f) * 255.0f));
		pal.push_back(e);
	}
	h["palette"] = pal;

	Dictionary mats;
	{
		const Array keys = materials.keys();
		for (int64_t i = 0; i < keys.size(); i++) {
			const Variant k = keys[i];
			mats[(String)k] = materials[k];
		}
	}
	h["materials"] = mats;

	Array parr;
	for (int i = 0; i < (int)parts.size(); i++) {
		Ref<VoxelVolume> v = parts[i];
		if (v.is_null()) {
			continue;
		}
		Dictionary p;
		p["name"] = v->get_part_name();
		p["bone"] = v->get_bone();
		p["voxel_size"] = (double)v->get_voxel_size();
		Array o;
		const Vector3 origin = v->get_origin();
		o.push_back((double)origin.x);
		o.push_back((double)origin.y);
		o.push_back((double)origin.z);
		p["origin"] = o;
		Array d;
		const Vector3i dims = v->get_dims();
		d.push_back((int64_t)dims.x);
		d.push_back((int64_t)dims.y);
		d.push_back((int64_t)dims.z);
		p["dims"] = d;
		Array f;
		f.push_back("material");
		f.push_back("color");
		f.push_back("state");
		p["fields"] = f;
		parr.push_back(p);
	}
	h["parts"] = parr;
	h["meta"] = meta;
	return h;
}

PackedByteArray VoxelModel::save_to_buffer() const {
	const String header_text = JSON::stringify(build_header());
	const CharString header_utf8 = header_text.utf8();
	const int64_t header_len = header_utf8.length();

	PackedByteArray out;
	out.resize(8 + header_len);
	uint8_t *w = out.ptrw();
	std::memcpy(w, VXM_MAGIC, 4);
	w[4] = (uint8_t)(header_len & 0xFF);
	w[5] = (uint8_t)((header_len >> 8) & 0xFF);
	w[6] = (uint8_t)((header_len >> 16) & 0xFF);
	w[7] = (uint8_t)((header_len >> 24) & 0xFF);
	std::memcpy(w + 8, header_utf8.get_data(), (size_t)header_len);

	for (int i = 0; i < (int)parts.size(); i++) {
		Ref<VoxelVolume> v = parts[i];
		if (v.is_null()) {
			continue;
		}
		out.append_array(v->get_material_data());
		out.append_array(v->get_color_data());
		out.append_array(v->get_state_data());
	}
	return out;
}

bool VoxelModel::save_to_file(const String &path) const {
	Ref<FileAccess> f = FileAccess::open(path, FileAccess::WRITE);
	ERR_FAIL_COND_V_MSG(f.is_null(), false, "VoxelModel.save_to_file: cannot open '" + path + "' for writing (error " + String::num_int64((int64_t)FileAccess::get_open_error()) + ")");
	const PackedByteArray bytes = save_to_buffer();
	f->store_buffer(bytes);
	const Error err = f->get_error();
	f->close();
	ERR_FAIL_COND_V_MSG(err != OK, false, "VoxelModel.save_to_file: write error " + String::num_int64((int64_t)err));
	return true;
}

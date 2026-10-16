#include "voxel_volume.h"

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

#include <algorithm>
#include <cmath>
#include <queue>

using namespace godot;

VoxelVolume::VoxelVolume() {}

void VoxelVolume::_bind_methods() {
	ClassDB::bind_method(D_METHOD("create", "dims", "voxel_size"), &VoxelVolume::create);
	ClassDB::bind_method(D_METHOD("duplicate_volume"), &VoxelVolume::duplicate_volume);
	ClassDB::bind_method(D_METHOD("get_dims"), &VoxelVolume::get_dims);
	ClassDB::bind_method(D_METHOD("get_voxel_count"), &VoxelVolume::get_voxel_count);
	ClassDB::bind_method(D_METHOD("set_voxel_size", "v"), &VoxelVolume::set_voxel_size);
	ClassDB::bind_method(D_METHOD("get_voxel_size"), &VoxelVolume::get_voxel_size);
	ClassDB::bind_method(D_METHOD("set_origin", "v"), &VoxelVolume::set_origin);
	ClassDB::bind_method(D_METHOD("get_origin"), &VoxelVolume::get_origin);
	ClassDB::bind_method(D_METHOD("set_part_name", "v"), &VoxelVolume::set_part_name);
	ClassDB::bind_method(D_METHOD("get_part_name"), &VoxelVolume::get_part_name);
	ClassDB::bind_method(D_METHOD("set_bone", "v"), &VoxelVolume::set_bone);
	ClassDB::bind_method(D_METHOD("get_bone"), &VoxelVolume::get_bone);

	ClassDB::bind_method(D_METHOD("is_inside", "v"), &VoxelVolume::is_inside);
	ClassDB::bind_method(D_METHOD("get_material", "v"), &VoxelVolume::get_material);
	ClassDB::bind_method(D_METHOD("get_color", "v"), &VoxelVolume::get_color);
	ClassDB::bind_method(D_METHOD("get_state", "v"), &VoxelVolume::get_state);
	ClassDB::bind_method(D_METHOD("set_voxel", "v", "material", "color"), &VoxelVolume::set_voxel);
	ClassDB::bind_method(D_METHOD("set_state", "v", "state"), &VoxelVolume::set_state);

	ClassDB::bind_method(D_METHOD("get_material_data"), &VoxelVolume::get_material_data);
	ClassDB::bind_method(D_METHOD("set_material_data", "d"), &VoxelVolume::set_material_data);
	ClassDB::bind_method(D_METHOD("get_color_data"), &VoxelVolume::get_color_data);
	ClassDB::bind_method(D_METHOD("set_color_data", "d"), &VoxelVolume::set_color_data);
	ClassDB::bind_method(D_METHOD("get_state_data"), &VoxelVolume::get_state_data);
	ClassDB::bind_method(D_METHOD("set_state_data", "d"), &VoxelVolume::set_state_data);

	ClassDB::bind_method(D_METHOD("count_solid"), &VoxelVolume::count_solid);
	ClassDB::bind_method(D_METHOD("get_solid_min"), &VoxelVolume::get_solid_min);
	ClassDB::bind_method(D_METHOD("get_solid_max"), &VoxelVolume::get_solid_max);

	ClassDB::bind_method(D_METHOD("carve_capsule", "a", "b", "radius", "mark_wound"), &VoxelVolume::carve_capsule, DEFVAL(true));
	ClassDB::bind_method(D_METHOD("carve_sphere", "center", "radius", "mark_wound"), &VoxelVolume::carve_sphere, DEFVAL(true));
	ClassDB::bind_method(D_METHOD("bruise_sphere", "center", "radius", "level"), &VoxelVolume::bruise_sphere);
	ClassDB::bind_method(D_METHOD("burn_sphere", "center", "radius", "level"), &VoxelVolume::burn_sphere);
	ClassDB::bind_method(D_METHOD("stamp_state_sphere", "center", "radius", "set_bits", "clear_bits"), &VoxelVolume::stamp_state_sphere);
	ClassDB::bind_method(D_METHOD("trim_hair_capsule", "a", "b", "radius", "keep_length"), &VoxelVolume::trim_hair_capsule);
	ClassDB::bind_method(D_METHOD("raycast", "origin", "dir", "max_dist"), &VoxelVolume::raycast);
	ClassDB::bind_method(D_METHOD("detach_disconnected", "anchor"), &VoxelVolume::detach_disconnected);

	ClassDB::bind_method(D_METHOD("set_dims", "dims"), &VoxelVolume::set_dims);

	// Storage-only properties so a VoxelVolume survives being saved inside a
	// .tres/.tscn (dims is registered first: the data setters validate against it).
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3I, "dims", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE | PROPERTY_USAGE_EDITOR | PROPERTY_USAGE_READ_ONLY), "set_dims", "get_dims");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "voxel_size"), "set_voxel_size", "get_voxel_size");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "origin"), "set_origin", "get_origin");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "part_name"), "set_part_name", "get_part_name");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "bone"), "set_bone", "get_bone");
	ADD_PROPERTY(PropertyInfo(Variant::PACKED_BYTE_ARRAY, "material_data", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE), "set_material_data", "get_material_data");
	ADD_PROPERTY(PropertyInfo(Variant::PACKED_BYTE_ARRAY, "color_data", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE), "set_color_data", "get_color_data");
	ADD_PROPERTY(PropertyInfo(Variant::PACKED_BYTE_ARRAY, "state_data", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE), "set_state_data", "get_state_data");

	BIND_ENUM_CONSTANT(MAT_AIR);
	BIND_ENUM_CONSTANT(MAT_SKIN);
	BIND_ENUM_CONSTANT(MAT_FAT);
	BIND_ENUM_CONSTANT(MAT_MUSCLE);
	BIND_ENUM_CONSTANT(MAT_BONE);
	BIND_ENUM_CONSTANT(MAT_TOOTH);
	BIND_ENUM_CONSTANT(MAT_CARTILAGE);
	BIND_ENUM_CONSTANT(MAT_VESSEL);
	BIND_ENUM_CONSTANT(MAT_HAIR);
	BIND_ENUM_CONSTANT(MAT_EYE);
	BIND_ENUM_CONSTANT(MAT_NAIL);
	BIND_ENUM_CONSTANT(MAT_WOOD);
	BIND_ENUM_CONSTANT(MAT_METAL);
	BIND_ENUM_CONSTANT(MAT_STONE);
	BIND_ENUM_CONSTANT(MAT_CLOTH);
	BIND_ENUM_CONSTANT(MAT_LEATHER);
	BIND_ENUM_CONSTANT(MAT_GLASS);
	BIND_ENUM_CONSTANT(MAT_CERAMIC);
	BIND_ENUM_CONSTANT(MAT_LIQUID);
	BIND_ENUM_CONSTANT(MAT_PLANT);
	BIND_ENUM_CONSTANT(MAT_GENERIC);

	BIND_ENUM_CONSTANT(STATE_BRUISE_MASK);
	BIND_ENUM_CONSTANT(STATE_BURN_SHIFT);
	BIND_ENUM_CONSTANT(STATE_BURN_MASK);
	BIND_ENUM_CONSTANT(STATE_CUT);
	BIND_ENUM_CONSTANT(STATE_BLOOD);
	BIND_ENUM_CONSTANT(STATE_WET);
}

// ---------------------------------------------------------------------------

bool VoxelVolume::material_bleeds(int m) {
	return m == MAT_FAT || m == MAT_MUSCLE || m == MAT_VESSEL || m == MAT_SKIN;
}

bool VoxelVolume::material_bruises(int m) {
	return m == MAT_SKIN;
}

void VoxelVolume::create(Vector3i p_dims, float p_voxel_size) {
	nx = std::max(0, p_dims.x);
	ny = std::max(0, p_dims.y);
	nz = std::max(0, p_dims.z);
	voxel_size = p_voxel_size;
	const int n = nx * ny * nz;
	material.resize(n);
	color.resize(n);
	state.resize(n);
	material.fill(0);
	color.fill(0);
	state.fill(0);
}

void VoxelVolume::set_dims(Vector3i p_dims) {
	if (p_dims == get_dims()) {
		return;
	}
	create(p_dims, voxel_size);
}

Ref<VoxelVolume> VoxelVolume::duplicate_volume() const {
	Ref<VoxelVolume> v;
	v.instantiate();
	v->nx = nx;
	v->ny = ny;
	v->nz = nz;
	v->voxel_size = voxel_size;
	v->origin = origin;
	v->part_name = part_name;
	v->bone = bone;
	v->material = material.duplicate();
	v->color = color.duplicate();
	v->state = state.duplicate();
	return v;
}

void VoxelVolume::set_voxel(Vector3i v, int p_material, int p_color) {
	if (!in_bounds(v.x, v.y, v.z)) {
		return;
	}
	const int i = idx(v.x, v.y, v.z);
	material[i] = (uint8_t)p_material;
	color[i] = (uint8_t)p_color;
	if (p_material == 0) {
		state[i] = 0;
	}
}

void VoxelVolume::set_state(Vector3i v, int p_state) {
	if (!in_bounds(v.x, v.y, v.z)) {
		return;
	}
	state[idx(v.x, v.y, v.z)] = (uint8_t)p_state;
}

void VoxelVolume::set_material_data(const PackedByteArray &d) {
	ERR_FAIL_COND_MSG(d.size() != get_voxel_count(), "material data size does not match dims");
	material = d;
}
void VoxelVolume::set_color_data(const PackedByteArray &d) {
	ERR_FAIL_COND_MSG(d.size() != get_voxel_count(), "color data size does not match dims");
	color = d;
}
void VoxelVolume::set_state_data(const PackedByteArray &d) {
	ERR_FAIL_COND_MSG(d.size() != get_voxel_count(), "state data size does not match dims");
	state = d;
}

int VoxelVolume::count_solid() const {
	int c = 0;
	const uint8_t *m = material.ptr();
	const int n = get_voxel_count();
	for (int i = 0; i < n; i++) {
		c += m[i] != 0;
	}
	return c;
}

Vector3i VoxelVolume::get_solid_min() const {
	Vector3i mn(nx, ny, nz);
	const uint8_t *m = material.ptr();
	for (int z = 0; z < nz; z++) {
		for (int y = 0; y < ny; y++) {
			for (int x = 0; x < nx; x++) {
				if (m[idx(x, y, z)]) {
					mn.x = std::min(mn.x, x);
					mn.y = std::min(mn.y, y);
					mn.z = std::min(mn.z, z);
				}
			}
		}
	}
	return mn;
}

Vector3i VoxelVolume::get_solid_max() const {
	Vector3i mx(-1, -1, -1);
	const uint8_t *m = material.ptr();
	for (int z = 0; z < nz; z++) {
		for (int y = 0; y < ny; y++) {
			for (int x = 0; x < nx; x++) {
				if (m[idx(x, y, z)]) {
					mx.x = std::max(mx.x, x);
					mx.y = std::max(mx.y, y);
					mx.z = std::max(mx.z, z);
				}
			}
		}
	}
	return mx;
}

// ---------------------------------------------------------------------------
// damage ops

static inline float dist2_point_segment(const Vector3 &p, const Vector3 &a, const Vector3 &b) {
	const Vector3 ab = b - a;
	const float len2 = ab.length_squared();
	float t = 0.0f;
	if (len2 > 1e-12f) {
		t = (p - a).dot(ab) / len2;
		t = std::clamp(t, 0.0f, 1.0f);
	}
	const Vector3 q = a + ab * t;
	return (p - q).length_squared();
}

// Mark solid neighbours of removed voxels as freshly cut.
int VoxelVolume::mark_exposed(const std::vector<int> &removed, bool bleed) {
	static const int off[6][3] = { { 1, 0, 0 }, { -1, 0, 0 }, { 0, 1, 0 }, { 0, -1, 0 }, { 0, 0, 1 }, { 0, 0, -1 } };
	int marked = 0;
	uint8_t *st = state.ptrw();
	const uint8_t *m = material.ptr();
	for (int i : removed) {
		const int z = i / (nx * ny);
		const int y = (i / nx) % ny;
		const int x = i % nx;
		for (auto &o : off) {
			const int ax = x + o[0], ay = y + o[1], az = z + o[2];
			if (!in_bounds(ax, ay, az)) {
				continue;
			}
			const int j = idx(ax, ay, az);
			if (m[j] == 0 || m[j] == MAT_HAIR) {
				continue;
			}
			uint8_t s = st[j] | STATE_CUT;
			if (bleed && material_bleeds(m[j])) {
				s |= STATE_BLOOD;
			}
			if (st[j] != s) {
				st[j] = s;
				marked++;
			}
		}
	}
	return marked;
}

int VoxelVolume::carve_capsule(Vector3 a, Vector3 b, float radius, bool mark_wound) {
	if (nx == 0) {
		return 0;
	}
	const float r2 = radius * radius;
	const int x0 = std::max(0, (int)std::floor(std::min(a.x, b.x) - radius - 1));
	const int y0 = std::max(0, (int)std::floor(std::min(a.y, b.y) - radius - 1));
	const int z0 = std::max(0, (int)std::floor(std::min(a.z, b.z) - radius - 1));
	const int x1 = std::min(nx - 1, (int)std::ceil(std::max(a.x, b.x) + radius + 1));
	const int y1 = std::min(ny - 1, (int)std::ceil(std::max(a.y, b.y) + radius + 1));
	const int z1 = std::min(nz - 1, (int)std::ceil(std::max(a.z, b.z) + radius + 1));
	std::vector<int> removed;
	uint8_t *m = material.ptrw();
	uint8_t *st = state.ptrw();
	for (int z = z0; z <= z1; z++) {
		for (int y = y0; y <= y1; y++) {
			for (int x = x0; x <= x1; x++) {
				const int i = idx(x, y, z);
				if (m[i] == 0) {
					continue;
				}
				const Vector3 p(x + 0.5f, y + 0.5f, z + 0.5f);
				if (dist2_point_segment(p, a, b) <= r2) {
					m[i] = 0;
					st[i] = 0;
					removed.push_back(i);
				}
			}
		}
	}
	if (mark_wound && !removed.empty()) {
		mark_exposed(removed, true);
	}
	return (int)removed.size();
}

int VoxelVolume::carve_sphere(Vector3 center, float radius, bool mark_wound) {
	return carve_capsule(center, center, radius, mark_wound);
}

int VoxelVolume::stamp_state_sphere(Vector3 center, float radius, int set_bits, int clear_bits) {
	if (nx == 0) {
		return 0;
	}
	const float r2 = radius * radius;
	const int x0 = std::max(0, (int)std::floor(center.x - radius - 1));
	const int y0 = std::max(0, (int)std::floor(center.y - radius - 1));
	const int z0 = std::max(0, (int)std::floor(center.z - radius - 1));
	const int x1 = std::min(nx - 1, (int)std::ceil(center.x + radius + 1));
	const int y1 = std::min(ny - 1, (int)std::ceil(center.y + radius + 1));
	const int z1 = std::min(nz - 1, (int)std::ceil(center.z + radius + 1));
	int changed = 0;
	const uint8_t *m = material.ptr();
	uint8_t *st = state.ptrw();
	for (int z = z0; z <= z1; z++) {
		for (int y = y0; y <= y1; y++) {
			for (int x = x0; x <= x1; x++) {
				const int i = idx(x, y, z);
				if (m[i] == 0) {
					continue;
				}
				const Vector3 p(x + 0.5f, y + 0.5f, z + 0.5f);
				if ((p - center).length_squared() > r2) {
					continue;
				}
				const uint8_t s = (st[i] & ~(uint8_t)clear_bits) | (uint8_t)set_bits;
				if (s != st[i]) {
					st[i] = s;
					changed++;
				}
			}
		}
	}
	return changed;
}

int VoxelVolume::bruise_sphere(Vector3 center, float radius, int level) {
	level = std::clamp(level, 0, 3);
	if (nx == 0 || level == 0) {
		return 0;
	}
	const float r2 = radius * radius;
	const int x0 = std::max(0, (int)std::floor(center.x - radius - 1));
	const int y0 = std::max(0, (int)std::floor(center.y - radius - 1));
	const int z0 = std::max(0, (int)std::floor(center.z - radius - 1));
	const int x1 = std::min(nx - 1, (int)std::ceil(center.x + radius + 1));
	const int y1 = std::min(ny - 1, (int)std::ceil(center.y + radius + 1));
	const int z1 = std::min(nz - 1, (int)std::ceil(center.z + radius + 1));
	int changed = 0;
	const uint8_t *m = material.ptr();
	uint8_t *st = state.ptrw();
	for (int z = z0; z <= z1; z++) {
		for (int y = y0; y <= y1; y++) {
			for (int x = x0; x <= x1; x++) {
				const int i = idx(x, y, z);
				if (!material_bruises(m[i])) {
					continue;
				}
				const Vector3 p(x + 0.5f, y + 0.5f, z + 0.5f);
				if ((p - center).length_squared() > r2) {
					continue;
				}
				const int cur = st[i] & STATE_BRUISE_MASK;
				if (cur < level) {
					st[i] = (st[i] & ~STATE_BRUISE_MASK) | (uint8_t)level;
					changed++;
				}
			}
		}
	}
	return changed;
}

int VoxelVolume::burn_sphere(Vector3 center, float radius, int level) {
	level = std::clamp(level, 0, 3);
	if (nx == 0 || level == 0) {
		return 0;
	}
	const float r2 = radius * radius;
	const int x0 = std::max(0, (int)std::floor(center.x - radius - 1));
	const int y0 = std::max(0, (int)std::floor(center.y - radius - 1));
	const int z0 = std::max(0, (int)std::floor(center.z - radius - 1));
	const int x1 = std::min(nx - 1, (int)std::ceil(center.x + radius + 1));
	const int y1 = std::min(ny - 1, (int)std::ceil(center.y + radius + 1));
	const int z1 = std::min(nz - 1, (int)std::ceil(center.z + radius + 1));
	int changed = 0;
	uint8_t *m = material.ptrw();
	uint8_t *st = state.ptrw();
	for (int z = z0; z <= z1; z++) {
		for (int y = y0; y <= y1; y++) {
			for (int x = x0; x <= x1; x++) {
				const int i = idx(x, y, z);
				if (m[i] == 0) {
					continue;
				}
				const Vector3 p(x + 0.5f, y + 0.5f, z + 0.5f);
				if ((p - center).length_squared() > r2) {
					continue;
				}
				if (m[i] == MAT_HAIR) {
					if (level >= 2) {
						m[i] = 0;
						st[i] = 0;
						changed++;
					}
					continue;
				}
				const int cur = (st[i] & STATE_BURN_MASK) >> STATE_BURN_SHIFT;
				if (cur < level) {
					st[i] = (st[i] & ~STATE_BURN_MASK) | (uint8_t)(level << STATE_BURN_SHIFT);
					changed++;
				}
			}
		}
	}
	return changed;
}

int VoxelVolume::trim_hair_capsule(Vector3 a, Vector3 b, float radius, int keep_length) {
	if (nx == 0) {
		return 0;
	}
	// Hair height: distance (in voxels, 6-connected steps) from the nearest non-hair solid voxel.
	// We compute it lazily only inside the affected box, via a small BFS seeded by non-hair solids.
	const float r2 = radius * radius;
	const int pad = 2 + std::max(0, keep_length);
	const int x0 = std::max(0, (int)std::floor(std::min(a.x, b.x) - radius - pad));
	const int y0 = std::max(0, (int)std::floor(std::min(a.y, b.y) - radius - pad));
	const int z0 = std::max(0, (int)std::floor(std::min(a.z, b.z) - radius - pad));
	const int x1 = std::min(nx - 1, (int)std::ceil(std::max(a.x, b.x) + radius + pad));
	const int y1 = std::min(ny - 1, (int)std::ceil(std::max(a.y, b.y) + radius + pad));
	const int z1 = std::min(nz - 1, (int)std::ceil(std::max(a.z, b.z) + radius + pad));
	const int bx = x1 - x0 + 1, by = y1 - y0 + 1, bz = z1 - z0 + 1;
	if (bx <= 0 || by <= 0 || bz <= 0) {
		return 0;
	}
	std::vector<int> height(bx * by * bz, -1);
	std::queue<int> q;
	auto bidx = [&](int x, int y, int z) { return (x - x0) + bx * ((y - y0) + by * (z - z0)); };
	const uint8_t *m = material.ptr();
	for (int z = z0; z <= z1; z++) {
		for (int y = y0; y <= y1; y++) {
			for (int x = x0; x <= x1; x++) {
				const uint8_t mm = m[idx(x, y, z)];
				if (mm != 0 && mm != MAT_HAIR) {
					height[bidx(x, y, z)] = 0;
					q.push(bidx(x, y, z));
				}
			}
		}
	}
	static const int off[6][3] = { { 1, 0, 0 }, { -1, 0, 0 }, { 0, 1, 0 }, { 0, -1, 0 }, { 0, 0, 1 }, { 0, 0, -1 } };
	while (!q.empty()) {
		const int bi = q.front();
		q.pop();
		const int lz = bi / (bx * by), ly = (bi / bx) % by, lx = bi % bx;
		const int h = height[bi];
		for (auto &o : off) {
			const int x = x0 + lx + o[0], y = y0 + ly + o[1], z = z0 + lz + o[2];
			if (x < x0 || y < y0 || z < z0 || x > x1 || y > y1 || z > z1) {
				continue;
			}
			if (m[idx(x, y, z)] != MAT_HAIR) {
				continue;
			}
			const int bj = bidx(x, y, z);
			if (height[bj] == -1) {
				height[bj] = h + 1;
				q.push(bj);
			}
		}
	}
	int removed = 0;
	uint8_t *mw = material.ptrw();
	uint8_t *st = state.ptrw();
	for (int z = z0; z <= z1; z++) {
		for (int y = y0; y <= y1; y++) {
			for (int x = x0; x <= x1; x++) {
				const int i = idx(x, y, z);
				if (mw[i] != MAT_HAIR) {
					continue;
				}
				const Vector3 p(x + 0.5f, y + 0.5f, z + 0.5f);
				if (dist2_point_segment(p, a, b) > r2) {
					continue;
				}
				const int h = height[bidx(x, y, z)];
				if (h == -1 || h > keep_length) {
					mw[i] = 0;
					st[i] = 0;
					removed++;
				}
			}
		}
	}
	return removed;
}

// ---------------------------------------------------------------------------

Dictionary VoxelVolume::raycast(Vector3 o, Vector3 dir, float max_dist) const {
	// Amanatides-Woo DDA in voxel units. `o` may be outside the volume: the ray
	// is first clipped to the bounding box and the face it enters through
	// becomes the normal if the very first voxel is solid. Distances are in
	// voxel units along the normalised direction.
	Dictionary r;
	r["hit"] = false;
	if (get_voxel_count() == 0 || dir.length_squared() < 1e-12f || !(max_dist > 0.0f)) {
		return r;
	}
	dir = dir.normalized();

	// Clip the ray to the volume's bounding box; remember which slab we entered through.
	float tmin = 0.0f, tmax = max_dist;
	int entry_axis = -1;
	const float bmax[3] = { (float)nx, (float)ny, (float)nz };
	for (int i = 0; i < 3; i++) {
		const float d = dir[i];
		if (std::fabs(d) < 1e-9f) {
			if (o[i] < 0.0f || o[i] > bmax[i]) {
				return r;
			}
			continue;
		}
		float t0 = (0.0f - o[i]) / d;
		float t1 = (bmax[i] - o[i]) / d;
		if (t0 > t1) {
			std::swap(t0, t1);
		}
		if (t0 > tmin) {
			tmin = t0;
			entry_axis = i;
		}
		tmax = std::min(tmax, t1);
		if (tmin > tmax) {
			return r;
		}
	}

	// Start a hair inside the box so floor() picks the entry voxel, not the one before it.
	const float eps = 1e-4f;
	const float t_start = tmin + eps;
	if (t_start > tmax) {
		return r; // the ray only grazes a corner/edge
	}
	const Vector3 p = o + dir * t_start;
	int vx = std::clamp((int)std::floor(p.x), 0, nx - 1);
	int vy = std::clamp((int)std::floor(p.y), 0, ny - 1);
	int vz = std::clamp((int)std::floor(p.z), 0, nz - 1);

	const int stepx = dir.x > 0 ? 1 : -1, stepy = dir.y > 0 ? 1 : -1, stepz = dir.z > 0 ? 1 : -1;
	const float INF = 1e30f;
	const float tdx = std::fabs(dir.x) > 1e-9f ? std::fabs(1.0f / dir.x) : INF;
	const float tdy = std::fabs(dir.y) > 1e-9f ? std::fabs(1.0f / dir.y) : INF;
	const float tdz = std::fabs(dir.z) > 1e-9f ? std::fabs(1.0f / dir.z) : INF;
	// Absolute ray parameter of the next crossing of each axis-aligned plane.
	float tmx = tdx < INF ? t_start + (stepx > 0 ? (vx + 1) - p.x : p.x - vx) * tdx : INF;
	float tmy = tdy < INF ? t_start + (stepy > 0 ? (vy + 1) - p.y : p.y - vy) * tdy : INF;
	float tmz = tdz < INF ? t_start + (stepz > 0 ? (vz + 1) - p.z : p.z - vz) * tdz : INF;

	// Distance to the face through which the current voxel was entered.
	float t = tmin;
	Vector3i normal(0, 0, 0);
	if (entry_axis == 0) {
		normal = Vector3i(-stepx, 0, 0);
	} else if (entry_axis == 1) {
		normal = Vector3i(0, -stepy, 0);
	} else if (entry_axis == 2) {
		normal = Vector3i(0, 0, -stepz);
	}
	const uint8_t *m = material.ptr();
	int guard = nx + ny + nz + 3;
	while (guard-- > 0) {
		const int i = idx(vx, vy, vz);
		if (m[i] != 0) {
			r["hit"] = true;
			r["voxel"] = Vector3i(vx, vy, vz);
			r["normal"] = normal;
			r["distance"] = t;
			r["material"] = (int)m[i];
			r["position"] = o + dir * t;
			return r;
		}
		if (tmx < tmy && tmx < tmz) {
			vx += stepx;
			t = tmx;
			tmx += tdx;
			normal = Vector3i(-stepx, 0, 0);
		} else if (tmy < tmz) {
			vy += stepy;
			t = tmy;
			tmy += tdy;
			normal = Vector3i(0, -stepy, 0);
		} else {
			vz += stepz;
			t = tmz;
			tmz += tdz;
			normal = Vector3i(0, 0, -stepz);
		}
		if (t > max_dist || !in_bounds(vx, vy, vz)) {
			break;
		}
	}
	return r;
}

// ---------------------------------------------------------------------------

TypedArray<VoxelVolume> VoxelVolume::detach_disconnected(Vector3i anchor) {
	TypedArray<VoxelVolume> result;
	const int n = get_voxel_count();
	if (n == 0) {
		return result;
	}
	std::vector<int32_t> label(n, -1);
	std::vector<int> comp_size;
	const uint8_t *m = material.ptr();
	static const int off[6][3] = { { 1, 0, 0 }, { -1, 0, 0 }, { 0, 1, 0 }, { 0, -1, 0 }, { 0, 0, 1 }, { 0, 0, -1 } };
	std::vector<int> stack;
	for (int seed = 0; seed < n; seed++) {
		if (m[seed] == 0 || label[seed] != -1) {
			continue;
		}
		const int32_t id = (int32_t)comp_size.size();
		comp_size.push_back(0);
		stack.clear();
		stack.push_back(seed);
		label[seed] = id;
		while (!stack.empty()) {
			const int i = stack.back();
			stack.pop_back();
			comp_size[id]++;
			const int z = i / (nx * ny), y = (i / nx) % ny, x = i % nx;
			for (auto &o : off) {
				const int ax = x + o[0], ay = y + o[1], az = z + o[2];
				if (!in_bounds(ax, ay, az)) {
					continue;
				}
				const int j = idx(ax, ay, az);
				if (m[j] != 0 && label[j] == -1) {
					label[j] = id;
					stack.push_back(j);
				}
			}
		}
	}
	if (comp_size.size() <= 1) {
		return result;
	}
	int keep = -1;
	if (in_bounds(anchor.x, anchor.y, anchor.z) && m[idx(anchor.x, anchor.y, anchor.z)] != 0) {
		keep = label[idx(anchor.x, anchor.y, anchor.z)];
	} else {
		int best = -1;
		for (size_t c = 0; c < comp_size.size(); c++) {
			if (comp_size[c] > best) {
				best = comp_size[c];
				keep = (int)c;
			}
		}
	}
	// bounds per component
	const int C = (int)comp_size.size();
	std::vector<Vector3i> cmin(C, Vector3i(nx, ny, nz)), cmax(C, Vector3i(-1, -1, -1));
	for (int i = 0; i < n; i++) {
		const int c = label[i];
		if (c < 0 || c == keep) {
			continue;
		}
		const int z = i / (nx * ny), y = (i / nx) % ny, x = i % nx;
		cmin[c] = Vector3i(std::min(cmin[c].x, x), std::min(cmin[c].y, y), std::min(cmin[c].z, z));
		cmax[c] = Vector3i(std::max(cmax[c].x, x), std::max(cmax[c].y, y), std::max(cmax[c].z, z));
	}
	std::vector<Ref<VoxelVolume>> pieces(C);
	for (int c = 0; c < C; c++) {
		if (c == keep) {
			continue;
		}
		Ref<VoxelVolume> v;
		v.instantiate();
		const Vector3i d = cmax[c] - cmin[c] + Vector3i(1, 1, 1);
		v->create(d, voxel_size);
		v->origin = origin + Vector3(cmin[c].x, cmin[c].y, cmin[c].z) * voxel_size;
		v->part_name = part_name + "_piece" + String::num_int64(c);
		v->bone = "";
		pieces[c] = v;
	}
	uint8_t *mw = material.ptrw();
	uint8_t *cw = color.ptrw();
	uint8_t *sw = state.ptrw();
	for (int i = 0; i < n; i++) {
		const int c = label[i];
		if (c < 0 || c == keep) {
			continue;
		}
		const int z = i / (nx * ny), y = (i / nx) % ny, x = i % nx;
		Ref<VoxelVolume> &v = pieces[c];
		const int j = v->idx(x - cmin[c].x, y - cmin[c].y, z - cmin[c].z);
		v->material[j] = mw[i];
		v->color[j] = cw[i];
		v->state[j] = sw[i];
		mw[i] = 0;
		cw[i] = 0;
		sw[i] = 0;
	}
	for (int c = 0; c < C; c++) {
		if (c != keep) {
			result.push_back(pieces[c]);
		}
	}
	return result;
}

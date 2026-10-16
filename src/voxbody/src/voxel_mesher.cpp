// VoxelMesher implementation. See voxel_mesher.h for the contract.
//
// Algorithm outline (per axis d, with u = (d+1)%3 and v = (d+2)%3):
//   for every plane k in 0..n[d] (planes sit between voxel k-1 and voxel k)
//     build a mask of nu*nv face keys: a face exists where exactly one of
//     the two voxels is solid; the key packs direction, color, material,
//     state and the four corner AO levels;
//     greedily merge runs of equal keys into rectangles and emit quads.
// Solidity is read from a zero-padded byte grid so no bounds checks are needed
// in the inner loops. Everything works on raw pointers; Variant is touched only
// when the finished arrays are handed to the ArrayMesh.

#include "voxel_mesher.h"

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/packed_int32_array.hpp>
#include <godot_cpp/variant/packed_vector2_array.hpp>
#include <godot_cpp/variant/packed_vector3_array.hpp>

#include <cstring>
#include <vector>

using namespace godot;

void VoxelMesher::_bind_methods() {
	ClassDB::bind_static_method("VoxelMesher", D_METHOD("build_mesh", "volume", "palette", "options"), &VoxelMesher::build_mesh, DEFVAL(Dictionary()));
	ClassDB::bind_static_method("VoxelMesher", D_METHOD("tint_color", "base", "state"), &VoxelMesher::tint_color);
	ClassDB::bind_static_method("VoxelMesher", D_METHOD("ao_factor", "level", "ao_strength"), &VoxelMesher::ao_factor);
}

// ---------------------------------------------------------------------------
// colour helpers

static inline Color lerp_rgb(const Color &a, const Color &b, float w) {
	return Color(a.r + (b.r - a.r) * w, a.g + (b.g - a.g) * w, a.b + (b.b - a.b) * w, 1.0f);
}

static inline Color rgb8(int r, int g, int b) {
	return Color(r / 255.0f, g / 255.0f, b / 255.0f, 1.0f);
}

Color VoxelMesher::tint_color(const Color &base, int state) {
	// Table from docs/spec/voxbody-api.md. Order: bruise, burn, then blood/cut on top.
	static const Color BRUISE[3] = { rgb8(217, 89, 89), rgb8(115, 64, 140), rgb8(153, 166, 77) };
	static const float BRUISE_W[3] = { 0.45f, 0.55f, 0.45f };
	static const Color BURN[3] = { rgb8(204, 77, 51), rgb8(230, 191, 153), rgb8(26, 20, 18) };
	static const float BURN_W[3] = { 0.5f, 0.6f, 0.9f };
	static const Color BLOOD = rgb8(115, 8, 8);

	Color c(base.r, base.g, base.b, 1.0f);
	const int bruise = state & VoxelVolume::STATE_BRUISE_MASK;
	const int burn = (state & VoxelVolume::STATE_BURN_MASK) >> VoxelVolume::STATE_BURN_SHIFT;
	if (bruise > 0) {
		c = lerp_rgb(c, BRUISE[bruise - 1], BRUISE_W[bruise - 1]);
	}
	if (burn > 0) {
		c = lerp_rgb(c, BURN[burn - 1], BURN_W[burn - 1]);
	}
	if (state & VoxelVolume::STATE_BLOOD) {
		c = lerp_rgb(c, BLOOD, 0.65f);
	} else if (state & VoxelVolume::STATE_CUT) {
		c = Color(c.r * 0.85f, c.g * 0.85f, c.b * 0.85f, 1.0f);
	}
	return c;
}

float VoxelMesher::ao_factor(int level, float ao_strength) {
	if (level < 0) {
		level = 0;
	} else if (level > 3) {
		level = 3;
	}
	return 1.0f - ao_strength * (float)(3 - level) / 3.0f;
}

// ---------------------------------------------------------------------------
// mesh builder

namespace {

// Face key layout (uint64): bit 0 = present, bit 1 = positive direction,
// bits 8..15 color, 16..23 material, 24..31 state, 32..39 ao0..ao3 (2 bits each).
constexpr uint64_t KEY_PRESENT = 1ull;
constexpr uint64_t KEY_POSITIVE = 2ull;

inline int vertex_ao(int side1, int side2, int corner) {
	// 0fps: a corner is fully occluded when both sides are solid.
	if (side1 && side2) {
		return 0;
	}
	return 3 - (side1 + side2 + corner);
}

struct MeshBuffers {
	std::vector<Vector3> vertices;
	std::vector<Vector3> normals;
	std::vector<Color> colors;
	std::vector<Vector2> uvs;
	std::vector<int32_t> indices;
};

} // namespace

Ref<ArrayMesh> VoxelMesher::build_mesh(const Ref<VoxelVolume> &volume, const PackedColorArray &palette, const Dictionary &options) {
	Ref<ArrayMesh> mesh;
	mesh.instantiate();
	if (volume.is_null()) {
		return mesh;
	}
	const int n[3] = { volume->width(), volume->height(), volume->depth() };
	if (n[0] <= 0 || n[1] <= 0 || n[2] <= 0) {
		return mesh;
	}

	const bool ao_enabled = (bool)options.get("ao", true);
	const float ao_strength = (float)(double)options.get("ao_strength", 0.45);
	float ao_lut[4];
	for (int l = 0; l < 4; l++) {
		ao_lut[l] = ao_enabled ? ao_factor(l, ao_strength) : 1.0f;
	}

	const uint8_t *mat = volume->material_ptr();
	const uint8_t *col = volume->color_ptr();
	const uint8_t *st = volume->state_ptr();
	const Vector3 origin = volume->get_origin();
	const float vs = volume->get_voxel_size();
	const Color *pal = palette.ptr();
	const int pal_size = (int)palette.size();
	// Palette entries that do not exist render mid-grey so a missing palette is
	// still visible (and obviously wrong) instead of crashing.
	const Color missing(0.75f, 0.75f, 0.75f, 1.0f);

	// Zero-padded solidity grid: index (x+1) + px*((y+1) + py*(z+1)).
	const int px = n[0] + 2, py = n[1] + 2, pz = n[2] + 2;
	std::vector<uint8_t> solid((size_t)px * py * pz, 0);
	{
		const uint8_t *m = mat;
		for (int z = 0; z < n[2]; z++) {
			for (int y = 0; y < n[1]; y++) {
				uint8_t *row = &solid[(size_t)1 + px * ((size_t)(y + 1) + (size_t)py * (z + 1))];
				for (int x = 0; x < n[0]; x++) {
					row[x] = *m++ != 0;
				}
			}
		}
	}
	const int pstride[3] = { 1, px, px * py };
	const int vstride[3] = { 1, n[0], n[0] * n[1] };

	MeshBuffers out;
	std::vector<uint64_t> mask;

	for (int d = 0; d < 3; d++) {
		const int u = (d + 1) % 3;
		const int v = (d + 2) % 3;
		const int nu = n[u], nv = n[v];
		mask.assign((size_t)nu * nv, 0);
		const int ps_d = pstride[d], ps_u = pstride[u], ps_v = pstride[v];

		for (int k = 0; k <= n[d]; k++) {
			// --- build the face mask for plane k ---------------------------
			bool any = false;
			for (int j = 0; j < nv; j++) {
				// padded index of voxel (k, 0, j) in (d, u, v) order
				const int base_p = pstride[0] + pstride[1] + pstride[2] + k * ps_d + j * ps_v;
				uint64_t *mrow = &mask[(size_t)j * nu];
				for (int i = 0; i < nu; i++) {
					const int pi = base_p + i * ps_u; // voxel k (front)
					const int front_solid = solid[pi];
					const int back_solid = solid[pi - ps_d]; // voxel k-1 (padding handles k == 0)
					if (front_solid == back_solid) {
						mrow[i] = 0;
						continue;
					}
					const bool positive = back_solid != 0; // face of voxel k-1 facing +d
					const int owner_k = positive ? k - 1 : k;
					const int vi = owner_k * vstride[d] + i * vstride[u] + j * vstride[v];
					// Outside layer: the air voxel in front of the face.
					const int q = positive ? pi : pi - ps_d;
					int ao0 = 3, ao1 = 3, ao2 = 3, ao3 = 3;
					if (ao_enabled) {
						const int s_un = solid[q - ps_u], s_up = solid[q + ps_u];
						const int s_vn = solid[q - ps_v], s_vp = solid[q + ps_v];
						const int c_nn = solid[q - ps_u - ps_v], c_pn = solid[q + ps_u - ps_v];
						const int c_np = solid[q - ps_u + ps_v], c_pp = solid[q + ps_u + ps_v];
						ao0 = vertex_ao(s_un, s_vn, c_nn);
						ao1 = vertex_ao(s_up, s_vn, c_pn);
						ao2 = vertex_ao(s_un, s_vp, c_np);
						ao3 = vertex_ao(s_up, s_vp, c_pp);
					}
					uint64_t key = KEY_PRESENT | (positive ? KEY_POSITIVE : 0ull);
					key |= (uint64_t)col[vi] << 8;
					key |= (uint64_t)mat[vi] << 16;
					key |= (uint64_t)st[vi] << 24;
					key |= (uint64_t)ao0 << 32 | (uint64_t)ao1 << 34 | (uint64_t)ao2 << 36 | (uint64_t)ao3 << 38;
					mrow[i] = key;
					any = true;
				}
			}
			if (!any) {
				continue;
			}

			// --- greedy merge -------------------------------------------------
			for (int j = 0; j < nv; j++) {
				uint64_t *mrow = &mask[(size_t)j * nu];
				for (int i = 0; i < nu;) {
					const uint64_t key = mrow[i];
					if (key == 0) {
						i++;
						continue;
					}
					int w = 1;
					while (i + w < nu && mrow[i + w] == key) {
						w++;
					}
					int h = 1;
					for (; j + h < nv; h++) {
						const uint64_t *nrow = &mask[(size_t)(j + h) * nu];
						bool ok = true;
						for (int q = 0; q < w; q++) {
							if (nrow[i + q] != key) {
								ok = false;
								break;
							}
						}
						if (!ok) {
							break;
						}
					}

					// --- emit quad ----------------------------------------------
					const bool positive = (key & KEY_POSITIVE) != 0;
					const int color_idx = (int)((key >> 8) & 0xFF);
					const int material = (int)((key >> 16) & 0xFF);
					const int state = (int)((key >> 24) & 0xFF);
					const int ao[4] = { (int)((key >> 32) & 3), (int)((key >> 34) & 3), (int)((key >> 36) & 3), (int)((key >> 38) & 3) };

					const Color base = (color_idx < pal_size) ? pal[color_idx] : missing;
					const Color tinted = tint_color(base, state);
					const Vector2 uv((float)material / 255.0f, (float)state / 255.0f);
					Vector3 normal;
					normal[d] = positive ? 1.0f : -1.0f;

					// Corners in voxel units: 0 = (i, j), 1 = (i+w, j), 2 = (i, j+h), 3 = (i+w, j+h).
					Vector3 corner[4];
					for (int c = 0; c < 4; c++) {
						Vector3 p;
						p[d] = (float)k;
						p[u] = (float)(i + ((c & 1) ? w : 0));
						p[v] = (float)(j + ((c & 2) ? h : 0));
						corner[c] = origin + p * vs;
					}
					const int32_t b = (int32_t)out.vertices.size();
					for (int c = 0; c < 4; c++) {
						out.vertices.push_back(corner[c]);
						out.normals.push_back(normal);
						out.colors.push_back(Color(tinted.r, tinted.g, tinted.b, ao_lut[ao[c]]));
						out.uvs.push_back(uv);
					}
					// Triangles are listed counter-clockwise as seen from +d. The
					// diagonal goes through the brighter vertex pair (0fps flip rule):
					// default diagonal 1-2, flipped to 0-3 when ao0 + ao3 > ao1 + ao2.
					int32_t tri[6];
					if (ao[0] + ao[3] > ao[1] + ao[2]) {
						tri[0] = 0; tri[1] = 1; tri[2] = 3;
						tri[3] = 0; tri[4] = 3; tri[5] = 2;
					} else {
						tri[0] = 0; tri[1] = 1; tri[2] = 2;
						tri[3] = 1; tri[4] = 3; tri[5] = 2;
					}
					// Godot front faces are clockwise, so a +d face (seen from +d)
					// needs the reversed order; a -d face keeps it.
					if (positive) {
						std::swap(tri[1], tri[2]);
						std::swap(tri[4], tri[5]);
					}
					for (int t = 0; t < 6; t++) {
						out.indices.push_back(b + tri[t]);
					}

					// clear the merged rectangle
					for (int q = 0; q < h; q++) {
						uint64_t *crow = &mask[(size_t)(j + q) * nu];
						for (int p = 0; p < w; p++) {
							crow[i + p] = 0;
						}
					}
					i += w;
				}
			}
		}
	}

	if (out.indices.empty()) {
		return mesh;
	}

	// --- hand over to Godot ------------------------------------------------
	const int64_t nverts = (int64_t)out.vertices.size();
	PackedVector3Array vertices;
	vertices.resize(nverts);
	std::memcpy(vertices.ptrw(), out.vertices.data(), sizeof(Vector3) * nverts);
	PackedVector3Array normals;
	normals.resize(nverts);
	std::memcpy(normals.ptrw(), out.normals.data(), sizeof(Vector3) * nverts);
	PackedColorArray colors;
	colors.resize(nverts);
	std::memcpy(colors.ptrw(), out.colors.data(), sizeof(Color) * nverts);
	PackedVector2Array uvs;
	uvs.resize(nverts);
	std::memcpy(uvs.ptrw(), out.uvs.data(), sizeof(Vector2) * nverts);
	PackedInt32Array indices;
	indices.resize((int64_t)out.indices.size());
	std::memcpy(indices.ptrw(), out.indices.data(), sizeof(int32_t) * out.indices.size());

	Array arrays;
	arrays.resize(Mesh::ARRAY_MAX);
	arrays[Mesh::ARRAY_VERTEX] = vertices;
	arrays[Mesh::ARRAY_NORMAL] = normals;
	arrays[Mesh::ARRAY_COLOR] = colors;
	arrays[Mesh::ARRAY_TEX_UV] = uvs;
	arrays[Mesh::ARRAY_INDEX] = indices;
	mesh->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, arrays);
	return mesh;
}

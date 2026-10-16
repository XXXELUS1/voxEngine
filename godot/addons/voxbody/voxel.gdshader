// voxel.gdshader -- surface shader for meshes built by VoxelMesher.
//
// Contract with the mesher (docs/spec/voxbody-api.md, section "Шейдер"):
//   COLOR.rgb : sRGB albedo, already tinted by the voxel state -> converted to linear here
//   COLOR.a   : per-vertex ambient occlusion 0.55..1.0          -> AO built-in
//   UV.x*255  : material id,  UV.y*255 : state byte              -> roughness rules
//   per-voxel colour noise from a hash of the voxel cell, which is found from the
//   local vertex position and the `voxel_size` uniform (VoxelPart sets it per part).
//
// Compiles on both Forward+ and Compatibility: only GLSL ES 3.0 level features
// are used (no texture queries, no SSS/backlight, integer math on uint only).
// Look target is "Teardown-like": soft, slightly desaturated, subtle noise.
shader_type spatial;
render_mode cull_back, depth_draw_opaque, diffuse_burley, specular_schlick_ggx;

// Edge length of one voxel in metres. VoxelPart duplicates the material and
// overrides this when a part's voxel size differs from the value stored here.
uniform float voxel_size : hint_range(0.0005, 0.1) = 0.005;
// Roughness of plain (dry, clean) voxels.
uniform float roughness : hint_range(0.0, 1.0) = 0.85;
// Roughness for voxels carrying the BLOOD / WET state bits.
uniform float blood_roughness : hint_range(0.0, 1.0) = 0.25;
uniform float wet_roughness : hint_range(0.0, 1.0) = 0.3;
// Lab decision (spec is silent): naturally glossy materials (eye, tooth, metal,
// glass, ceramic) use this roughness so eyes and teeth read as wet/hard.
uniform float gloss_roughness : hint_range(0.0, 1.0) = 0.35;
// Per-voxel noise: +-noise_strength on brightness, +-hue_noise (in turns) on hue.
uniform float noise_strength : hint_range(0.0, 0.5) = 0.06;
uniform float hue_noise : hint_range(0.0, 0.2) = 0.02;
// How much the baked AO also darkens direct light (0 = ambient only).
uniform float ao_light_affect : hint_range(0.0, 1.0) = 0.6;

varying vec3 v_local;        // vertex position in the part's local space (metres)
varying flat vec3 v_normal;  // face normal in local space (constant per face)

const int STATE_BLOOD = 32;  // 0x20
const int STATE_WET = 64;    // 0x40
const int MAT_TOOTH = 5;
const int MAT_EYE = 9;
const int MAT_METAL = 33;
const int MAT_GLASS = 37;
const int MAT_CERAMIC = 38;
// Added to the cell id before hashing so every realistic cell is non-negative.
const int CELL_OFFSET = 1048576;
// Pull the sample this far inside the voxel so a face maps to its own cell, not to the air cell.
const float FACE_INSET_M = 0.0001;

// Exact piecewise sRGB -> linear transfer.
vec3 srgb_to_linear(vec3 c) {
    vec3 lo = c / 12.92;
    vec3 hi = pow((c + 0.055) / 1.055, vec3(2.4));
    return mix(lo, hi, step(vec3(0.04045), c));
}

// lowbias32 integer hash (Chris Wellons), good enough for visual noise.
uint hash_u32(uint x) {
    x ^= x >> 16u;
    x *= 2135587861u;
    x ^= x >> 15u;
    x *= 2219398423u;
    x ^= x >> 16u;
    return x;
}

// Deterministic value in 0..1 for a voxel cell and a salt.
float cell_noise(ivec3 cell, uint salt) {
    uvec3 c = uvec3(cell + ivec3(CELL_OFFSET));
    uint h = hash_u32(c.x * 2654435761u ^ hash_u32(c.y * 2246822519u ^ hash_u32(c.z * 3266489917u ^ salt)));
    return float(h & 16777215u) / 16777215.0;
}

// Rotate a colour around the grey axis (Rodrigues' formula); angle in radians.
vec3 hue_rotate(vec3 c, float angle) {
    const vec3 k = vec3(0.57735027);
    float cs = cos(angle);
    float sn = sin(angle);
    return c * cs + cross(k, c) * sn + k * dot(k, c) * (1.0 - cs);
}

void vertex() {
    v_local = VERTEX;
    v_normal = NORMAL;
}

void fragment() {
    int material_id = int(round(UV.x * 255.0));
    int state = int(round(UV.y * 255.0));

    ivec3 cell = ivec3(floor((v_local - FACE_INSET_M * v_normal) / voxel_size));
    float value_noise = cell_noise(cell, 0u) * 2.0 - 1.0;
    float hue_shift = cell_noise(cell, 1u) * 2.0 - 1.0;

    // Noise is applied in sRGB (perceptual) space, then converted to linear.
    vec3 albedo_srgb = hue_rotate(COLOR.rgb, hue_shift * hue_noise * TAU);
    albedo_srgb *= 1.0 + value_noise * noise_strength;
    ALBEDO = srgb_to_linear(clamp(albedo_srgb, 0.0, 1.0));

    AO = COLOR.a;
    AO_LIGHT_AFFECT = ao_light_affect;

    float r = roughness;
    if (material_id == MAT_EYE || material_id == MAT_TOOTH || material_id == MAT_METAL
            || material_id == MAT_GLASS || material_id == MAT_CERAMIC) {
        r = gloss_roughness;
    }
    if ((state & STATE_WET) != 0) {
        r = wet_roughness;
    }
    if ((state & STATE_BLOOD) != 0) {
        r = blood_roughness;
    }
    ROUGHNESS = r;
    METALLIC = material_id == MAT_METAL ? 0.7 : 0.0;
    SPECULAR = 0.5;
}

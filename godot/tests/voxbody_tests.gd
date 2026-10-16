extends RefCounted
## Test cases for the voxbody extension, run by res://tests/run_tests.gd.
## Statically typed against the extension classes, so this file only parses
## when the extension is loaded; the runner reports that case separately.

const HEAD_MODEL_PATH := "res://assets/models/head_test.vxm"
const ROUNDTRIP_PATH := "user://test_roundtrip.vxm"
const VOXEL_SIZE_M := 0.01
const CUBE := 8
const CUBE_FACES := 6
const VERTICES_PER_QUAD := 4
const INDICES_PER_QUAD := 6
const EPS := 1e-4
const PALETTE_SIZE := 256
const SKIN_COLOR_INDEX := 1
const BONE_COLOR_INDEX := 2
## head_test landmarks in the part's local space (metres): the forehead sits at
## this height; the test ray comes from in front of the face (-Z) towards +Z.
const FOREHEAD_Y_M := 0.165
const FOREHEAD_HALF_WIDTH_M := 0.04
const FOREHEAD_CUT_RADIUS_M := 0.006
const FOREHEAD_CUT_DEPTH_M := 0.004
const RAY_START_Z_M := -0.5
const RAY_LENGTH_M := 1.0
const MIN_HEAD_SOLID := 1000
const RAY_GAP_M := 1.0

var _check: Callable
var _begin: Callable
var _root: Window


func run(check: Callable, begin_case: Callable, root: Window) -> void:
	_check = check
	_begin = begin_case
	_root = root
	for case: Callable in [
		test_volume_basics,
		test_carve_marks_cut,
		test_raycast,
		test_detach_disconnected,
		test_mesher_cube,
		test_model_roundtrip,
		test_part_in_tree,
		test_head_model,
	]:
		case.call()


# --- helpers ---------------------------------------------------------------

func check(condition: bool, check_name: String) -> bool:
	return _check.call(condition, check_name)


func check_approx(actual: float, expected: float, check_name: String) -> bool:
	return check(absf(actual - expected) <= EPS, "%s (%s ~ %s)" % [check_name, actual, expected])


func begin(case_name: String) -> void:
	_begin.call(case_name)


## A cube volume of `size` voxels per side, fully solid with one material/colour.
func filled_volume(size: int, material: int, color: int) -> VoxelVolume:
	var vol := VoxelVolume.new()
	vol.create(Vector3i(size, size, size), VOXEL_SIZE_M)
	var count := vol.get_voxel_count()
	var materials := PackedByteArray()
	materials.resize(count)
	materials.fill(material)
	vol.set_material_data(materials)
	var colors := PackedByteArray()
	colors.resize(count)
	colors.fill(color)
	vol.set_color_data(colors)
	return vol


func fill_box(vol: VoxelVolume, lo: Vector3i, hi: Vector3i, material: int, color: int) -> void:
	for z in range(lo.z, hi.z + 1):
		for y in range(lo.y, hi.y + 1):
			for x in range(lo.x, hi.x + 1):
				vol.set_voxel(Vector3i(x, y, z), material, color)


func test_palette() -> PackedColorArray:
	var palette := PackedColorArray()
	palette.resize(PALETTE_SIZE)
	palette.fill(Color.BLACK)
	palette[SKIN_COLOR_INDEX] = Color8(224, 180, 150)
	palette[BONE_COLOR_INDEX] = Color8(235, 228, 210)
	return palette


# --- cases ------------------------------------------------------------------

func test_volume_basics() -> void:
	begin("VoxelVolume create / fill / count")
	var vol := VoxelVolume.new()
	vol.create(Vector3i(4, 5, 6), VOXEL_SIZE_M)
	check(vol.get_dims() == Vector3i(4, 5, 6), "dims after create")
	check(vol.get_voxel_count() == 4 * 5 * 6, "voxel count")
	check(vol.count_solid() == 0, "empty after create")
	check_approx(vol.voxel_size, VOXEL_SIZE_M, "voxel_size property")
	vol.set_voxel(Vector3i(1, 2, 3), VoxelVolume.MAT_SKIN, SKIN_COLOR_INDEX)
	vol.set_voxel(Vector3i(3, 4, 5), VoxelVolume.MAT_BONE, BONE_COLOR_INDEX)
	check(vol.count_solid() == 2, "two solid voxels after set_voxel")
	check(vol.get_material(Vector3i(1, 2, 3)) == VoxelVolume.MAT_SKIN, "material readback")
	check(vol.get_color(Vector3i(1, 2, 3)) == SKIN_COLOR_INDEX, "color readback")
	check(vol.get_material(Vector3i(9, 9, 9)) == VoxelVolume.MAT_AIR, "out of bounds reads air")
	check(vol.get_solid_min() == Vector3i(1, 2, 3) and vol.get_solid_max() == Vector3i(3, 4, 5), "solid bounds")
	vol.set_state(Vector3i(1, 2, 3), VoxelVolume.STATE_WET)
	check(vol.get_state(Vector3i(1, 2, 3)) == VoxelVolume.STATE_WET, "state readback")
	vol.set_voxel(Vector3i(1, 2, 3), VoxelVolume.MAT_AIR, 0)
	check(vol.count_solid() == 1 and vol.get_state(Vector3i(1, 2, 3)) == 0, "clearing a voxel clears its state")
	var full := filled_volume(CUBE, VoxelVolume.MAT_SKIN, SKIN_COLOR_INDEX)
	check(full.count_solid() == CUBE * CUBE * CUBE, "filled cube counts every voxel")
	var copy := full.duplicate_volume()
	copy.set_voxel(Vector3i.ZERO, VoxelVolume.MAT_AIR, 0)
	check(full.count_solid() == CUBE * CUBE * CUBE and copy.count_solid() == CUBE * CUBE * CUBE - 1, "duplicate_volume is a deep copy")


func test_carve_marks_cut() -> void:
	begin("carve_capsule marks STATE_CUT on neighbours")
	const SIZE := 16
	var vol := filled_volume(SIZE, VoxelVolume.MAT_SKIN, SKIN_COLOR_INDEX)
	# Segment through voxel centres along X at y = z = 8, radius 1.5 voxels.
	var removed := vol.carve_capsule(Vector3(3.5, 8.5, 8.5), Vector3(12.5, 8.5, 8.5), 1.5)
	check(removed > 0, "removed %d voxels" % removed)
	check(vol.count_solid() == SIZE * SIZE * SIZE - removed, "count_solid matches the removed count")
	check(vol.get_material(Vector3i(8, 8, 8)) == VoxelVolume.MAT_AIR, "voxel on the segment is gone")
	check(vol.get_material(Vector3i(8, 9, 8)) == VoxelVolume.MAT_AIR, "voxel 1 away is gone")
	var neighbour_state := vol.get_state(Vector3i(8, 10, 8))
	check(vol.get_material(Vector3i(8, 10, 8)) == VoxelVolume.MAT_SKIN, "voxel 2 away stays")
	check(neighbour_state & VoxelVolume.STATE_CUT != 0, "exposed neighbour has STATE_CUT")
	check(neighbour_state & VoxelVolume.STATE_BLOOD != 0, "exposed skin neighbour has STATE_BLOOD")
	check(vol.get_state(Vector3i(8, 11, 8)) == 0, "voxel not touching the cut keeps state 0")
	var untouched := filled_volume(SIZE, VoxelVolume.MAT_SKIN, SKIN_COLOR_INDEX)
	untouched.carve_sphere(Vector3(8.5, 8.5, 8.5), 1.5, false)
	check(untouched.get_state(Vector3i(8, 10, 8)) == 0, "mark_wound=false leaves no state")


func test_raycast() -> void:
	begin("raycast from outside")
	var vol := filled_volume(CUBE, VoxelVolume.MAT_SKIN, SKIN_COLOR_INDEX)
	var hit: Dictionary = vol.raycast(Vector3(-5.0, 4.5, 4.5), Vector3(1.0, 0.0, 0.0), 100.0)
	check(hit["hit"], "ray along +X hits")
	check(hit["voxel"] == Vector3i(0, 4, 4), "hit voxel (0,4,4)")
	check(hit["normal"] == Vector3i(-1, 0, 0), "normal faces the ray (-1,0,0)")
	check_approx(hit["distance"], 5.0, "distance to the entry face")
	check(hit["material"] == VoxelVolume.MAT_SKIN, "material of the hit voxel")
	var from_above: Dictionary = vol.raycast(Vector3(4.5, 20.0, 4.5), Vector3(0.0, -1.0, 0.0), 100.0)
	check(from_above["hit"] and from_above["voxel"] == Vector3i(4, 7, 4) and from_above["normal"] == Vector3i(0, 1, 0), "ray from above hits the top face")
	var miss: Dictionary = vol.raycast(Vector3(-5.0, 20.0, 4.5), Vector3(1.0, 0.0, 0.0), 100.0)
	check(not miss["hit"], "ray above the volume misses")
	var short: Dictionary = vol.raycast(Vector3(-5.0, 4.5, 4.5), Vector3(1.0, 0.0, 0.0), 2.0)
	check(not short["hit"], "ray shorter than the gap misses")
	vol.set_voxel(Vector3i(0, 4, 4), VoxelVolume.MAT_AIR, 0)
	var deeper: Dictionary = vol.raycast(Vector3(-5.0, 4.5, 4.5), Vector3(1.0, 0.0, 0.0), 100.0)
	check(deeper["hit"] and deeper["voxel"] == Vector3i(1, 4, 4) and absf(float(deeper["distance"]) - 6.0) <= EPS, "ray passes an air voxel and hits the next one")


func test_detach_disconnected() -> void:
	begin("detach_disconnected splits two blobs")
	const SIZE := 16
	var vol := VoxelVolume.new()
	vol.create(Vector3i(SIZE, SIZE, SIZE), VOXEL_SIZE_M)
	vol.origin = Vector3(-0.1, 0.0, 0.2)
	fill_box(vol, Vector3i(2, 2, 2), Vector3i(5, 5, 5), VoxelVolume.MAT_BONE, BONE_COLOR_INDEX)
	fill_box(vol, Vector3i(10, 10, 10), Vector3i(13, 13, 13), VoxelVolume.MAT_SKIN, SKIN_COLOR_INDEX)
	const BLOB := 4 * 4 * 4
	check(vol.count_solid() == 2 * BLOB, "two blobs filled")
	var pieces: Array[VoxelVolume] = vol.detach_disconnected(Vector3i(3, 3, 3))
	check(pieces.size() == 1, "one piece detached")
	check(vol.count_solid() == BLOB, "anchored blob stays")
	check(vol.get_material(Vector3i(3, 3, 3)) == VoxelVolume.MAT_BONE and vol.get_material(Vector3i(11, 11, 11)) == VoxelVolume.MAT_AIR, "the right blob stayed")
	if pieces.size() == 1:
		var piece := pieces[0]
		check(piece.count_solid() == BLOB, "piece holds the other blob")
		check(piece.get_dims() == Vector3i(4, 4, 4), "piece is cropped to its bounds")
		check(piece.get_material(Vector3i(0, 0, 0)) == VoxelVolume.MAT_SKIN, "piece keeps materials")
		var expected_origin := vol.origin + Vector3(10, 10, 10) * VOXEL_SIZE_M
		check(piece.origin.is_equal_approx(expected_origin), "piece origin keeps the local space")
		check_approx(piece.voxel_size, VOXEL_SIZE_M, "piece voxel size")
	var nothing: Array[VoxelVolume] = vol.detach_disconnected(Vector3i(3, 3, 3))
	check(nothing.is_empty(), "nothing to detach the second time")
	var keep_largest := VoxelVolume.new()
	keep_largest.create(Vector3i(SIZE, SIZE, SIZE), VOXEL_SIZE_M)
	fill_box(keep_largest, Vector3i(0, 0, 0), Vector3i(5, 5, 5), VoxelVolume.MAT_BONE, BONE_COLOR_INDEX)
	fill_box(keep_largest, Vector3i(12, 12, 12), Vector3i(13, 13, 13), VoxelVolume.MAT_SKIN, SKIN_COLOR_INDEX)
	var small: Array[VoxelVolume] = keep_largest.detach_disconnected(Vector3i(-1, -1, -1))
	check(small.size() == 1 and small[0].count_solid() == 8 and keep_largest.count_solid() == 6 * 6 * 6, "no anchor keeps the largest component")


func test_mesher_cube() -> void:
	begin("VoxelMesher.build_mesh on an 8^3 cube")
	var vol := filled_volume(CUBE, VoxelVolume.MAT_SKIN, SKIN_COLOR_INDEX)
	vol.origin = Vector3(-0.04, 0.0, -0.04)
	var mesh: ArrayMesh = VoxelMesher.build_mesh(vol, test_palette())
	check(mesh != null, "mesh returned")
	if mesh == null:
		return
	check(mesh.get_surface_count() == 1, "one surface")
	if mesh.get_surface_count() != 1:
		return
	var arrays: Array = mesh.surface_get_arrays(0)
	var vertices: PackedVector3Array = arrays[Mesh.ARRAY_VERTEX]
	var indices: PackedInt32Array = arrays[Mesh.ARRAY_INDEX]
	var normals: PackedVector3Array = arrays[Mesh.ARRAY_NORMAL]
	var colors: PackedColorArray = arrays[Mesh.ARRAY_COLOR]
	var uvs: PackedVector2Array = arrays[Mesh.ARRAY_TEX_UV]
	check(vertices.size() == CUBE_FACES * VERTICES_PER_QUAD, "24 vertices (%d)" % vertices.size())
	check(indices.size() == CUBE_FACES * INDICES_PER_QUAD, "36 indices (%d)" % indices.size())
	check(normals.size() == vertices.size() and colors.size() == vertices.size() and uvs.size() == vertices.size(), "normal/colour/uv per vertex")
	var side_m := CUBE * VOXEL_SIZE_M
	var in_bounds := true
	for v in vertices:
		var local := v - vol.origin
		if local.x < -EPS or local.y < -EPS or local.z < -EPS or local.x > side_m + EPS or local.y > side_m + EPS or local.z > side_m + EPS:
			in_bounds = false
	check(in_bounds, "vertices are in local metres (origin + voxel * size)")
	if colors.size() > 0 and uvs.size() > 0:
		check(colors[0].a > 0.99, "open corners have AO 1.0")
		check(colors[0].is_equal_approx(Color(Color8(224, 180, 150), colors[0].a)), "vertex colour is the sRGB palette colour")
		check(roundi(uvs[0].x * 255.0) == VoxelVolume.MAT_SKIN and roundi(uvs[0].y * 255.0) == 0, "UV = (material, state) / 255")
	var empty := VoxelVolume.new()
	empty.create(Vector3i(CUBE, CUBE, CUBE), VOXEL_SIZE_M)
	var empty_mesh: ArrayMesh = VoxelMesher.build_mesh(empty, test_palette())
	check(empty_mesh != null and empty_mesh.get_surface_count() == 0, "empty volume gives a mesh without surfaces")
	var cut := VoxelMesher.tint_color(Color8(200, 100, 50), VoxelVolume.STATE_CUT)
	check(cut.is_equal_approx(Color(200.0 / 255.0 * 0.85, 100.0 / 255.0 * 0.85, 50.0 / 255.0 * 0.85)), "tint_color: CUT darkens by 0.85")


func test_model_roundtrip() -> void:
	begin("VoxelModel save + load roundtrip (user://)")
	var model := VoxelModel.new()
	model.model_name = "roundtrip"
	model.voxel_size = VOXEL_SIZE_M
	model.palette = test_palette()
	model.materials = {VoxelVolume.MAT_SKIN: "skin", VoxelVolume.MAT_BONE: "bone"}
	var vol := VoxelVolume.new()
	vol.create(Vector3i(4, 5, 6), VOXEL_SIZE_M)
	vol.part_name = "blob"
	vol.bone = "head"
	vol.origin = Vector3(-0.02, 0.0, 0.01)
	vol.set_voxel(Vector3i(1, 2, 3), VoxelVolume.MAT_SKIN, SKIN_COLOR_INDEX)
	vol.set_voxel(Vector3i(3, 4, 5), VoxelVolume.MAT_BONE, BONE_COLOR_INDEX)
	vol.set_state(Vector3i(1, 2, 3), VoxelVolume.STATE_WET | VoxelVolume.STATE_CUT)
	model.add_part(vol)
	check(model.get_part_count() == 1 and model.find_part("blob") == vol, "part added and found by name")
	check(model.save_to_file(ROUNDTRIP_PATH), "save_to_file")
	var loaded := VoxelModel.load_from_file(ROUNDTRIP_PATH)
	check(loaded != null, "load_from_file")
	if loaded == null:
		return
	check(loaded.model_name == "roundtrip", "model name survives")
	check(loaded.palette.size() == PALETTE_SIZE, "palette has 256 entries")
	check(loaded.palette[SKIN_COLOR_INDEX].is_equal_approx(Color8(224, 180, 150)), "palette colour survives")
	var part := loaded.find_part("blob")
	check(part != null, "part found after load")
	if part == null:
		return
	check(part.get_dims() == Vector3i(4, 5, 6), "part dims")
	check_approx(part.voxel_size, VOXEL_SIZE_M, "part voxel size")
	check(part.origin.is_equal_approx(Vector3(-0.02, 0.0, 0.01)), "part origin")
	check(part.bone == "head", "part bone")
	check(part.count_solid() == 2, "solid count")
	check(part.get_material(Vector3i(1, 2, 3)) == VoxelVolume.MAT_SKIN and part.get_color(Vector3i(1, 2, 3)) == SKIN_COLOR_INDEX, "material/colour of voxel (1,2,3)")
	check(part.get_state(Vector3i(1, 2, 3)) == VoxelVolume.STATE_WET | VoxelVolume.STATE_CUT, "state of voxel (1,2,3)")
	check(part.get_material(Vector3i(3, 4, 5)) == VoxelVolume.MAT_BONE, "material of voxel (3,4,5)")
	var buffer := model.save_to_buffer()
	var from_buffer := VoxelModel.new()
	check(buffer.size() > 8 and from_buffer.load_from_buffer(buffer) and from_buffer.get_part_count() == 1, "save_to_buffer / load_from_buffer")
	DirAccess.remove_absolute(ProjectSettings.globalize_path(ROUNDTRIP_PATH))


func test_part_in_tree() -> void:
	begin("VoxelPart in tree: cut -> rebuild -> mesh")
	var part := VoxelPart.new()
	part.name = "cube"
	var vol := filled_volume(CUBE, VoxelVolume.MAT_SKIN, SKIN_COLOR_INDEX)
	part.volume = vol
	part.palette = test_palette()
	_root.add_child(part)
	var side_m := CUBE * VOXEL_SIZE_M
	var removed := part.cut(Vector3(0.0, side_m * 0.5, side_m * 0.5), Vector3(side_m, side_m * 0.5, side_m * 0.5), 1.5 * VOXEL_SIZE_M)
	check(removed > 0, "cut removed %d voxels" % removed)
	check(part.get_solid_count() == CUBE * CUBE * CUBE - removed, "get_solid_count after cut")
	check(part.is_dirty(), "part is dirty after cut")
	part.rebuild()
	check(not part.is_dirty(), "rebuild clears dirty")
	var mesh_instance := part.get_mesh_instance()
	check(mesh_instance != null and mesh_instance.is_inside_tree(), "mesh instance exists in the tree")
	if mesh_instance != null:
		check(mesh_instance.mesh != null and mesh_instance.mesh.get_surface_count() == 1, "mesh has 1 surface")
		check(mesh_instance.material_override is ShaderMaterial, "part uses voxel_material.tres (ShaderMaterial)")
		var material := mesh_instance.material_override as ShaderMaterial
		if material != null:
			check_approx(material.get_shader_parameter("voxel_size"), VOXEL_SIZE_M, "shader voxel_size matches the part")
	var local_point := part.voxel_to_local(Vector3(1, 2, 3))
	check(part.local_to_voxel(local_point).is_equal_approx(Vector3(1, 2, 3)), "voxel_to_local / local_to_voxel roundtrip")
	# Row (y, z) = (1, 1) is untouched by the cut above; the ray starts 1 m before the cube.
	var ray_row_m := 1.5 * VOXEL_SIZE_M
	var hit: Dictionary = part.raycast_local(Vector3(-RAY_GAP_M, ray_row_m, ray_row_m), Vector3(1.0, 0.0, 0.0), 5.0)
	check(hit["hit"] and absf(float(hit["distance"]) - RAY_GAP_M) <= EPS and hit["normal"] == Vector3(-1.0, 0.0, 0.0) and hit["voxel"] == Vector3i(0, 1, 1), "raycast_local returns metres and a Vector3 normal")
	var bruised := part.bruise(Vector3(side_m * 0.5, side_m, side_m * 0.5), 2.0 * VOXEL_SIZE_M, 2)
	check(bruised > 0, "bruise changed %d voxels" % bruised)
	var pieces: Array[VoxelVolume] = part.detach_loose()
	check(pieces.is_empty(), "a connected cube has nothing to detach")
	_root.remove_child(part)
	part.free()


func test_head_model() -> void:
	begin("head_test.vxm: load, cut the forehead, detach")
	if not FileAccess.file_exists(HEAD_MODEL_PATH):
		check(true, "skipped: %s not built (python3 -m voxgen build-all tools/voxgen/specs)" % HEAD_MODEL_PATH)
		return
	var model := VoxelModel.load_from_file(HEAD_MODEL_PATH)
	check(model != null, "VoxelModel.load_from_file")
	if model == null:
		return
	check(model.find_part("head") != null, "part 'head' exists")
	var node := VoxelModelNode.new()
	node.name = "HeadUnderTest"
	_root.add_child(node)
	check(node.load_model(HEAD_MODEL_PATH), "VoxelModelNode.load_model")
	var part := node.get_part_node("head")
	check(part != null, "get_part_node('head')")
	if part == null:
		node.free()
		return
	var before := part.get_solid_count()
	check(before > MIN_HEAD_SOLID, "solid count %d > %d" % [before, MIN_HEAD_SOLID])
	var hit: Dictionary = part.raycast_local(Vector3(0.0, FOREHEAD_Y_M, RAY_START_Z_M), Vector3(0.0, 0.0, 1.0), RAY_LENGTH_M)
	check(hit["hit"], "ray from in front of the face hits the forehead")
	var removed := 0
	if hit["hit"]:
		var surface: Vector3 = hit["position"]
		var inward: Vector3 = -hit["normal"]
		var a := surface + Vector3(-FOREHEAD_HALF_WIDTH_M, 0.0, 0.0) + inward * FOREHEAD_CUT_DEPTH_M
		var b := surface + Vector3(FOREHEAD_HALF_WIDTH_M, 0.0, 0.0) + inward * FOREHEAD_CUT_DEPTH_M
		removed = part.cut(a, b, FOREHEAD_CUT_RADIUS_M)
	check(removed > 0, "cut across the forehead removed %d voxels" % removed)
	check(part.get_solid_count() == before - removed, "solid count dropped by the removed amount")
	var pieces: Array[VoxelVolume] = part.detach_loose()
	check(typeof(pieces) == TYPE_ARRAY, "detach_loose returns an Array (%d piece(s))" % pieces.size())
	part.rebuild()
	var mesh_instance := part.get_mesh_instance()
	check(mesh_instance != null and mesh_instance.mesh != null and mesh_instance.mesh.get_surface_count() == 1, "head mesh rebuilt with 1 surface")
	_root.remove_child(node)
	node.free()

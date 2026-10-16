class_name VoxelToolController
extends Node
## Mouse-driven barber tools that damage the VoxelModelNode under the cursor.
##
## Every frame while the left mouse button is held, a ray is cast from `camera`
## through the mouse into every VoxelPart of `model`; the nearest hit receives
## the current tool:
##   razor    - cut() between the previous and the current hit, both pushed
##              inward by a depth that grows with mouse speed; a sphere on the
##              first touch; loose pieces fall off as VoxelDebris,
##   club     - bruise(), level 1 on click, +1 per second held on the same spot,
##   candle   - burn(), level rising with the time held on the same spot,
##   scissors - trim_hair() along the stroke,
##   scoop    - carve_sphere_local() bites (spoon / pliers placeholder).
##
## Decisions where the task is silent:
##   * the razor only cuts after the mouse travelled `razor_min_stroke_px` since
##     its previous cut (a still razor would otherwise drill deeper every frame,
##     because the surface recedes under it),
##   * pieces smaller than `min_piece_voxels` (hair crumbs) vanish instead of
##     becoming rigid bodies,
##   * the scoop bites at most once per `scoop_interval_s`,
##   * "same spot" for club/candle means within `same_spot_tolerance_vox` of the
##     previous hit on the same part; moving further restarts the hold timer,
##   * op timing covers the voxel operation, detach_loose() and the rebuild.

signal tool_changed(tool: Tool)
signal op_applied(tool: Tool, part: VoxelPart, changed: int, usec: int)
signal piece_spawned(body: VoxelDebris)

enum Tool { RAZOR, CLUB, CANDLE, SCISSORS, SCOOP }

const TOOL_NAMES: Dictionary = {
	Tool.RAZOR: "бритва",
	Tool.CLUB: "дубина",
	Tool.CANDLE: "свеча",
	Tool.SCISSORS: "ножницы",
	Tool.SCOOP: "ложка",
}
const TOOL_KEYS: Dictionary = {
	Tool.RAZOR: "1",
	Tool.CLUB: "2",
	Tool.CANDLE: "3",
	Tool.SCISSORS: "4",
	Tool.SCOOP: "5",
}
const TOOL_ACTIONS: Dictionary = {
	"tool_razor": Tool.RAZOR,
	"tool_club": Tool.CLUB,
	"tool_candle": Tool.CANDLE,
	"tool_scissors": Tool.SCISSORS,
	"tool_scoop": Tool.SCOOP,
}
## Half a voxel: centres club/candle spheres on the surface voxel, not on its face.
const SURFACE_INSET_VOX := 0.5

@export var camera: Camera3D
@export var model: VoxelModelNode
## Parent for spawned debris bodies (world space); the controller's parent when empty.
@export var debris_parent: Node3D
@export_range(0.1, 50.0) var max_ray_distance_m := 5.0

@export_group("Razor")
@export var razor_radius_vox := 1.2
## Cut depth in metres per pixel/second of mouse speed (1000 px/s -> 2 voxels of 5 mm).
@export var razor_depth_per_px_speed_m := 1.0e-5
@export var razor_min_depth_vox := 0.5
@export var razor_max_depth_vox := 4.0
## Mouse travel (pixels) needed since the previous cut before the razor cuts again.
@export var razor_min_stroke_px := 2.0

@export_group("Club")
@export var club_radius_vox := 3.0
@export var club_levels_per_second := 1.0
@export_range(1, 3) var club_max_level := 3
@export var same_spot_tolerance_vox := 2.0

@export_group("Candle")
@export var candle_radius_vox := 2.5
## Seconds of holding after which burn level 1, 2 and 3 are reached.
@export var candle_level_times_s := PackedFloat32Array([0.3, 1.0, 2.0])

@export_group("Scissors")
@export var scissors_radius_vox := 2.0
@export var scissors_keep_length_vox := 1

@export_group("Scoop")
@export var scoop_radius_vox := 2.0
@export var scoop_bite_depth_vox := 1.0
@export var scoop_interval_s := 0.15

@export_group("Debris")
## Impulse (N*s) given to a fallen piece, directed away from the camera.
@export var piece_impulse_ns := 0.02
@export var piece_lifetime_s := 30.0
## Pieces with fewer solid voxels than this are dropped silently (dust, hair crumbs).
@export var min_piece_voxels := 2

var tool: Tool = Tool.RAZOR:
	set(value):
		if tool != value:
			tool = value
			_end_stroke()
			tool_changed.emit(tool)

## Hit of the previous frame (see _raycast_model) or empty when the stroke ended.
var _prev_hit: Dictionary = {}
var _hold_time_s := 0.0
var _scoop_cooldown_s := 0.0
## Mouse position at the razor's previous cut in the current stroke.
var _razor_last_mouse := Vector2.ZERO
var _prev_mouse := Vector2.ZERO
var _have_prev_mouse := false
var _mouse_speed_px_s := 0.0


func _unhandled_input(event: InputEvent) -> void:
	for action: String in TOOL_ACTIONS:
		if event.is_action_pressed(action):
			tool = TOOL_ACTIONS[action]
			get_viewport().set_input_as_handled()
			return


func _process(delta: float) -> void:
	var mouse := get_viewport().get_mouse_position()
	if _have_prev_mouse and delta > 0.0:
		_mouse_speed_px_s = (mouse - _prev_mouse).length() / delta
	_prev_mouse = mouse
	_have_prev_mouse = true
	_scoop_cooldown_s = maxf(0.0, _scoop_cooldown_s - delta)

	if camera == null or model == null or not Input.is_mouse_button_pressed(MOUSE_BUTTON_LEFT):
		_end_stroke()
		return
	_apply_tool(mouse, delta)


func get_tool_name() -> String:
	return TOOL_NAMES[tool]


func get_tool_key() -> String:
	return TOOL_KEYS[tool]


## Forget the current stroke (call after the model was reloaded).
func reset() -> void:
	_end_stroke()


func _end_stroke() -> void:
	_prev_hit = {}
	_hold_time_s = 0.0


## Nearest hit among the model's parts for the ray through `mouse`, or {}.
## Keys: part, distance, position, normal (global), local_position, local_normal, voxel, material.
func _raycast_model(mouse: Vector2) -> Dictionary:
	var origin := camera.project_ray_origin(mouse)
	var direction := camera.project_ray_normal(mouse)
	var best: Dictionary = {}
	for part: VoxelPart in model.get_part_nodes():
		var hit: Dictionary = part.raycast_global(origin, direction, max_ray_distance_m)
		if not hit.get("hit", false):
			continue
		var hit_distance: float = hit["distance"]
		if not best.is_empty() and hit_distance >= float(best["distance"]):
			continue
		var global_normal: Vector3 = hit["normal"]
		best = {
			"part": part,
			"distance": hit_distance,
			"position": hit["position"],
			"normal": global_normal,
			"local_position": part.to_local(hit["position"]),
			"local_normal": (part.global_transform.basis.inverse() * global_normal).normalized(),
			"voxel": hit["voxel"],
			"material": hit["material"],
		}
	return best


func _apply_tool(mouse: Vector2, delta: float) -> void:
	var hit := _raycast_model(mouse)
	if hit.is_empty():
		_end_stroke()
		return
	var part: VoxelPart = hit["part"]
	var vs: float = part.volume.voxel_size
	var same_part: bool = not _prev_hit.is_empty() and is_instance_valid(_prev_hit["part"]) and _prev_hit["part"] == part
	var moved_vox := INF
	if same_part:
		var prev_position: Vector3 = _prev_hit["local_position"]
		var current_position: Vector3 = hit["local_position"]
		moved_vox = prev_position.distance_to(current_position) / vs
	_hold_time_s = _hold_time_s + delta if moved_vox <= same_spot_tolerance_vox else 0.0

	var started_usec := Time.get_ticks_usec()
	var changed := 0
	var removes_voxels := false
	match tool:
		Tool.RAZOR:
			changed = _apply_razor(part, hit, vs, same_part, mouse)
			removes_voxels = true
		Tool.CLUB:
			changed = _apply_club(part, hit, vs)
		Tool.CANDLE:
			changed = _apply_candle(part, hit, vs)
			removes_voxels = true
		Tool.SCISSORS:
			changed = _apply_scissors(part, hit, vs, same_part)
			removes_voxels = true
		Tool.SCOOP:
			changed = _apply_scoop(part, hit, vs)
			removes_voxels = true
	if changed > 0:
		if removes_voxels:
			_drop_loose_pieces(part)
		part.rebuild()
		op_applied.emit(tool, part, changed, Time.get_ticks_usec() - started_usec)
	_prev_hit = hit


func _apply_razor(part: VoxelPart, hit: Dictionary, vs: float, same_part: bool, mouse: Vector2) -> int:
	if same_part and mouse.distance_to(_razor_last_mouse) < razor_min_stroke_px:
		return 0
	_razor_last_mouse = mouse
	var depth := clampf(_mouse_speed_px_s * razor_depth_per_px_speed_m, razor_min_depth_vox * vs, razor_max_depth_vox * vs)
	var radius := razor_radius_vox * vs
	var normal: Vector3 = hit["local_normal"]
	var position: Vector3 = hit["local_position"]
	var b := position - normal * depth
	if not same_part:
		return part.carve_sphere_local(b, radius)
	var prev_position: Vector3 = _prev_hit["local_position"]
	var prev_normal: Vector3 = _prev_hit["local_normal"]
	var a := prev_position - prev_normal * depth
	return part.cut(a, b, radius)


func _apply_club(part: VoxelPart, hit: Dictionary, vs: float) -> int:
	var level := mini(1 + int(floor(_hold_time_s * club_levels_per_second)), club_max_level)
	return part.bruise(_surface_centre(hit, vs), club_radius_vox * vs, level)


func _apply_candle(part: VoxelPart, hit: Dictionary, vs: float) -> int:
	var level := 0
	for threshold in candle_level_times_s:
		if _hold_time_s >= threshold:
			level += 1
	if level == 0:
		return 0
	return part.burn(_surface_centre(hit, vs), candle_radius_vox * vs, level)


func _apply_scissors(part: VoxelPart, hit: Dictionary, vs: float, same_part: bool) -> int:
	var b: Vector3 = hit["local_position"]
	var a := b
	if same_part:
		a = _prev_hit["local_position"]
	return part.trim_hair(a, b, scissors_radius_vox * vs, scissors_keep_length_vox)


func _apply_scoop(part: VoxelPart, hit: Dictionary, vs: float) -> int:
	if _scoop_cooldown_s > 0.0:
		return 0
	_scoop_cooldown_s = scoop_interval_s
	var normal: Vector3 = hit["local_normal"]
	var position: Vector3 = hit["local_position"]
	return part.carve_sphere_local(position - normal * scoop_bite_depth_vox * vs, scoop_radius_vox * vs)


## Centre for sphere operations: the hit point pushed half a voxel inward.
func _surface_centre(hit: Dictionary, vs: float) -> Vector3:
	var normal: Vector3 = hit["local_normal"]
	var position: Vector3 = hit["local_position"]
	return position - normal * SURFACE_INSET_VOX * vs


## Detach everything no longer connected to the part and turn it into debris.
func _drop_loose_pieces(part: VoxelPart) -> void:
	var pieces: Array[VoxelVolume] = part.detach_loose()
	if pieces.is_empty():
		return
	var parent: Node = debris_parent if debris_parent != null else get_parent()
	for piece: VoxelVolume in pieces:
		if piece.count_solid() < min_piece_voxels:
			continue
		var body := VoxelDebris.create(part, piece, piece_lifetime_s)
		parent.add_child(body)
		var away := (body.spawn_transform.origin - camera.global_position).normalized()
		body.place(away * piece_impulse_ns)
		piece_spawned.emit(body)

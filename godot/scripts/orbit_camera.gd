class_name OrbitCamera
extends Camera3D
## Orbit camera for the lab: right mouse drag orbits, the wheel zooms, middle
## mouse drag pans. The camera always looks at `target_node` + `target_offset`
## (the head centre, ~0.12 m above the model origin) plus whatever was panned.
##
## The camera sits on the -Z side of the target at yaw 0 so that a model facing
## -Z (the .vxm convention) shows its face by default.

## Node the camera orbits around; the scene origin when empty.
@export var target_node: Node3D
## Offset from the target node's origin to the point of interest (head centre).
@export var target_offset := Vector3(0.0, 0.12, 0.0)
## Distance from the point of interest to the camera, metres.
@export_range(0.05, 20.0) var distance := 0.75
@export_range(0.01, 1.0) var min_distance := 0.08
@export_range(0.5, 50.0) var max_distance := 6.0
## Orbit angles in degrees; pitch > 0 looks down from above.
@export var yaw_deg := 25.0
@export var pitch_deg := 12.0
@export_range(-89.0, 89.0) var min_pitch_deg := -80.0
@export_range(-89.0, 89.0) var max_pitch_deg := 85.0
## Degrees of orbit per pixel of mouse travel.
@export_range(0.01, 2.0) var orbit_deg_per_px := 0.3
## Pan distance per pixel as a fraction of the current camera distance.
@export_range(0.0001, 0.02) var pan_fraction_per_px := 0.0015
## Multiplicative zoom step per wheel notch.
@export_range(1.01, 2.0) var zoom_step := 1.12

## Accumulated pan offset from middle-mouse drags (world space).
var _pan_offset := Vector3.ZERO


func _ready() -> void:
	_update_transform()


func _process(_delta: float) -> void:
	# Follow the target every frame; it is cheap and keeps the camera glued to a
	# target that moves (a head on a ragdoll later).
	_update_transform()


func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventMouseMotion:
		var motion := event as InputEventMouseMotion
		if motion.button_mask & MOUSE_BUTTON_MASK_RIGHT:
			_orbit(motion.relative)
		elif motion.button_mask & MOUSE_BUTTON_MASK_MIDDLE:
			_pan(motion.relative)
	elif event is InputEventMouseButton:
		var button := event as InputEventMouseButton
		if not button.pressed:
			return
		if button.button_index == MOUSE_BUTTON_WHEEL_UP:
			_zoom(1.0 / zoom_step)
		elif button.button_index == MOUSE_BUTTON_WHEEL_DOWN:
			_zoom(zoom_step)


## Point in world space the camera looks at.
func get_focus_point() -> Vector3:
	var base := Vector3.ZERO
	if target_node != null:
		base = target_node.global_position
	return base + target_offset + _pan_offset


## Forget panning and return to the default framing of the target.
func reset_view() -> void:
	_pan_offset = Vector3.ZERO
	_update_transform()


func _orbit(relative: Vector2) -> void:
	yaw_deg = wrapf(yaw_deg - relative.x * orbit_deg_per_px, -180.0, 180.0)
	pitch_deg = clampf(pitch_deg + relative.y * orbit_deg_per_px, min_pitch_deg, max_pitch_deg)
	_update_transform()


func _pan(relative: Vector2) -> void:
	var step := pan_fraction_per_px * distance
	# Dragging right moves the scene right, i.e. the focus point moves left.
	_pan_offset -= global_transform.basis.x * relative.x * step
	_pan_offset += global_transform.basis.y * relative.y * step
	_update_transform()


func _zoom(factor: float) -> void:
	distance = clampf(distance * factor, min_distance, max_distance)
	_update_transform()


func _update_transform() -> void:
	var focus := get_focus_point()
	var rotation_basis := Basis.from_euler(Vector3(deg_to_rad(pitch_deg), deg_to_rad(yaw_deg), 0.0))
	# Start on the -Z side of the focus (in front of a face that looks at -Z).
	var offset := rotation_basis * Vector3(0.0, 0.0, -distance)
	global_position = focus + offset
	look_at(focus, Vector3.UP)

class_name VoxelLab
extends Node3D
## Lab scene controller: HUD, model reload, screenshots and the wireframe toggle.
##
## scenes/lab.tscn holds the lights, the environment, the orbit camera, the
## "Head" VoxelModelNode and the tool controller; this script glues them
## together and reports what happens in the HUD.

## Model the lab shows; the "Head" node carries the same path in the scene.
const HEAD_MODEL_PATH := "res://assets/models/head_test.vxm"
## Optional prop beside the head, skipped silently when voxgen has not built it.
const STOOL_MODEL_PATH := "res://assets/models/stool.vxm"
const SCREENSHOT_DIR := "user://screenshots"
const MISSING_MODEL_HINT := "Нет модели: запусти  python3 -m voxgen build-all tools/voxgen/specs"
const HUD_REFRESH_INTERVAL_S := 0.1
const USEC_PER_MSEC := 1000.0
const MSEC_PER_SEC := 1000
const NO_OP_TEXT := "—"

@onready var head: VoxelModelNode = $Head
@onready var stool: VoxelModelNode = $Stool
@onready var tools: VoxelToolController = $Tools
@onready var debris: Node3D = $Debris
@onready var info_label: Label = $HUD/InfoLabel
@onready var missing_label: Label = $HUD/MissingLabel

var _solid_total := 0
var _last_op_text := NO_OP_TEXT
var _hud_timer := 0.0
var _wireframe := false


func _init() -> void:
	# Wireframe index buffers exist only for meshes created after this call, so
	# it must run before the model loads (children enter the tree after their
	# parent is constructed).
	RenderingServer.set_debug_generate_wireframes(true)


func _ready() -> void:
	missing_label.text = MISSING_MODEL_HINT
	_load_optional_model(stool, STOOL_MODEL_PATH)
	tools.op_applied.connect(_on_op_applied)
	tools.tool_changed.connect(_on_tool_changed)
	_refresh_model_state()


func _process(delta: float) -> void:
	_hud_timer += delta
	if _hud_timer >= HUD_REFRESH_INTERVAL_S:
		_hud_timer = 0.0
		_refresh_hud()


func _unhandled_input(event: InputEvent) -> void:
	if event.is_action_pressed("reload_model"):
		reload_model()
	elif event.is_action_pressed("screenshot"):
		take_screenshot()
	elif event.is_action_pressed("toggle_wireframe"):
		toggle_wireframe()


## Re-read the head from disk (a fresh, undamaged head) and clear the debris.
func reload_model() -> void:
	tools.reset()
	for child: Node in debris.get_children():
		child.queue_free()
	var path := head.model_path if not head.model_path.is_empty() else HEAD_MODEL_PATH
	if FileAccess.file_exists(path):
		head.load_model(path)
	_last_op_text = NO_OP_TEXT
	_refresh_model_state()


## Save the next fully drawn frame to user://screenshots/<timestamp>.png.
func take_screenshot() -> void:
	RenderingServer.frame_post_draw.connect(_save_screenshot, CONNECT_ONE_SHOT)


func toggle_wireframe() -> void:
	_wireframe = not _wireframe
	get_viewport().debug_draw = Viewport.DEBUG_DRAW_WIREFRAME if _wireframe else Viewport.DEBUG_DRAW_DISABLED


func _save_screenshot() -> void:
	var image: Image = get_viewport().get_texture().get_image()
	var dir_error := DirAccess.make_dir_recursive_absolute(ProjectSettings.globalize_path(SCREENSHOT_DIR))
	if dir_error != OK:
		push_error("Screenshot: cannot create %s (%s)" % [SCREENSHOT_DIR, error_string(dir_error)])
		return
	var stamp := Time.get_datetime_string_from_system(false, false).replace(":", "-")
	var path := "%s/%s_%03d.png" % [SCREENSHOT_DIR, stamp, Time.get_ticks_msec() % MSEC_PER_SEC]
	var error := image.save_png(path)
	if error != OK:
		push_error("Screenshot: cannot save %s (%s)" % [path, error_string(error)])
		return
	print("Screenshot saved: %s" % ProjectSettings.globalize_path(path))


func _load_optional_model(node: VoxelModelNode, path: String) -> void:
	if FileAccess.file_exists(path):
		node.load_model(path)


func _refresh_model_state() -> void:
	missing_label.visible = head.model == null
	_solid_total = _count_solid(head)
	_refresh_hud()


func _count_solid(model_node: VoxelModelNode) -> int:
	var total := 0
	for part: VoxelPart in model_node.get_part_nodes():
		total += part.get_solid_count()
	return total


func _refresh_hud() -> void:
	var model_name := NO_OP_TEXT
	if head.model != null:
		model_name = head.model.model_name
	var lines := PackedStringArray([
		"Инструмент: %s  [%s]" % [tools.get_tool_name(), tools.get_tool_key()],
		"Модель: %s" % model_name,
		"Вокселей: %d" % _solid_total,
		"Операция: %s" % _last_op_text,
		"FPS: %d" % Engine.get_frames_per_second(),
		"",
		"1-5 инструмент · ЛКМ применить · ПКМ орбита · колесо зум · СКМ панорама",
		"R новая голова · F каркас · F12 скриншот",
	])
	info_label.text = "\n".join(lines)


func _on_op_applied(_tool: VoxelToolController.Tool, part: VoxelPart, changed: int, usec: int) -> void:
	_last_op_text = "%s → %s: %d вокс., %.2f мс" % [tools.get_tool_name(), part.name, changed, usec / USEC_PER_MSEC]
	_solid_total = _count_solid(head)
	_refresh_hud()


func _on_tool_changed(_tool: VoxelToolController.Tool) -> void:
	_refresh_hud()

extends SceneTree
## Renders scenes/lab.tscn for a few frames and saves the viewport to a PNG.
##
##   xvfb-run -a godot --path godot --rendering-driver opengl3 --resolution 1280x720 \
##       -s res://tests/screenshot.gd -- build/lab.png
##
## The output path is the first user argument (after "--"), default user://lab.png.
## A relative path is resolved against the directory the command was started in
## (the PWD variable; Godot itself chdirs into the project), or, without PWD,
## against the parent of the project folder (the repository root).

const SCENE_PATH := "res://scenes/lab.tscn"
const DEFAULT_OUTPUT := "user://lab.png"
const FRAMES_TO_WAIT := 10
const EXIT_OK := 0
const EXIT_FAIL := 1

var _output := DEFAULT_OUTPUT
var _frames := 0
var _capturing := false


func _initialize() -> void:
	_output = _resolve_output_path()
	var packed: PackedScene = load(SCENE_PATH)
	if packed == null:
		printerr("screenshot: cannot load %s" % SCENE_PATH)
		quit(EXIT_FAIL)
		return
	get_root().add_child(packed.instantiate())
	process_frame.connect(_on_frame)


func _on_frame() -> void:
	_frames += 1
	if _capturing or _frames < FRAMES_TO_WAIT:
		return
	_capturing = true
	RenderingServer.frame_post_draw.connect(_capture, CONNECT_ONE_SHOT)


func _capture() -> void:
	var image: Image = get_root().get_viewport().get_texture().get_image()
	if image == null or image.is_empty():
		printerr("screenshot: the viewport returned no image")
		quit(EXIT_FAIL)
		return
	var absolute := ProjectSettings.globalize_path(_output)
	var dir_error := DirAccess.make_dir_recursive_absolute(absolute.get_base_dir())
	if dir_error != OK:
		printerr("screenshot: cannot create %s (%s)" % [absolute.get_base_dir(), error_string(dir_error)])
		quit(EXIT_FAIL)
		return
	var error := image.save_png(_output)
	if error != OK:
		printerr("screenshot: cannot save %s (%s)" % [absolute, error_string(error)])
		quit(EXIT_FAIL)
		return
	print("screenshot: %s (%dx%d)" % [absolute, image.get_width(), image.get_height()])
	quit(EXIT_OK)


func _resolve_output_path() -> String:
	var args := OS.get_cmdline_user_args()
	if args.is_empty():
		return DEFAULT_OUTPUT
	var path := args[0]
	if path.begins_with("res://") or path.begins_with("user://") or path.is_absolute_path():
		return path
	var base := OS.get_environment("PWD")
	if base.is_empty():
		base = ProjectSettings.globalize_path("res://").rstrip("/").get_base_dir()
	return base.path_join(path)

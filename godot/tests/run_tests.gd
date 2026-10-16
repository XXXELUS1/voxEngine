extends SceneTree
## Headless test runner for the voxbody extension and the lab project.
##
##   godot --headless --path godot -s res://tests/run_tests.gd
##
## The cases live in res://tests/voxbody_tests.gd and are loaded at runtime so
## that a missing voxbody build gives one clear message instead of a parse
## error. Every check prints one line; the summary sets the exit code
## (1 on any failure, 0 otherwise).

const CASES_SCRIPT_PATH := "res://tests/voxbody_tests.gd"
const EXTENSION_PROBE_CLASS := "VoxelVolume"
const EXIT_OK := 0
const EXIT_FAIL := 1

var _cases_object: Object
var _passed := 0
var _failed := 0
var _cases := 0


func _initialize() -> void:
	print("voxEngine tests — Godot %s" % Engine.get_version_info()["string"])
	if not ClassDB.class_exists(EXTENSION_PROBE_CLASS):
		printerr("FAIL: the voxbody GDExtension is not loaded. Build it first: cd src/voxbody && scons target=template_debug")
		quit(EXIT_FAIL)
		return
	var script: GDScript = load(CASES_SCRIPT_PATH)
	if script == null or not script.can_instantiate():
		printerr("FAIL: cannot load %s" % CASES_SCRIPT_PATH)
		quit(EXIT_FAIL)
		return
	_cases_object = script.new()
	# The root window enters the tree only after _initialize() returns, so run
	# on the first frame: nodes the cases add then get NOTIFICATION_READY as in
	# a real scene.
	process_frame.connect(_run_cases, CONNECT_ONE_SHOT)


func _run_cases() -> void:
	_cases_object.call("run", check, begin_case, root)
	_summary()
	quit(EXIT_FAIL if _failed > 0 else EXIT_OK)


## Start a named case; the checks that follow are listed under it.
func begin_case(case_name: String) -> void:
	_cases += 1
	print("[%d] %s" % [_cases, case_name])


## Record one check and print one line for it.
func check(condition: bool, check_name: String) -> bool:
	if condition:
		_passed += 1
		print("    ok    %s" % check_name)
	else:
		_failed += 1
		print("    FAIL  %s" % check_name)
	return condition


func _summary() -> void:
	print("")
	print("%d case(s), %d check(s): %d passed, %d failed — %s" % [
		_cases, _passed + _failed, _passed, _failed, "FAILED" if _failed > 0 else "OK"])

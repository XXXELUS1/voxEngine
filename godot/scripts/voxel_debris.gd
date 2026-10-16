class_name VoxelDebris
extends RigidBody3D
## A piece that fell off a VoxelPart ("отвалившееся становится предметом"):
## a RigidBody3D with a VoxelPart child showing the detached volume and a
## SphereShape3D sized from the piece bounds.
##
## Decisions where the task is silent:
##   * the body origin is the centre of the piece's bounding box (not the source
##     part's origin) so the sphere shape and the centre of mass sit on the piece;
##     the VoxelPart child is offset by the negative of that centre, so the mesh
##     appears exactly where the voxels were before they fell off,
##   * mass = solid voxels * voxel volume * DENSITY_KG_M3 (soft tissue ~ water),
##   * pieces live on physics layer "debris" (2) and collide with "world" (1) and
##     other debris; they free themselves after `lifetime_s` seconds.

const DENSITY_KG_M3 := 1000.0
const MIN_MASS_KG := 0.005
const MIN_RADIUS_VOX := 0.5
## Physics layer bits, see [layer_names] in project.godot.
const LAYER_WORLD := 1
const LAYER_DEBRIS := 2

## Seconds before the piece removes itself; 0 or less keeps it forever.
@export var lifetime_s := 30.0

## Counter for unique node names (siblings with equal names get auto-renamed otherwise).
static var _spawned_count := 0

## The visual part; its volume is the detached piece.
var part: VoxelPart
## World transform the piece should get once it is in the tree.
var spawn_transform := Transform3D.IDENTITY


## Build a debris body for `piece`, a volume returned by VoxelPart.detach_loose()
## on `source`. The caller adds the body to the tree and then calls place().
static func create(source: VoxelPart, piece: VoxelVolume, lifetime: float) -> VoxelDebris:
	var body := VoxelDebris.new()
	_spawned_count += 1
	var stem := piece.part_name if not piece.part_name.is_empty() else "piece"
	body.name = "%s_%d" % [stem, _spawned_count]
	body.lifetime_s = lifetime

	var vs := piece.voxel_size
	var lo := piece.get_solid_min()
	var hi := piece.get_solid_max()
	if hi.x < lo.x:
		# Empty piece (should not happen): fall back to the whole volume box.
		lo = Vector3i.ZERO
		hi = piece.get_dims() - Vector3i.ONE
	var size_vox := Vector3(hi - lo + Vector3i.ONE)
	var centre_local := piece.origin + (Vector3(lo) + size_vox * 0.5) * vs
	var half_extent_vox := maxf(maxf(size_vox.x, size_vox.y), size_vox.z) * 0.5
	var radius := maxf(half_extent_vox, MIN_RADIUS_VOX) * vs

	body.mass = maxf(MIN_MASS_KG, float(piece.count_solid()) * vs * vs * vs * DENSITY_KG_M3)
	body.collision_layer = 1 << (LAYER_DEBRIS - 1)
	body.collision_mask = (1 << (LAYER_WORLD - 1)) | (1 << (LAYER_DEBRIS - 1))
	body.continuous_cd = true

	var shape := CollisionShape3D.new()
	var sphere := SphereShape3D.new()
	sphere.radius = radius
	shape.shape = sphere
	body.add_child(shape)

	body.part = VoxelPart.new()
	body.part.name = "VoxelPart"
	body.part.volume = piece
	body.part.palette = source.palette
	body.part.material_override = source.material_override
	body.part.position = -centre_local
	body.add_child(body.part)

	body.spawn_transform = Transform3D(source.global_transform.basis, source.to_global(centre_local))
	return body


## Put the body where the piece was and push it; call after add_child().
func place(impulse: Vector3) -> void:
	global_transform = spawn_transform
	part.rebuild()
	apply_central_impulse(impulse)


func _ready() -> void:
	if lifetime_s > 0.0:
		get_tree().create_timer(lifetime_s).timeout.connect(queue_free)

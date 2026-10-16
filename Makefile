# voxEngine developer shortcuts (Linux / macOS / WSL; on Windows use the commands from README.md).
#
#   make ext          debug build of the voxbody GDExtension (what the Godot editor loads)
#   make ext-release  release build of the extension
#   make assets       regenerate every model from tools/voxgen/specs (+ previews, .vox, reports)
#   make test         pytest for voxgen, then the headless Godot suite (if the extension is built)
#   make screenshot   off-screen render of the lab scene -> build/lab.png (Xvfb + llvmpipe)
#   make clean        remove build/, godot/.godot/, built libraries, generated models, caches
#   make distclean    clean + wipe the godot-cpp build (the next `make ext` takes ~10 min again)
#
# Everything is a thin wrapper over scripts/*.sh; see the usage comment at the top of each script.

.PHONY: help ext ext-release assets test screenshot clean distclean

help:
	@sed -n '3,9p' $(MAKEFILE_LIST) | sed 's/^#   //'

ext:
	./scripts/build_ext.sh debug

ext-release:
	./scripts/build_ext.sh release

assets:
	./scripts/build_assets.sh

test:
	./scripts/test_all.sh

screenshot:
	./scripts/screenshot.sh build/lab.png

clean:
	rm -rf build godot/.godot
	find godot/addons/voxbody/bin -mindepth 1 ! -name .gitkeep -exec rm -rf {} +
	rm -f godot/assets/models/*.vxm godot/assets/models/*.vox
	rm -f src/voxbody/src/*.o src/voxbody/src/*.os src/voxbody/src/*.obj
	find tools -type d \( -name __pycache__ -o -name .pytest_cache -o -name '*.egg-info' \) -prune -exec rm -rf {} +

distclean: clean
	rm -rf src/voxbody/godot-cpp/bin src/voxbody/godot-cpp/gen
	find src/voxbody/godot-cpp/src -name '*.o' -delete 2>/dev/null || true
	rm -f src/voxbody/.sconsign.dblite

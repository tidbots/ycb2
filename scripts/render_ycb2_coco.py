import blenderproc as bproc  # MUST be the first import for BlenderProc (no shebang/docstring above!)

# render_ycb2_coco.py (Py3.8 + Blender 2.93 compatible)
#
# YCB (OBJ+MTL+PNG) -> synthetic RGB + COCO (bbox/seg) generator with BlenderProc.
# + Optional CC Textures (PBR) background randomization for "more realistic" floor/wall.
#
# Notes:
# - Do NOT put shebang or a module docstring before the blenderproc import.
# - Python 3.8 compatible: avoid `X | None` typing.
#
# Example:
# blenderproc run /work/scripts/render_ycb2_coco.py \
#   --assets_root /work/models/ycb \
#   --out_dir /work/data/synth_coco \
#   --n_images 400 --width 640 --height 480 \
#   --force_texture \
#   --cc_textures_dir /work/resources/cctextures \
#   --cc_used_assets "Asphalt020"

import argparse
import os
import random
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np

# bpy is available inside BlenderProc's Blender Python
import bpy


YCB_DEFAULT_OBJECTS: List[Tuple[str, int]] = [
    ("002_master_chef_can", 1),
    ("003_cracker_box", 2),
]


def find_textured_obj(assets_root: Path, obj_name: str) -> Tuple[Path, Optional[Path]]:
    """Find a YCB textured.obj and (optionally) textured.png near it."""
    candidates = [
        assets_root / obj_name / obj_name / "tsdf" / "textured.obj",
        assets_root / obj_name / obj_name / "poisson" / "textured.obj",
    ]
    for p in candidates:
        if p.exists():
            png = p.with_name("textured.png")
            return p, (png if png.exists() else None)
    raise FileNotFoundError("Could not find textured.obj for {} under {}".format(obj_name, assets_root))


def force_apply_png_texture(bp_obj, png_path: Path) -> None:
    """Force-assign textured.png via nodes (robust when MTL relative paths fail)."""
    bo = bp_obj.blender_obj
    img = bpy.data.images.load(str(png_path), check_existing=True)

    mat = bpy.data.materials.new(name="ycb_tex_{}".format(bo.name))
    mat.use_nodes = True
    nt = mat.node_tree
    nodes = nt.nodes
    links = nt.links

    for n in list(nodes):
        nodes.remove(n)

    out = nodes.new(type="ShaderNodeOutputMaterial")
    out.location = (400, 0)

    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf.location = (150, 0)

    tex = nodes.new(type="ShaderNodeTexImage")
    tex.location = (-200, 0)
    tex.image = img

    links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    if hasattr(bo.data, "materials"):
        if bo.data.materials:
            bo.data.materials[0] = mat
        else:
            bo.data.materials.append(mat)


def setup_cycles_gpu(prefer_optix: bool) -> None:
    """Try to enable Cycles GPU rendering. If not possible, Blender may fall back to CPU."""
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "GPU"

    try:
        prefs = bpy.context.preferences
        cycles_prefs = prefs.addons["cycles"].preferences

        try_order = ["OPTIX", "CUDA"] if prefer_optix else ["CUDA", "OPTIX"]
        for backend in try_order:
            try:
                cycles_prefs.compute_device_type = backend
                cycles_prefs.get_devices()
                for d in cycles_prefs.devices:
                    d.use = True
                print("[GPU] Cycles backend:", backend)
                return
            except Exception as e:
                print("[GPU] failed backend {}: {}".format(backend, e))
                continue

        print("[GPU] Could not enable OPTIX/CUDA explicitly. Blender may use CPU.")
    except Exception as e:
        print("[GPU] Cycles GPU setup exception:", e)


def make_plane(name: str, size: float, location: List[float], rotation_euler: List[float]):
    plane = bproc.object.create_primitive("PLANE", scale=[size, size, 1.0])
    plane.set_name(name)
    plane.set_location(location)
    plane.set_rotation_euler(rotation_euler)
    return plane


def set_plane_flat_color(plane, rgb: Tuple[float, float, float]) -> None:
    mat = bproc.material.create(name="bg_flat_mat")
    mat.set_principled_shader_value("Base Color", [rgb[0], rgb[1], rgb[2], 1.0])
    mat.set_principled_shader_value("Roughness", 0.9)
    plane.replace_materials(mat)


def make_point_light():
    light = bproc.types.Light()
    light.set_type("POINT")
    light.set_energy(200.0)
    light.set_location([1.0, -1.0, 2.0])
    return light


def sample_cam_pose_look_at(poi: np.ndarray, r_min: float, r_max: float, z_min: float, z_max: float) -> np.ndarray:
    theta = np.random.uniform(0, 2 * np.pi)
    r = np.random.uniform(r_min, r_max)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.random.uniform(z_min, z_max)
    location = np.array([x, y, z])

    forward = poi - location
    rot = bproc.camera.rotation_from_forward_vec(forward, up_axis="Y")
    cam2world = bproc.math.build_transformation_mat(location, rot)
    return cam2world


def parse_csv_names(s: Optional[str]) -> Optional[List[str]]:
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    return [x.strip() for x in s.split(",") if x.strip()]


def load_cc_materials(cc_dir: Path, allow_names: Optional[List[str]], debug: bool):
    """
    Load CC Textures materials using BlenderProc helper.
    It expects cc_dir to contain many subfolders (e.g., Asphalt020) with PBR texture files inside.
    """
    if not cc_dir.exists():
        raise FileNotFoundError("cc_textures_dir does not exist: {}".format(cc_dir))

    # Quick sanity: list subfolders
    subdirs = sorted([p for p in cc_dir.iterdir() if p.is_dir()])
    if debug:
        print("[CC] cc_dir:", cc_dir)
        print("[CC] subdirs:", [p.name for p in subdirs[:30]], ("..." if len(subdirs) > 30 else ""))

    if allow_names is not None:
        wanted = set(allow_names)
        subdirs = [p for p in subdirs if p.name in wanted]
        if debug:
            print("[CC] filtered subdirs:", [p.name for p in subdirs])

    # If folders exist but empty, BlenderProc loader will yield nothing → catch early
    any_files = False
    for d in subdirs:
        if any(d.glob("*")):
            any_files = True
            break
    if not any_files:
        raise RuntimeError(
            "cc_textures_dir has no files under selected folders. "
            "Example selected folder is empty: {}. "
            "Did you run: blenderproc download cc_textures {} ?".format(
                (subdirs[0].as_posix() if subdirs else cc_dir.as_posix()),
                cc_dir.as_posix(),
            )
        )

    # BlenderProc loader: loads all materials it can find under cc_dir
    # (If you only downloaded a few folders, it's fast.)
    mats = bproc.loader.load_ccmaterials(str(cc_dir))
    if allow_names is not None:
        # keep only those whose material name contains folder name (robust-ish)
        filtered = []
        for m in mats:
            mname = getattr(m, "get_name", None)
            nm = m.get_name() if callable(mname) else str(m)
            if any(a in nm for a in allow_names):
                filtered.append(m)
        mats = filtered

    if len(mats) == 0:
        raise RuntimeError("No CC materials loaded from {} (after filtering).".format(cc_dir))

    if debug:
        try:
            names = [m.get_name() for m in mats[:20]]
            print("[CC] loaded materials (sample):", names, ("..." if len(mats) > 20 else ""))
        except Exception:
            print("[CC] loaded materials count:", len(mats))

    return mats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets_root", required=True, help="Path to models/ycb (download root).")
    ap.add_argument("--out_dir", required=True, help="Output directory.")
    ap.add_argument("--n_images", type=int, default=400)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prefer_optix", action="store_true", help="Prefer OPTIX over CUDA.")
    ap.add_argument("--force_texture", action="store_true", help="Force-apply textured.png via nodes.")
    ap.add_argument("--only_single_object", action="store_true", help="Render only one object per frame.")
    ap.add_argument("--radius_min", type=float, default=0.6)
    ap.add_argument("--radius_max", type=float, default=1.0)
    ap.add_argument("--debug_print_paths", action="store_true")

    # CC Textures (PBR) backgrounds
    ap.add_argument("--use_cc_textures", action="store_true", help="(compat) If set, enable CC textures (requires --cc_textures_dir).")
    ap.add_argument("--cc_textures_dir", type=str, default=None, help="Path to cc_textures root dir.")
    ap.add_argument("--cc_used_assets", type=str, default=None, help='Comma list of folder names to prefer, e.g. "Asphalt020,WoodFloor041".')

    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    assets_root = Path(args.assets_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    coco_dir = out_dir / "coco_data"
    coco_dir.mkdir(parents=True, exist_ok=True)

    bproc.init()
    bproc.camera.set_resolution(args.width, args.height)

    setup_cycles_gpu(prefer_optix=args.prefer_optix)

    # Stage: floor + (optional) wall
    floor = make_plane("floor", size=3.0, location=[0, 0, 0], rotation_euler=[0, 0, 0])
    wall = make_plane("wall", size=3.0, location=[0, 1.3, 1.0], rotation_euler=[np.pi / 2.0, 0, 0])

    light = make_point_light()

    # Decide background material strategy
    cc_mats = None
    want_cc = args.use_cc_textures or (args.cc_textures_dir is not None)
    cc_allow = parse_csv_names(args.cc_used_assets)

    if want_cc:
        if args.cc_textures_dir is None:
            raise RuntimeError("--use_cc_textures was set but --cc_textures_dir is missing.")
        cc_dir = Path(args.cc_textures_dir).resolve()
        cc_mats = load_cc_materials(cc_dir, cc_allow, debug=args.debug_print_paths)
    else:
        # fallback flat color
        set_plane_flat_color(floor, (0.9, 0.9, 0.85))
        set_plane_flat_color(wall, (0.92, 0.92, 0.95))

    # Load objects
    loaded = []  # list of (bp_obj, name, cat_id)
    for name, cat_id in YCB_DEFAULT_OBJECTS:
        obj_path, png_path = find_textured_obj(assets_root, name)
        if args.debug_print_paths:
            print("[YCB] {}: obj={} png={}".format(name, obj_path, png_path))

        # Key trick: chdir to OBJ folder so mtllib/map_Kd relative paths resolve
        prev_cwd = os.getcwd()
        os.chdir(str(obj_path.parent))
        try:
            bp_objs = bproc.loader.load_obj(str(obj_path), use_image_search=True)
        finally:
            os.chdir(prev_cwd)

        for bp_obj in bp_objs:
            bp_obj.set_cp("category_id", int(cat_id))
            bp_obj.set_name(name)

            if args.force_texture and png_path is not None:
                try:
                    force_apply_png_texture(bp_obj, png_path)
                except Exception as e:
                    print("[WARN] force texture failed for {}: {}".format(name, e))

            loaded.append((bp_obj, name, cat_id))

    hide_loc = np.array([1000.0, 1000.0, 1000.0])
    poi = np.array([0.0, 0.0, 0.08])

    # Per-frame randomization + camera poses
    for i in range(args.n_images):
        # Background materials
        if cc_mats is not None:
            mat = random.choice(cc_mats)
            floor.replace_materials(mat)
            # wall: either same or another random (same gives coherent scene)
            if random.random() < 0.7:
                wall.replace_materials(mat)
            else:
                wall.replace_materials(random.choice(cc_mats))
        else:
            # tiny jitter in flat colors
            c1 = float(np.random.uniform(0.85, 0.95))
            c2 = float(np.random.uniform(0.85, 0.95))
            set_plane_flat_color(floor, (c1, c1, c2))
            set_plane_flat_color(wall, (c2, c2, c1))

        # Lighting randomization (strength + color)
        light.set_location(
            [
                float(np.random.uniform(-1.2, 1.2)),
                float(np.random.uniform(-1.2, 1.2)),
                float(np.random.uniform(1.2, 2.5)),
            ],
            frame=i,
        )
        light.set_energy(float(np.random.uniform(80.0, 320.0)), frame=i)

        # Choose 1 or 2 objects
        if args.only_single_object:
            chosen = [random.choice(loaded)]
        else:
            k = 1 if random.random() < 0.5 else 2
            chosen = random.sample(loaded, k=min(k, len(loaded)))

        # Hide all
        for bp_obj, _, _ in loaded:
            bp_obj.set_location(hide_loc.tolist(), frame=i)

        # Place chosen with simple separation
        positions: List[np.ndarray] = []
        for bp_obj, _, _ in chosen:
            pos = None
            for _try in range(50):
                x = np.random.uniform(-0.25, 0.25)
                y = np.random.uniform(-0.25, 0.25)
                z = np.random.uniform(0.01, 0.03)
                p = np.array([x, y, z])
                if all(np.linalg.norm(p - q) > 0.18 for q in positions):
                    pos = p
                    break
            if pos is None:
                pos = np.array([0.0, 0.0, 0.02])
            positions.append(pos)

            rot = np.array([0.0, 0.0, np.random.uniform(-np.pi, np.pi)])
            bp_obj.set_location(pos.tolist(), frame=i)
            bp_obj.set_rotation_euler(rot.tolist(), frame=i)

        cam2world = sample_cam_pose_look_at(
            poi=poi,
            r_min=args.radius_min,
            r_max=args.radius_max,
            z_min=0.25,
            z_max=0.7,
        )
        bproc.camera.add_camera_pose(cam2world, frame=i)

    # Render
    data = bproc.renderer.render()
    # --- after: data = bproc.renderer.render() ---
    import imageio.v2 as imageio

    img_dir = os.path.join(args.out_dir, "coco_data", "images")
    os.makedirs(img_dir, exist_ok=True)

    colors = data.get("colors", None)
    if colors is None or len(colors) == 0:
        raise RuntimeError("No RGB images in data['colors']. Rendering produced no color buffers.")

    # colors: list of HxWx3 float32 in [0,1] or uint8
    for i, img in enumerate(colors, start=1):
        out_path = os.path.join(img_dir, f"rgb_{i:04d}.jpg")
        if img.dtype != np.uint8:
            img8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        else:
            img8 = img
        imageio.imwrite(out_path, img8, quality=95)

# make COCO writer use our saved filenames (relative to coco root)
# (We will pass image paths explicitly when writing.)

    seg = bproc.renderer.render_segmap(map_by=["instance", "class", "name"])

    # Write COCO
    bproc.writer.write_coco_annotations(
        str(coco_dir),
        instance_segmaps=seg["instance_segmaps"],
        instance_attribute_maps=seg["instance_attribute_maps"],
        colors=data["colors"],
        color_file_format="JPEG",
    )

    print("[OK] Wrote COCO dataset to:", coco_dir)
    print("     - coco_annotations.json")
    print("     - images (JPEG) + masks/segmaps")


if __name__ == "__main__":
    main()

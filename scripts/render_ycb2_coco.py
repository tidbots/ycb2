import blenderproc as bproc  # MUST be the first import for BlenderProc

"""
render_ycb2_coco.py (Py3.8 compatible)

YCB (OBJ+MTL+PNG) -> synthetic RGB + COCO (bbox/seg) generator with BlenderProc.

- Smoke test objects:
  - 002_master_chef_can (category_id=1)
  - 003_cracker_box     (category_id=2)
- Prefers tsdf/textured.obj. Falls back to poisson/textured.obj.
- Robust texture handling:
  - chdir to OBJ folder before import (helps relative MTL/PNG)
  - optional --force_texture: ignores MTL and wires textured.png to Principled BSDF
- Docker-friendly / Blender 2.93 / Python 3.8 friendly
"""

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
    """
    Find a YCB textured.obj and (optionally) textured.png near it.

    Expected layout:
      <assets_root>/<obj_name>/<obj_name>/tsdf/textured.obj
      <assets_root>/<obj_name>/<obj_name>/tsdf/textured.png
    """
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
    """
    Replace/assign material with a simple ImageTexture -> Principled BSDF -> Output chain.
    Robust when MTL image paths don't resolve.
    """
    bo = bp_obj.blender_obj

    # Load image once (check_existing avoids duplicates)
    img = bpy.data.images.load(str(png_path), check_existing=True)

    # Create material
    mat = bpy.data.materials.new(name="ycb_tex_{}".format(bo.name))
    mat.use_nodes = True
    nt = mat.node_tree
    nodes = nt.nodes
    links = nt.links

    # Clear nodes
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

    # Assign
    if hasattr(bo.data, "materials"):
        if bo.data.materials:
            bo.data.materials[0] = mat
        else:
            bo.data.materials.append(mat)


def setup_cycles_gpu(prefer_optix: bool) -> None:
    """
    Try to enable Cycles GPU rendering. If not possible, Blender may fall back to CPU.
    """
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


def make_simple_plane(size: float = 3.0):
    plane = bproc.object.create_primitive("PLANE", scale=[size, size, 1.0])
    plane.set_location([0, 0, 0])
    plane.set_rotation_euler([0, 0, 0])
    return plane


def set_plane_color(plane, rgb: Tuple[float, float, float]) -> None:
    mat = bproc.material.create(name="bg_mat")
    mat.set_principled_shader_value("Base Color", [rgb[0], rgb[1], rgb[2], 1.0])
    mat.set_principled_shader_value("Roughness", 0.9)
    plane.replace_materials(mat)


def make_point_light():
    light = bproc.types.Light()
    light.set_type("POINT")
    light.set_energy(200.0)
    light.set_location([1.0, -1.0, 2.0])
    return light


def sample_cam_pose_look_at(
    poi: np.ndarray,
    r_min: float,
    r_max: float,
    z_min: float,
    z_max: float,
) -> np.ndarray:
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

    # Stage
    plane = make_simple_plane(size=3.0)
    set_plane_color(plane, (0.9, 0.9, 0.85))
    light = make_point_light()

    loaded = []  # list of (bp_obj, name, cat_id)
    for name, cat_id in YCB_DEFAULT_OBJECTS:
        obj_path, png_path = find_textured_obj(assets_root, name)

        if args.debug_print_paths:
            print("[YCB] {}: obj={} png={}".format(name, obj_path, png_path))

        # Key trick: chdir to OBJ folder so mtllib/map_Kd relative paths resolve
        prev_cwd = os.getcwd()
        os.chdir(str(obj_path.parent))
        try:
            # Blender 2.93 importer supports use_image_search
            bp_objs = bproc.loader.load_obj(str(obj_path), use_image_search=True)
        finally:
            os.chdir(prev_cwd)

        for bp_obj in bp_objs:
            bp_obj.set_cp("category_id", int(cat_id))
            bp_obj.set_name(name)

            # Robust fallback: ignore MTL and force assign textured.png
            if args.force_texture and png_path is not None:
                try:
                    force_apply_png_texture(bp_obj, png_path)
                except Exception as e:
                    print("[WARN] force texture failed for {}: {}".format(name, e))

            loaded.append((bp_obj, name, cat_id))

    hide_loc = np.array([1000.0, 1000.0, 1000.0])

    poi = np.array([0.0, 0.0, 0.08])

    for i in range(args.n_images):
        # light jitter per frame (helps domain randomization)
        light.set_location(
            [
                float(np.random.uniform(-1.2, 1.2)),
                float(np.random.uniform(-1.2, 1.2)),
                float(np.random.uniform(1.2, 2.5)),
            ],
            frame=i,
        )
        light.set_energy(float(np.random.uniform(80.0, 260.0)), frame=i)

        # choose 1 or 2 objects
        if args.only_single_object:
            chosen = [random.choice(loaded)]
        else:
            k = 1 if random.random() < 0.5 else 2
            chosen = random.sample(loaded, k=min(k, len(loaded)))

        # hide all
        for bp_obj, _, _ in loaded:
            bp_obj.set_location(hide_loc.tolist(), frame=i)

        # place chosen with simple separation
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

            # mainly yaw rotation
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
    seg = bproc.renderer.render_segmap(map_by=["instance", "class", "name"])

    # COCO writer
    bproc.writer.write_coco_annotations(
        str(coco_dir),
        instance_segmaps=seg["instance_segmaps"],
        instance_attribute_maps=seg["instance_attribute_maps"],
        colors=data["colors"],
        color_file_format="JPEG",
    )

    print("[OK] Wrote COCO dataset to:", coco_dir)
    print("     - coco_annotations.json")
    print("     - rgb_*.jpg (and masks/segmaps as needed)")

    # Quick hint for sanity check
    print("[TIP] Check output:")
    print("  ls -1 {}/*.jpg | head".format(coco_dir))
    print("  ls -lh {}/coco_annotations.json".format(coco_dir))


if __name__ == "__main__":
    main()


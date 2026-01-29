
import blenderproc as bproc  # MUST be the first import (BlenderProc requirement)
"""
render_ycb_coco_pbr_modes_bg.py

BlenderProc v2 script:
- Load YCB objects (Berkeley meshes) with strict mesh_preference (e.g., tsdf only).
- Randomly choose 1 of 3 scene modes per frame:
    floor_mode : floor + wall, objects on floor (optionally farther camera)
    shelf_mode : shelf board + backboard, objects on shelf (stronger shadow feel)
    table_mode : tabletop plane, objects on table (surface texture varied)
- Three-point lights (key/fill/rim) randomized per frame.
- Optional low-frequency blur.
- Add primitive clutter (2-6) with random materials.
- Randomize roughness/specular.
- Background plane compositing (方式A): render RGBA with transparent film,
  then composite onto a real background image from assets/backgrounds/<split>/*.

Outputs:
- output_dir/images/000000.jpg ...
- output_dir/coco_annotations.json  (COCO bbox annotations; segmentation not used)
- output_dir/meta.json (run metadata)

Notes / design choices:
- Avoids BlenderProc segmentation APIs (often missing in some "blenderproc/blenderproc:v2" builds).
- Bounding boxes are computed from projected 3D bounding box (no per-pixel occlusion test).
- OBJ import failures are caught and the object is skipped (pipeline continues).
- JPEG is written atomically (tmp -> rename). If JPEG encoder fails, fallback to PNG.

CLI example:
docker run --rm -it --gpus all --ipc=host \
  -v "$PWD:/work" -v /dev/shm:/dev/shm \
  blenderproc/blenderproc:v2 \
  blenderproc run /work/scripts/render_ycb_coco_pbr_modes_bg.py -- \
    --base_config /work/configs/synth_config.json \
    --dr_config /work/configs/synth_dr_config.json \
    --paths_config /work/configs/synth_dr_paths.json \
    --split val \
    --num_images 30 \
    --output_dir /work/out/coco_bg_val \
    --seed 1 \
    --overwrite
"""


import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# These imports are safe AFTER blenderproc import:
import bpy  # type: ignore
from mathutils import Matrix, Vector  # type: ignore

# Pillow is bundled in BlenderProc's python typically.
try:
    from PIL import Image, ImageFile, ImageFilter
    ImageFile.LOAD_TRUNCATED_IMAGES = True
except Exception:  # pragma: no cover
    Image = None  # type: ignore
    ImageFile = None  # type: ignore
    ImageFilter = None  # type: ignore


# -----------------------------
# Utilities
# -----------------------------

def _now_ms() -> int:
    return int(time.time() * 1000)

def _read_json(path: str) -> Dict[str, Any]:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8"))

def _mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _is_nonempty_dir(p: Path) -> bool:
    return p.is_dir() and any(p.iterdir())

def _safe_relpath(p: Path, root: Path) -> str:
    try:
        return str(p.relative_to(root))
    except Exception:
        return str(p)

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def _uniform(rng: random.Random, a: float, b: float) -> float:
    if a > b:
        a, b = b, a
    return a + (b - a) * rng.random()

def _choice(rng: random.Random, xs: Sequence[Any]) -> Any:
    return xs[int(rng.random() * len(xs))]

def _rand_rgb(rng: random.Random, jitter: float = 0.2) -> Tuple[float, float, float]:
    # base neutral-ish light with mild color variation
    base = 1.0
    r = _clamp(base + rng.uniform(-jitter, jitter), 0.2, 2.0)
    g = _clamp(base + rng.uniform(-jitter, jitter), 0.2, 2.0)
    b = _clamp(base + rng.uniform(-jitter, jitter), 0.2, 2.0)
    return (r, g, b)

def _atomic_write_bytes(dst: Path, data: bytes) -> None:
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    _mkdir(dst.parent)
    tmp.write_bytes(data)
    os.replace(str(tmp), str(dst))

def _atomic_copytree(src: Path, dst: Path) -> None:
    # dst must not exist; use a tmp dir then rename
    tmp = dst.parent / (dst.name + ".tmp_" + str(_now_ms()))
    if tmp.exists():
        shutil.rmtree(tmp)
    shutil.copytree(src, tmp, dirs_exist_ok=False)
    os.replace(str(tmp), str(dst))

def _list_images(glob_pattern: str) -> List[Path]:
    # Expand glob; accept jpg/png/webp/etc
    return sorted([Path(p) for p in glob.glob(glob_pattern)])  # type: ignore

# glob is only used here
import glob  # noqa: E402


# -----------------------------
# Config models
# -----------------------------

@dataclass
class CameraCfg:
    fx: float
    fy: float
    cx: float
    cy: float
    loc_min: Tuple[float, float, float]
    loc_max: Tuple[float, float, float]
    inplane_rot_min: float
    inplane_rot_max: float

@dataclass
class TableCfg:
    size: float
    height: float

@dataclass
class RenderCfg:
    engine: str
    samples: int
    use_denoise: bool

@dataclass
class ExportCfg:
    image_ext: str
    jpg_quality: int
    blur_prob: float
    blur_sigma_min: float
    blur_sigma_max: float

@dataclass
class ResourcesCfg:
    cctextures_root: Path
    backgrounds_train_glob: str
    backgrounds_val_glob: str

@dataclass
class BaseCfg:
    seed: int
    ycb_root: Path
    mesh_preference: List[str]
    classes: List[str]
    image_width: int
    image_height: int
    num_images: int
    min_objects_per_image: int
    max_objects_per_image: int
    camera: CameraCfg
    table: TableCfg
    resources: ResourcesCfg
    render: RenderCfg
    export: ExportCfg

@dataclass
class LightingDR:
    key_energy: Tuple[float, float]
    fill_energy: Tuple[float, float]
    rim_energy: Tuple[float, float]
    elev_deg: Tuple[float, float]
    dist: Tuple[float, float]
    color_jitter: float

@dataclass
class MaterialDR:
    roughness: Tuple[float, float]
    specular: Tuple[float, float]
    metallic: Tuple[float, float]

@dataclass
class ClutterDR:
    count: Tuple[int, int]
    size: Tuple[float, float]
    height: Tuple[float, float]

@dataclass
class ModesDR:
    p_floor: float
    p_shelf: float
    p_table: float

@dataclass
class DRConfig:
    lighting: LightingDR
    material: MaterialDR
    clutter: ClutterDR
    modes: ModesDR


def _parse_base_cfg(d: Dict[str, Any]) -> BaseCfg:
    cam = d.get("camera", {})
    tbl = d.get("table", {})
    res = d.get("resources", {})
    ren = d.get("render", {})
    exp = d.get("export", {})

    camera = CameraCfg(
        fx=float(cam.get("fx", 600.0)),
        fy=float(cam.get("fy", 600.0)),
        cx=float(cam.get("cx", d.get("image_width", 640) / 2)),
        cy=float(cam.get("cy", d.get("image_height", 480) / 2)),
        loc_min=tuple(cam.get("loc_min", [-0.6, -0.6, 0.45])),
        loc_max=tuple(cam.get("loc_max", [0.6, 0.6, 1.1])),
        inplane_rot_min=float(cam.get("inplane_rot_min", -0.6)),
        inplane_rot_max=float(cam.get("inplane_rot_max", 0.6)),
    )

    table = TableCfg(
        size=float(tbl.get("size", 1.4)),
        height=float(tbl.get("height", 0.75)),
    )

    resources = ResourcesCfg(
        cctextures_root=Path(res.get("cctextures_root", "resources/cctextures")),
        backgrounds_train_glob=str(res.get("backgrounds_train_glob", "assets/backgrounds/train/*.*")),
        backgrounds_val_glob=str(res.get("backgrounds_val_glob", "assets/backgrounds/val/*.*")),
    )

    render = RenderCfg(
        engine=str(ren.get("engine", "CYCLES")).upper(),
        samples=int(ren.get("samples", 32)),
        use_denoise=bool(ren.get("use_denoise", True)),
    )

    export = ExportCfg(
        image_ext=str(exp.get("image_ext", "jpg")).lower(),
        jpg_quality=int(exp.get("jpg_quality", 90)),
        blur_prob=float(exp.get("blur_prob", 0.05)),
        blur_sigma_min=float(exp.get("blur_sigma_min", 0.4)),
        blur_sigma_max=float(exp.get("blur_sigma_max", 1.2)),
    )

    return BaseCfg(
        seed=int(d.get("seed", 0)),
        ycb_root=Path(d.get("ycb_root", "models/ycb")),
        mesh_preference=list(d.get("mesh_preference", ["tsdf"])),
        classes=list(d.get("classes", [])),
        image_width=int(d.get("image_width", 640)),
        image_height=int(d.get("image_height", 480)),
        num_images=int(d.get("num_images", 200)),
        min_objects_per_image=int(d.get("min_objects_per_image", 1)),
        max_objects_per_image=int(d.get("max_objects_per_image", 4)),
        camera=camera,
        table=table,
        resources=resources,
        render=render,
        export=export,
    )

def _parse_dr_cfg(d: Dict[str, Any]) -> DRConfig:
    # Reasonable defaults for "DR"
    L = d.get("lighting", {})
    M = d.get("material", {})
    C = d.get("clutter", {})
    P = d.get("modes", {})

    lighting = LightingDR(
        key_energy=tuple(L.get("key_energy", [800.0, 1800.0])),
        fill_energy=tuple(L.get("fill_energy", [150.0, 600.0])),
        rim_energy=tuple(L.get("rim_energy", [200.0, 900.0])),
        elev_deg=tuple(L.get("elev_deg", [15.0, 75.0])),
        dist=tuple(L.get("dist", [1.2, 2.5])),
        color_jitter=float(L.get("color_jitter", 0.25)),
    )

    material = MaterialDR(
        roughness=tuple(M.get("roughness", [0.15, 0.85])),
        specular=tuple(M.get("specular", [0.1, 0.8])),
        metallic=tuple(M.get("metallic", [0.0, 0.25])),
    )

    clutter = ClutterDR(
        count=tuple(C.get("count", [2, 6])),
        size=tuple(C.get("size", [0.03, 0.12])),
        height=tuple(C.get("height", [0.0, 0.02])),
    )

    modes = ModesDR(
        p_floor=float(P.get("p_floor", 0.34)),
        p_shelf=float(P.get("p_shelf", 0.33)),
        p_table=float(P.get("p_table", 0.33)),
    )

    return DRConfig(lighting=lighting, material=material, clutter=clutter, modes=modes)


# -----------------------------
# YCB loading (robust)
# -----------------------------

def _alt_names(name: str) -> List[str]:
    # Try both "-" and "_" variants
    s = set([name, name.replace("-", "_"), name.replace("_", "-")])
    return list(s)

def _find_ycb_obj_mesh(ycb_root: Path, cls_name: str, mesh_pref: List[str]) -> Optional[Path]:
    # Expected folder patterns in Berkeley tgz:
    # ycb_root/<obj>/<obj>/<mesh_type>/textured.obj
    # where mesh_type in {tsdf, poisson, google_16k} (google_16k sometimes absent)
    # Some objects include "-" in folder name (e.g., 027-skillet).
    for obj_dir_name in _alt_names(cls_name):
        obj_dir = ycb_root / obj_dir_name
        if not obj_dir.exists():
            continue
        # inside might be <obj_dir>/<something>/<mesh>/textured.obj
        inner_candidates = [p for p in obj_dir.iterdir() if p.is_dir()]
        # many tgz extract into obj_dir/<obj_name>/...
        # but sometimes obj_dir itself is the inner folder; handle both
        inner_candidates.append(obj_dir)

        for inner in inner_candidates:
            for mesh in mesh_pref:
                p = inner / mesh / "textured.obj"
                if p.is_file():
                    return p
                # sometimes nested exactly as <obj>/<obj>/<mesh>/textured.obj
                p2 = inner / obj_dir_name / mesh / "textured.obj"
                if p2.is_file():
                    return p2
                # sometimes inner is already <obj>/<obj>
                p3 = obj_dir / obj_dir_name / mesh / "textured.obj"
                if p3.is_file():
                    return p3
    return None

def _load_ycb_objects(
    ycb_root: Path,
    classes: List[str],
    mesh_pref: List[str],
) -> Dict[str, "bproc.types.MeshObject"]:
    loaded: Dict[str, "bproc.types.MeshObject"] = {}
    for cls in classes:
        mesh_path = _find_ycb_obj_mesh(ycb_root, cls, mesh_pref)
        if mesh_path is None:
            print(f"[WARN] missing mesh for class={cls} (pref={mesh_pref}) -> skip")
            continue
        try:
            parts = bproc.loader.load_obj(str(mesh_path))
            if not parts:
                print(f"[WARN] loaded 0 parts for {cls} from {mesh_path} -> skip")
                continue
            # Merge parts into one object for bbox/pose handling
            obj = parts[0]
            for p in parts[1:]:
                try:
                    obj.join_with_other_objects([p])
                except Exception:
                    pass
            obj.set_name(cls)
            loaded[cls] = obj
            # Ensure smooth shading (helps specular look)
            try:
                obj.shade_smooth()
            except Exception:
                pass
            print(f"[OK] loaded {cls} mesh={mesh_path}")
        except Exception as e:
            print(f"[WARN] OBJ import failed for {cls} at {mesh_path}: {e} -> skip")
            continue
    print(f"[INFO] loaded objects: {len(loaded)}/{len(classes)}")
    return loaded


# -----------------------------
# CC textures (PBR) assignment
# -----------------------------

def _scan_texture_sets(root: Path) -> List[Path]:
    if not root.exists():
        return []
    sets = [p for p in root.iterdir() if p.is_dir()]
    # Filter out obvious junk folders if any; keep most
    return sorted(sets)

def _find_map_file(tex_dir: Path, keys: List[str], exts: Tuple[str, ...] = (".jpg", ".png", ".jpeg", ".tif", ".tiff")) -> Optional[Path]:
    files = []
    for ext in exts:
        files.extend(list(tex_dir.rglob(f"*{ext}")))
    # Prefer direct children, but accept nested
    def score(p: Path) -> int:
        s = p.name.lower()
        kscore = 0
        for k in keys:
            if k in s:
                kscore += 10
        # prefer color/albedo over others if requested
        # prefer not too deep
        depth = len(p.relative_to(tex_dir).parts)
        return kscore * 100 - depth
    best = None
    best_sc = -10**9
    for p in files:
        s = p.name.lower()
        if any(k in s for k in keys):
            sc = score(p)
            if sc > best_sc:
                best_sc = sc
                best = p
    return best

def _ensure_uv(obj: "bproc.types.MeshObject") -> None:
    # Some OBJ already has UVs; if not, do a simple smart projection.
    try:
        bo = obj.blender_obj  # type: ignore
    except Exception:
        return
    try:
        me = bo.data
        if me is None:
            return
        if me.uv_layers and len(me.uv_layers) > 0:
            return
        # Create UV map
        bpy.context.view_layer.objects.active = bo
        bo.select_set(True)
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.uv.smart_project(angle_limit=66.0, island_margin=0.02)
        bpy.ops.object.mode_set(mode="OBJECT")
        bo.select_set(False)
    except Exception:
        try:
            bpy.ops.object.mode_set(mode="OBJECT")
        except Exception:
            pass

def _apply_pbr_material_from_dir(obj: "bproc.types.MeshObject", tex_dir: Path, rng: random.Random) -> None:
    """
    Build a Principled BSDF material and connect:
    - Base Color (Color/Albedo/BaseColor)
    - Normal map if available
    - Roughness map if available
    - Metallic map if available (rare for these)
    """
    _ensure_uv(obj)

    # Find maps
    color = _find_map_file(tex_dir, ["color", "albedo", "basecolor", "diffuse"])
    normal = _find_map_file(tex_dir, ["normal"])
    rough = _find_map_file(tex_dir, ["roughness", "rough"])
    metal = _find_map_file(tex_dir, ["metallic", "metalness", "metal"])

    # Create material
    mat = bpy.data.materials.new(name=f"MAT_{tex_dir.name}_{rng.randint(0, 999999)}")
    mat.use_nodes = True
    nt = mat.node_tree
    assert nt is not None
    nodes = nt.nodes
    links = nt.links

    # Clear default nodes
    for n in list(nodes):
        nodes.remove(n)

    out = nodes.new(type="ShaderNodeOutputMaterial")
    out.location = (400, 0)
    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf.location = (0, 0)
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    def add_tex_node(path: Path, loc: Tuple[float, float]) -> bpy.types.Node:
        tex = nodes.new(type="ShaderNodeTexImage")
        tex.location = loc
        img = bpy.data.images.load(str(path), check_existing=True)
        tex.image = img
        tex.interpolation = "Smart"
        return tex

    if color is not None:
        tex = add_tex_node(color, (-600, 120))
        links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])

    if rough is not None:
        texr = add_tex_node(rough, (-600, -60))
        # roughness map should be non-color
        try:
            texr.image.colorspace_settings.name = "Non-Color"
        except Exception:
            pass
        links.new(texr.outputs["Color"], bsdf.inputs["Roughness"])

    if metal is not None:
        texm = add_tex_node(metal, (-600, -220))
        try:
            texm.image.colorspace_settings.name = "Non-Color"
        except Exception:
            pass
        links.new(texm.outputs["Color"], bsdf.inputs["Metallic"])

    if normal is not None:
        texn = add_tex_node(normal, (-600, -420))
        try:
            texn.image.colorspace_settings.name = "Non-Color"
        except Exception:
            pass
        nmap = nodes.new(type="ShaderNodeNormalMap")
        nmap.location = (-250, -420)
        links.new(texn.outputs["Color"], nmap.inputs["Color"])
        links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])

    # Assign to object
    try:
        bo = obj.blender_obj  # type: ignore
        if bo.data.materials:
            bo.data.materials[0] = mat
        else:
            bo.data.materials.append(mat)
    except Exception:
        pass

def _set_material_dr_on_object(obj: "bproc.types.MeshObject", dr: DRConfig, rng: random.Random) -> None:
    # Randomize Principled BSDF params if present
    try:
        bo = obj.blender_obj  # type: ignore
        mats = list(getattr(bo.data, "materials", []))
    except Exception:
        mats = []
    for mat in mats:
        if mat is None or not getattr(mat, "use_nodes", False):
            continue
        nt = mat.node_tree
        if nt is None:
            continue
        for n in nt.nodes:
            if n.type == "BSDF_PRINCIPLED":
                try:
                    n.inputs["Roughness"].default_value = _uniform(rng, float(dr.material.roughness[0]), float(dr.material.roughness[1]))
                except Exception:
                    pass
                try:
                    n.inputs["Specular"].default_value = _uniform(rng, float(dr.material.specular[0]), float(dr.material.specular[1]))
                except Exception:
                    pass
                try:
                    n.inputs["Metallic"].default_value = _uniform(rng, float(dr.material.metallic[0]), float(dr.material.metallic[1]))
                except Exception:
                    pass


# -----------------------------
# Background image I/O + compositing
# -----------------------------

def _load_bg_image(path: Path) -> np.ndarray:
    """
    Return uint8 RGB array.
    Handles truncated JPEGs.
    """
    if Image is None:
        raise RuntimeError("PIL not available in this BlenderProc image; cannot load backgrounds.")
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            return np.array(im)
    except Exception:
        # last resort: read bytes then open
        b = path.read_bytes()
        from io import BytesIO
        with Image.open(BytesIO(b)) as im:
            im = im.convert("RGB")
            return np.array(im)

def _resize_bg(bg: np.ndarray, w: int, h: int) -> np.ndarray:
    if Image is None:
        raise RuntimeError("PIL not available.")
    im = Image.fromarray(bg)
    im = im.resize((w, h), resample=Image.BILINEAR)
    return np.array(im)

def _composite_rgba_on_bg(rgba: np.ndarray, bg_rgb: np.ndarray) -> np.ndarray:
    """
    rgba: float32 or uint8, shape (H,W,4)
    bg_rgb: uint8 (H,W,3)
    returns uint8 (H,W,3)
    """
    if rgba.dtype != np.float32:
        rgba_f = rgba.astype(np.float32) / 255.0
    else:
        rgba_f = rgba
    if bg_rgb.dtype != np.float32:
        bg_f = bg_rgb.astype(np.float32) / 255.0
    else:
        bg_f = bg_rgb
    a = rgba_f[..., 3:4]
    rgb = rgba_f[..., :3] * a + bg_f[..., :3] * (1.0 - a)
    out = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    return out

def _maybe_blur(rgb: np.ndarray, rng: random.Random, prob: float, s0: float, s1: float) -> np.ndarray:
    if Image is None or ImageFilter is None:
        return rgb
    if rng.random() > prob:
        return rgb
    sigma = _uniform(rng, s0, s1)
    im = Image.fromarray(rgb)
    im = im.filter(ImageFilter.GaussianBlur(radius=float(sigma)))
    return np.array(im)

def _save_rgb_atomic(rgb: np.ndarray, out_path: Path, jpg_quality: int = 90) -> None:
    """
    Save RGB as JPEG/PNG with atomic rename. Falls back to PNG on JPEG encoder errors.
    """
    _mkdir(out_path.parent)
    suffix = out_path.suffix.lower()
    if Image is None:
        raise RuntimeError("PIL not available; cannot write images.")

    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    im = Image.fromarray(rgb)

    try:
        if suffix in [".jpg", ".jpeg"]:
            im.save(tmp, format="JPEG", quality=int(jpg_quality), subsampling=0, optimize=True)
        elif suffix == ".png":
            im.save(tmp, format="PNG", compress_level=3)
        else:
            # default to PNG if unknown
            im.save(tmp, format="PNG", compress_level=3)
        os.replace(str(tmp), str(out_path))
    except Exception as e:
        # fallback: write PNG instead
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass
        fallback = out_path.with_suffix(".png")
        tmp2 = fallback.with_suffix(".png.tmp")
        im.save(tmp2, format="PNG", compress_level=3)
        os.replace(str(tmp2), str(fallback))
        raise RuntimeError(f"Image save failed for {out_path} (fallback saved to {fallback}): {e}")


# -----------------------------
# Scene building: modes
# -----------------------------

def _create_plane(name: str, size: float, location: Tuple[float, float, float], rotation_euler: Tuple[float, float, float]) -> bpy.types.Object:
    bpy.ops.mesh.primitive_plane_add(size=size, location=location, rotation=rotation_euler)
    obj = bpy.context.active_object
    assert obj is not None
    obj.name = name
    return obj

def _delete_all_objects() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

def _look_at_rotation(from_pos: Vector, to_pos: Vector, up: Vector = Vector((0, 0, 1))) -> Tuple[float, float, float]:
    direction = (to_pos - from_pos).normalized()
    # Blender "track to" convention: -Z forward, Y up for cameras; lights can be similar.
    quat = direction.to_track_quat('-Z', 'Y')
    eul = quat.to_euler()
    return (float(eul.x), float(eul.y), float(eul.z))

def _ensure_cycles(engine: str, samples: int, denoise: bool) -> None:
    engine = engine.upper()
    if engine not in ["CYCLES", "EEVEE"]:
        engine = "CYCLES"

    bproc.renderer.set_render_engine(engine)
    # Many BlenderProc builds expose this:
    try:
        bproc.renderer.set_max_amount_of_samples(int(samples))
    except Exception:
        # fallback via bpy
        try:
            bpy.context.scene.cycles.samples = int(samples)
        except Exception:
            pass

    # Transparent film for RGBA
    try:
        bpy.context.scene.render.film_transparent = True
    except Exception:
        pass

    # Denoise
    try:
        if hasattr(bpy.context.scene, "cycles"):
            bpy.context.scene.cycles.use_denoising = bool(denoise)
    except Exception:
        pass

def _setup_camera(base: BaseCfg) -> None:
    K = np.array([[base.camera.fx, 0, base.camera.cx],
                  [0, base.camera.fy, base.camera.cy],
                  [0, 0, 1]], dtype=np.float32)
    bproc.camera.set_intrinsics_from_K_matrix(K, base.image_width, base.image_height)

def _random_camera_pose(rng: random.Random, base: BaseCfg, focus: Vector) -> Matrix:
    x = _uniform(rng, base.camera.loc_min[0], base.camera.loc_max[0])
    y = _uniform(rng, base.camera.loc_min[1], base.camera.loc_max[1])
    z = _uniform(rng, base.camera.loc_min[2], base.camera.loc_max[2])
    cam_pos = Vector((x, y, z))

    rot = _look_at_rotation(cam_pos, focus)
    # small in-plane rotation
    roll = _uniform(rng, base.camera.inplane_rot_min, base.camera.inplane_rot_max)
    rot = (rot[0], rot[1], rot[2] + roll)

    cam2world = bproc.math.build_transformation_mat(np.array(cam_pos), np.array(rot))
    return cam2world

def _add_three_point_lights(rng: random.Random, dr: DRConfig, target: Vector) -> List[bpy.types.Object]:
    lights: List[bpy.types.Object] = []

    def add_light(name: str, energy_range: Tuple[float, float], az_deg: float, elev_deg: float, dist: float, color: Tuple[float, float, float]) -> bpy.types.Object:
        bpy.ops.object.light_add(type="AREA", location=(0, 0, 0))
        L = bpy.context.active_object
        assert L is not None
        L.name = name
        # position in spherical coords around target
        az = math.radians(az_deg)
        el = math.radians(elev_deg)
        px = target.x + dist * math.cos(el) * math.cos(az)
        py = target.y + dist * math.cos(el) * math.sin(az)
        pz = target.z + dist * math.sin(el)
        L.location = (px, py, pz)
        L.rotation_euler = _look_at_rotation(Vector(L.location), target)

        # size (softness)
        try:
            L.data.size = 0.5
        except Exception:
            pass

        try:
            L.data.energy = _uniform(rng, float(energy_range[0]), float(energy_range[1]))
        except Exception:
            pass
        try:
            L.data.color = color
        except Exception:
            pass
        return L

    elev0, elev1 = dr.lighting.elev_deg
    dist0, dist1 = dr.lighting.dist
    jitter = dr.lighting.color_jitter

    key = add_light(
        "L_key",
        dr.lighting.key_energy,
        az_deg=_uniform(rng, -60, 60),
        elev_deg=_uniform(rng, float(elev0), float(elev1)),
        dist=_uniform(rng, float(dist0), float(dist1)),
        color=_rand_rgb(rng, jitter),
    )
    fill = add_light(
        "L_fill",
        dr.lighting.fill_energy,
        az_deg=_uniform(rng, 90, 150),
        elev_deg=_uniform(rng, float(elev0), float(elev1)),
        dist=_uniform(rng, float(dist0), float(dist1)),
        color=_rand_rgb(rng, jitter),
    )
    rim = add_light(
        "L_rim",
        dr.lighting.rim_energy,
        az_deg=_uniform(rng, -150, -90),
        elev_deg=_uniform(rng, float(elev0), float(elev1)),
        dist=_uniform(rng, float(dist0), float(dist1)),
        color=_rand_rgb(rng, jitter),
    )
    lights.extend([key, fill, rim])
    return lights

def _pick_mode(rng: random.Random, dr: DRConfig) -> str:
    pf = dr.modes.p_floor
    ps = dr.modes.p_shelf
    pt = dr.modes.p_table
    s = pf + ps + pt
    if s <= 0:
        return "table_mode"
    r = rng.random() * s
    if r < pf:
        return "floor_mode"
    if r < pf + ps:
        return "shelf_mode"
    return "table_mode"

def _apply_surface_texture(surface_obj: bpy.types.Object, tex_dir: Path, rng: random.Random) -> None:
    # Wrap the Blender object in a bproc "MeshObject" only to reuse UV/material code
    # But BlenderProc doesn't provide a direct constructor; so we build nodes directly here.
    mat = bpy.data.materials.new(name=f"SURF_{surface_obj.name}_{rng.randint(0,999999)}")
    mat.use_nodes = True
    nt = mat.node_tree
    assert nt is not None
    nodes = nt.nodes
    links = nt.links
    for n in list(nodes):
        nodes.remove(n)
    out = nodes.new(type="ShaderNodeOutputMaterial")
    out.location = (400, 0)
    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf.location = (0, 0)
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    def add_tex(path: Path, loc: Tuple[float, float], noncolor: bool = False) -> bpy.types.Node:
        tex = nodes.new(type="ShaderNodeTexImage")
        tex.location = loc
        img = bpy.data.images.load(str(path), check_existing=True)
        tex.image = img
        if noncolor:
            try:
                img.colorspace_settings.name = "Non-Color"
            except Exception:
                pass
        return tex

    color = _find_map_file(tex_dir, ["color", "albedo", "basecolor", "diffuse"])
    normal = _find_map_file(tex_dir, ["normal"])
    rough = _find_map_file(tex_dir, ["roughness", "rough"])
    metal = _find_map_file(tex_dir, ["metallic", "metalness", "metal"])

    if color is not None:
        t = add_tex(color, (-600, 120), noncolor=False)
        links.new(t.outputs["Color"], bsdf.inputs["Base Color"])
    if rough is not None:
        t = add_tex(rough, (-600, -60), noncolor=True)
        links.new(t.outputs["Color"], bsdf.inputs["Roughness"])
    if metal is not None:
        t = add_tex(metal, (-600, -220), noncolor=True)
        links.new(t.outputs["Color"], bsdf.inputs["Metallic"])
    if normal is not None:
        t = add_tex(normal, (-600, -420), noncolor=True)
        nmap = nodes.new(type="ShaderNodeNormalMap")
        nmap.location = (-250, -420)
        links.new(t.outputs["Color"], nmap.inputs["Color"])
        links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])

    surface_obj.data.materials.clear()
    surface_obj.data.materials.append(mat)

def _build_mode_geometry(mode: str, base: BaseCfg, tex_sets: List[Path], rng: random.Random) -> Dict[str, Any]:
    """
    Returns dict with:
      - "support_z": z height where objects should be placed
      - "region": tuple(xmin,xmax,ymin,ymax) object placement region
      - "focus": Vector focus point for camera & lights
    """
    # Clean everything (except camera which BlenderProc handles)
    _delete_all_objects()

    # Minimal world
    bpy.context.scene.world.use_nodes = True  # type: ignore

    # Pick texture for surfaces
    tex_dir = _choice(rng, tex_sets) if tex_sets else None

    if mode == "floor_mode":
        # floor plane at z=0, size large, plus wall at y=+size/2
        floor = _create_plane("floor", size=4.0, location=(0, 0, 0.0), rotation_euler=(0, 0, 0))
        wall = _create_plane("wall", size=4.0, location=(0, 2.0, 2.0), rotation_euler=(math.radians(90), 0, 0))
        if tex_dir is not None:
            _apply_surface_texture(floor, tex_dir, rng)
            _apply_surface_texture(wall, tex_dir, rng)
        support_z = 0.0
        region = (-0.8, 0.8, -0.8, 0.8)
        focus = Vector((0, 0, 0.15))
        return {"support_z": support_z, "region": region, "focus": focus}

    if mode == "shelf_mode":
        # shelf board and backboard; put at around z=0.9
        shelf_z = 0.9
        board = _create_plane("shelf_board", size=2.0, location=(0, 0, shelf_z), rotation_euler=(0, 0, 0))
        back = _create_plane("shelf_back", size=2.0, location=(0, 0.9, shelf_z + 0.9), rotation_euler=(math.radians(90), 0, 0))
        if tex_dir is not None:
            _apply_surface_texture(board, tex_dir, rng)
            _apply_surface_texture(back, tex_dir, rng)
        support_z = shelf_z
        region = (-0.5, 0.5, -0.35, 0.35)
        focus = Vector((0, 0, shelf_z + 0.12))
        return {"support_z": support_z, "region": region, "focus": focus}

    # table_mode (default)
    table_z = float(base.table.height)
    top = _create_plane("table_top", size=float(base.table.size), location=(0, 0, table_z), rotation_euler=(0, 0, 0))
    if tex_dir is not None:
        _apply_surface_texture(top, tex_dir, rng)
    support_z = table_z
    region = (-0.55, 0.55, -0.40, 0.40)
    focus = Vector((0, 0, table_z + 0.12))
    return {"support_z": support_z, "region": region, "focus": focus}


# -----------------------------
# Object placement + clutter
# -----------------------------

def _random_xy_in_region(rng: random.Random, region: Tuple[float, float, float, float]) -> Tuple[float, float]:
    xmin, xmax, ymin, ymax = region
    x = _uniform(rng, xmin, xmax)
    y = _uniform(rng, ymin, ymax)
    return x, y

def _spawn_clutter(dr: DRConfig, rng: random.Random, region: Tuple[float, float, float, float], z: float) -> List[bpy.types.Object]:
    c0, c1 = dr.clutter.count
    n = rng.randint(int(c0), int(c1))
    objs: List[bpy.types.Object] = []

    for i in range(n):
        shape = _choice(rng, ["cube", "cylinder", "cone", "uvsphere"])
        size = _uniform(rng, float(dr.clutter.size[0]), float(dr.clutter.size[1]))
        x, y = _random_xy_in_region(rng, region)
        dz = _uniform(rng, float(dr.clutter.height[0]), float(dr.clutter.height[1]))
        loc = (x, y, z + dz + size * 0.5)

        if shape == "cube":
            bpy.ops.mesh.primitive_cube_add(size=size, location=loc)
        elif shape == "cylinder":
            bpy.ops.mesh.primitive_cylinder_add(radius=size * 0.35, depth=size, location=loc)
        elif shape == "cone":
            bpy.ops.mesh.primitive_cone_add(radius1=size * 0.35, depth=size, location=loc)
        else:
            bpy.ops.mesh.primitive_uv_sphere_add(radius=size * 0.35, location=loc)

        o = bpy.context.active_object
        assert o is not None
        o.name = f"clutter_{i:02d}"

        # Simple material with random roughness/specular
        mat = bpy.data.materials.new(name=f"CLUT_{i}_{rng.randint(0,999999)}")
        mat.use_nodes = True
        nt = mat.node_tree
        assert nt is not None
        bsdf = None
        for nnode in nt.nodes:
            if nnode.type == "BSDF_PRINCIPLED":
                bsdf = nnode
                break
        if bsdf is not None:
            try:
                bsdf.inputs["Base Color"].default_value = (rng.random(), rng.random(), rng.random(), 1.0)
            except Exception:
                pass
            try:
                bsdf.inputs["Roughness"].default_value = _uniform(rng, float(dr.material.roughness[0]), float(dr.material.roughness[1]))
            except Exception:
                pass
            try:
                bsdf.inputs["Specular"].default_value = _uniform(rng, float(dr.material.specular[0]), float(dr.material.specular[1]))
            except Exception:
                pass
            try:
                bsdf.inputs["Metallic"].default_value = _uniform(rng, float(dr.material.metallic[0]), float(dr.material.metallic[1]))
            except Exception:
                pass

        o.data.materials.clear()
        o.data.materials.append(mat)

        # random rotation
        o.rotation_euler = (rng.random() * 2 * math.pi, rng.random() * 2 * math.pi, rng.random() * 2 * math.pi)
        objs.append(o)
    return objs

def _place_ycb_instance(
    template: "bproc.types.MeshObject",
    cls_name: str,
    rng: random.Random,
    region: Tuple[float, float, float, float],
    support_z: float,
) -> "bproc.types.MeshObject":
    inst = template.duplicate()
    inst.set_name(cls_name)

    x, y = _random_xy_in_region(rng, region)
    # random yaw
    yaw = rng.random() * 2 * math.pi
    # small tilt
    tilt_x = rng.uniform(-0.15, 0.15)
    tilt_y = rng.uniform(-0.15, 0.15)

    # Place slightly above support plane; no physics simulation here
    # Use bbox to approximate "standing" height
    try:
        bb = inst.get_bound_box()
        zmin = float(np.min(bb[:, 2]))
        zmax = float(np.max(bb[:, 2]))
        height = max(1e-4, zmax - zmin)
    except Exception:
        height = 0.10

    z = support_z + height * 0.5 + rng.uniform(0.0, 0.01)

    inst.set_location([x, y, z])
    inst.set_rotation_euler([tilt_x, tilt_y, yaw])
    return inst


# -----------------------------
# COCO bbox computation
# -----------------------------

def _project_world_to_image(P_world: np.ndarray, cam2world: np.ndarray) -> np.ndarray:
    # BlenderProc has project_points, but we implement using bproc.camera.project_points when possible
    # P_world: (N,3)
    pts = bproc.camera.project_points(P_world, cam2world)  # returns (N,2) pixel coords
    return pts

def _bbox_from_object(obj: "bproc.types.MeshObject", cam2world: np.ndarray, w: int, h: int) -> Optional[Tuple[float, float, float, float]]:
    # Project 3D bbox corners -> 2D bbox
    try:
        bb = obj.get_bound_box()  # (8,3) in world? BlenderProc returns in local; so we transform from object matrix
        # Convert local bbox corners to world using Blender object matrix_world
        bo = obj.blender_obj  # type: ignore
        mw = np.array(bo.matrix_world, dtype=np.float32)  # 4x4
        bb_h = np.concatenate([bb.astype(np.float32), np.ones((bb.shape[0], 1), dtype=np.float32)], axis=1)
        bb_w = (mw @ bb_h.T).T[:, :3]
    except Exception:
        return None

    try:
        pts = _project_world_to_image(bb_w, cam2world)
    except Exception:
        return None

    xs = pts[:, 0]
    ys = pts[:, 1]

    xmin = float(np.min(xs))
    xmax = float(np.max(xs))
    ymin = float(np.min(ys))
    ymax = float(np.max(ys))

    # Require some overlap with image
    if xmax < 0 or ymax < 0 or xmin > w or ymin > h:
        return None

    xmin = _clamp(xmin, 0, w - 1)
    ymin = _clamp(ymin, 0, h - 1)
    xmax = _clamp(xmax, 0, w - 1)
    ymax = _clamp(ymax, 0, h - 1)

    bw = xmax - xmin
    bh = ymax - ymin
    if bw < 2 or bh < 2:
        return None
    return (xmin, ymin, bw, bh)


# -----------------------------
# Main
# -----------------------------

def _build_coco_skeleton(categories: List[str]) -> Dict[str, Any]:
    cats = []
    for i, name in enumerate(categories, start=1):
        cats.append({"id": i, "name": name, "supercategory": "object"})
    coco = {
        "info": {"description": "YCB synthetic (BlenderProc)", "version": "1.0"},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": cats,
    }
    return coco

def _bg_list_for_split(base: BaseCfg, split: str) -> List[Path]:
    if split.lower() == "train":
        pats = base.resources.backgrounds_train_glob
    else:
        pats = base.resources.backgrounds_val_glob
    # support direct glob string
    paths = [Path(p) for p in glob.glob(pats)]
    return sorted([p for p in paths if p.is_file()])

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_config", required=True)
    ap.add_argument("--dr_config", required=False, default="")
    ap.add_argument("--paths_config", required=False, default="")
    ap.add_argument("--split", required=True, choices=["train", "val"])
    ap.add_argument("--num_images", type=int, default=-1)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--seed", type=int, default=-1)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    base = _parse_base_cfg(_read_json(args.base_config))
    dr = _parse_dr_cfg(_read_json(args.dr_config)) if args.dr_config else _parse_dr_cfg({})

    split = args.split
    num_images = int(args.num_images) if args.num_images and args.num_images > 0 else int(base.num_images)
    seed = int(args.seed) if args.seed >= 0 else int(base.seed)

    out_dir = Path(args.output_dir)
    if out_dir.exists():
        if args.overwrite:
            shutil.rmtree(out_dir)
        else:
            raise RuntimeError(f"output_dir exists: {out_dir} (use --overwrite)")
    _mkdir(out_dir)
    img_dir = out_dir / "images"
    _mkdir(img_dir)

    print(f"[INFO] split={split} num_images={num_images} seed={seed}")
    print(f"[INFO] output_dir={out_dir}")
    print(f"[INFO] ycb_root={base.ycb_root}")

    rng = random.Random(seed)

    # Backgrounds
    bgs = _bg_list_for_split(base, split)
    print(f"[INFO] backgrounds: {len(bgs)} (pattern={base.resources.backgrounds_train_glob if split=='train' else base.resources.backgrounds_val_glob})")

    # CC textures sets (for surfaces)
    tex_sets = _scan_texture_sets(base.resources.cctextures_root)
    print(f"[INFO] cc_textures: {len(tex_sets)} sets (root={base.resources.cctextures_root})")

    # Init BlenderProc
    bproc.init()

    _ensure_cycles(base.render.engine, base.render.samples, base.render.use_denoise)
    _setup_camera(base)

    # Load YCB templates once
    ycb_templates = _load_ycb_objects(base.ycb_root, base.classes, base.mesh_preference)
    if not ycb_templates:
        raise RuntimeError("No YCB objects loaded. Check ycb_root / mesh_preference / classes.")

    # COCO
    categories = [c for c in base.classes if c in ycb_templates]
    coco = _build_coco_skeleton(categories)
    cat2id = {c: i + 1 for i, c in enumerate(categories)}
    ann_id = 1

    # Render loop
    for idx in range(num_images):
        frame_seed = seed * 1000003 + idx * 9176
        frng = random.Random(frame_seed)

        mode = _pick_mode(frng, dr)
        geom = _build_mode_geometry(mode, base, tex_sets, frng)
        support_z: float = float(geom["support_z"])
        region = geom["region"]
        focus: Vector = geom["focus"]

        # Lights
        _add_three_point_lights(frng, dr, focus)

        # Choose objects count
        nobj = frng.randint(int(base.min_objects_per_image), int(base.max_objects_per_image))
        cls_list = list(ycb_templates.keys())
        chosen = [cls_list[frng.randrange(0, len(cls_list))] for _ in range(nobj)]

        instances: List["bproc.types.MeshObject"] = []
        for cls in chosen:
            inst = _place_ycb_instance(ycb_templates[cls], cls, frng, region, support_z)
            _set_material_dr_on_object(inst, dr, frng)
            instances.append(inst)

        # Clutter primitives
        _spawn_clutter(dr, frng, region, support_z)

        # Camera pose
        cam2world = _random_camera_pose(frng, base, focus)
        bproc.camera.add_camera_pose(cam2world)

        # Render RGBA
        data = bproc.renderer.render()
        if "colors" not in data:
            raise RuntimeError("Renderer did not return 'colors'.")
        rgba = data["colors"][0]  # (H,W,4) float [0,1] or uint8 depending build
        if rgba.dtype != np.uint8:
            rgba_u8 = np.clip(rgba * 255.0, 0, 255).astype(np.uint8)
        else:
            rgba_u8 = rgba

        # Pick background (if available) and composite
        if bgs:
            bg_path = bgs[frng.randrange(0, len(bgs))]
            bg = _load_bg_image(bg_path)
            bg = _resize_bg(bg, base.image_width, base.image_height)
            rgb = _composite_rgba_on_bg(rgba_u8, bg)
        else:
            rgb = rgba_u8[..., :3].copy()

        # Optional blur
        rgb = _maybe_blur(rgb, frng, base.export.blur_prob, base.export.blur_sigma_min, base.export.blur_sigma_max)

        # Save image
        fn = f"{idx:06d}.{base.export.image_ext}"
        out_img = img_dir / fn
        try:
            _save_rgb_atomic(rgb, out_img, jpg_quality=base.export.jpg_quality)
        except Exception as e:
            # If jpeg failed and fallback png saved, keep going but log.
            print(f"[WARN] image save issue at {out_img}: {e}")

        # COCO image entry
        coco["images"].append({
            "id": idx + 1,
            "file_name": f"images/{fn}",
            "width": int(base.image_width),
            "height": int(base.image_height),
        })

        # COCO annotations (bbox only)
        for inst in instances:
            cls = inst.get_name()
            if cls not in cat2id:
                continue
            bbox = _bbox_from_object(inst, cam2world, base.image_width, base.image_height)
            if bbox is None:
                continue
            x, y, w, h = bbox
            coco["annotations"].append({
                "id": ann_id,
                "image_id": idx + 1,
                "category_id": cat2id[cls],
                "bbox": [x, y, w, h],
                "area": float(w * h),
                "iscrowd": 0,
                "segmentation": [],
            })
            ann_id += 1

        if (idx + 1) % 10 == 0 or idx == num_images - 1:
            print(f"[INFO] rendered {idx+1}/{num_images} (mode={mode})")

        # Important: clear BlenderProc camera pose list each iteration
        # Otherwise poses accumulate and render() may render all poses at once on some versions.
        try:
            bproc.camera.clear_camera_poses()
        except Exception:
            # If not available, re-init camera pose storage by resetting scene frame
            pass

    # Write COCO JSON
    coco_path = out_dir / "coco_annotations.json"
    coco_path.write_text(json.dumps(coco, indent=2), encoding="utf-8")

    meta = {
        "split": split,
        "num_images": num_images,
        "seed": seed,
        "ycb_root": str(base.ycb_root),
        "mesh_preference": base.mesh_preference,
        "classes_requested": base.classes,
        "classes_loaded": list(ycb_templates.keys()),
        "background_count": len(bgs),
        "cc_texture_sets": len(tex_sets),
        "timestamp_ms": _now_ms(),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[OK] done. output_dir={out_dir}")
    print(f"[OK] COCO: {coco_path}")

if __name__ == "__main__":
    main()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scan YCB model root and generate/update configs/synth_config.json with classes that have:
  **/tsdf/textured.obj

Usage (inside repo):
  python3 scripts/ycb_make_classes_tsdf.py --ycb_root /work/models/ycb --out_config /work/configs/synth_config.json

Env:
  LIMIT_CLASSES=0    # 0 = no limit
  SEED=0
  SHUFFLE=0|1
"""

import json
import os
import random
import re
from pathlib import Path
from typing import List, Dict, Any


def scan_tsdf_classes(ycb_root: Path) -> List[str]:
    if not ycb_root.exists():
        raise FileNotFoundError(f"ycb_root not found: {ycb_root}")

    classes: List[str] = []
    # Top-level dirs are object names (e.g., 003_cracker_box)
    for d in sorted([p for p in ycb_root.iterdir() if p.is_dir()]):
        name = d.name

        # YCB objects are often like 001_xxx..., but we accept anything directory-like.
        # We search robustly: /ycb_root/<name>/**/tsdf/textured.obj
        hits = list(d.glob("**/tsdf/textured.obj"))
        if hits:
            classes.append(name)

    # uniq + stable
    seen = set()
    out = []
    for c in classes:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def default_config(ycb_root: str, classes: List[str]) -> Dict[str, Any]:
    # あなたが先に貼ってくれた形式をベースに、必要十分だけ入れておく版
    return {
        "seed": 0,
        "ycb_root": ycb_root,
        # tsdfしか残していないので原則これでOK（poisson等を混ぜたいなら後で足す）
        "mesh_preference": ["tsdf"],
        "classes": classes,
        "image_width": 640,
        "image_height": 480,
        "num_images": 200,
        "min_objects_per_image": 1,
        "max_objects_per_image": 4,
        "camera": {
            "fx": 600.0,
            "fy": 600.0,
            "cx": 320.0,
            "cy": 240.0,
            "loc_min": [-0.6, -0.6, 0.45],
            "loc_max": [0.6, 0.6, 1.10],
            "inplane_rot_min": -0.6,
            "inplane_rot_max": 0.6
        },
        "table": {
            "size": 1.4,
            "height": 0.75
        },
        "resources": {
            "cctextures_root": "/work/resources/cctextures",
            "backgrounds_train_glob": "/work/assets/backgrounds/train/*.*",
            "backgrounds_val_glob": "/work/assets/backgrounds/val/*.*"
        },
        "render": {
            "engine": "CYCLES",
            "samples": 32,
            "use_denoise": True
        },
        "export": {
            "image_ext": "jpg",
            "jpg_quality": 90
        }
    }


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--ycb_root", required=True)
    ap.add_argument("--out_config", required=True)
    args = ap.parse_args()

    ycb_root = Path(args.ycb_root)
    out_config = Path(args.out_config)
    out_config.parent.mkdir(parents=True, exist_ok=True)

    classes = scan_tsdf_classes(ycb_root)

    # optional shuffle/limit
    shuffle = os.environ.get("SHUFFLE", "0").strip() in ("1", "true", "yes", "on")
    seed = int(os.environ.get("SEED", "0"))
    limit = int(os.environ.get("LIMIT_CLASSES", "0"))

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(classes)

    if limit > 0:
        classes = classes[:limit]

    print(f"[INFO] ycb_root={ycb_root}")
    print(f"[INFO] tsdf classes found: {len(classes)}")
    if len(classes) <= 50:
        print("[INFO] classes:", ", ".join(classes))
    else:
        print("[INFO] classes head:", ", ".join(classes[:20]), "...")
        print("[INFO] classes tail:", ", ".join(classes[-10:]))

    # write also a plain list for convenience
    txt_path = out_config.parent / "classes_tsdf.txt"
    txt_path.write_text("\n".join(classes) + "\n", encoding="utf-8")
    print(f"[OK] wrote: {txt_path}")

    # update or create synth_config.json
    if out_config.exists():
        cfg = json.loads(out_config.read_text(encoding="utf-8"))
        cfg["ycb_root"] = str(ycb_root)
        cfg["classes"] = classes
        # mesh_preference が空/未設定なら tsdf を入れる（既にあるなら尊重）
        if not cfg.get("mesh_preference"):
            cfg["mesh_preference"] = ["tsdf"]
        out_config.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[OK] updated: {out_config} (classes overwritten)")
    else:
        cfg = default_config(str(ycb_root), classes)
        out_config.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[OK] created: {out_config}")


if __name__ == "__main__":
    main()


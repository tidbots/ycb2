#!/usr/bin/env python3
from pathlib import Path
import json, os

YCB_ROOT = Path(os.environ.get("YCB_ROOT", "models/ycb"))
OUT_JSON = Path(os.environ.get("OUT_JSON", "configs/classes_tsdf.json"))
OUT_YAML = Path(os.environ.get("OUT_YAML", "configs/classes_tsdf.yaml"))

def find_tsdf_obj(root: Path, obj: str) -> Path | None:
    # 典型: <root>/<obj>/<obj>/tsdf/textured.obj
    # ただし objに - がある/ない両方、フォルダが一段深い/浅い両方を許す
    cands = []
    for name in {obj, obj.replace("-", "_"), obj.replace("_", "-")}:
        cands += [
            root / obj / obj / "tsdf" / "textured.obj",
            root / obj / name / "tsdf" / "textured.obj",
            root / name / name / "tsdf" / "textured.obj",
            root / name / obj / "tsdf" / "textured.obj",
            root / obj / "tsdf" / "textured.obj",
            root / name / "tsdf" / "textured.obj",
        ]
    for p in cands:
        if p.is_file():
            return p
    return None

def main():
    if not YCB_ROOT.is_dir():
        raise SystemExit(f"YCB_ROOT not found: {YCB_ROOT}")

    objs = sorted([p.name for p in YCB_ROOT.iterdir() if p.is_dir() and not p.name.startswith(".")])
    ok, ng = [], []
    for obj in objs:
        p = find_tsdf_obj(YCB_ROOT, obj)
        if p:
            ok.append(obj)
        else:
            ng.append(obj)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_YAML.parent.mkdir(parents=True, exist_ok=True)

    OUT_JSON.write_text(json.dumps({"mesh_preference": ["tsdf"], "classes": ok}, indent=2), encoding="utf-8")
    OUT_YAML.write_text("mesh_preference:\n  - tsdf\nclasses:\n" + "".join([f"  - {c}\n" for c in ok]), encoding="utf-8")

    print(f"[OK] tsdf available: {len(ok)}")
    print(f"[NG] tsdf missing:  {len(ng)}")
    print(f"[OUT] {OUT_JSON}")
    print(f"[OUT] {OUT_YAML}")
    if ng:
        print("[NG LIST]")
        for x in ng[:50]:
            print("  -", x)
        if len(ng) > 50:
            print(f"  ... and {len(ng)-50} more")

if __name__ == "__main__":
    main()


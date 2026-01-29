#!/usr/bin/env python3
import os
import random
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from urllib.request import urlopen

from PIL import Image, ImageFile

# 壊れ気味JPEGでも読み切れる可能性を上げる（それでもダメなものはスキップ）
ImageFile.LOAD_TRUNCATED_IMAGES = True


def env_int(key: str, default: int) -> int:
    v = os.environ.get(key, str(default)).strip()
    return int(v)


def download(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"[INFO] zip already exists: {out_path}")
        return
    print(f"[INFO] downloading: {url}")
    with urlopen(url) as r, open(out_path, "wb") as f:
        shutil.copyfileobj(r, f)
    print(f"[OK] downloaded: {out_path} ({out_path.stat().st_size} bytes)")


def iter_jpgs(root: Path):
    for p in root.rglob("*.jpg"):
        yield p
    for p in root.rglob("*.jpeg"):
        yield p


def safe_reencode_to_jpg(src: Path, dst: Path, quality: int = 92) -> bool:
    try:
        with Image.open(src) as im:
            im = im.convert("RGB")
            dst.parent.mkdir(parents=True, exist_ok=True)
            im.save(dst, format="JPEG", quality=quality, subsampling=0, optimize=True)
        # 0バイト防止
        return dst.exists() and dst.stat().st_size > 0
    except Exception as e:
        print(f"[WARN] skip broken image: {src} ({e})")
        return False


def main() -> int:
    work = Path("/work")
    out_root = work / "assets" / "backgrounds"
    train_dir = out_root / "train"
    val_dir = out_root / "val"
    cache_dir = out_root / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    n_train = env_int("BG_TRAIN", 200)
    n_val = env_int("BG_VAL", 50)
    seed = env_int("BG_SEED", 0)
    url = os.environ.get(
        "BG_URL",
        "https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip",
    )

    # 既に十分あるならスキップ
    existing_train = len(list(train_dir.glob("*.jpg")))
    existing_val = len(list(val_dir.glob("*.jpg")))
    if existing_train >= n_train and existing_val >= n_val:
        print(f"[OK] backgrounds already prepared: train={existing_train} val={existing_val}")
        return 0

    zip_path = cache_dir / "coco128.zip"
    download(url, zip_path)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        print(f"[INFO] extracting: {zip_path}")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(td)

        # coco128.zip の中は coco128/images/train2017/*.jpg 形式が多い
        jpgs = sorted(iter_jpgs(td))
        if not jpgs:
            print("[ERROR] no jpgs found after extract")
            return 2

        rng = random.Random(seed)
        rng.shuffle(jpgs)

        # 必要数だけ（足りなければあるだけ）
        need_train = max(0, n_train - existing_train)
        need_val = max(0, n_val - existing_val)

        # 出力ファイル名は連番（既存の続きから）
        def next_index(d: Path) -> int:
            nums = []
            for p in d.glob("*.jpg"):
                try:
                    nums.append(int(p.stem))
                except Exception:
                    pass
            return (max(nums) + 1) if nums else 0

        ti = next_index(train_dir)
        vi = next_index(val_dir)

        print(f"[INFO] found jpgs: {len(jpgs)}")
        print(f"[INFO] will add: train={need_train} val={need_val}")

        made_train = 0
        made_val = 0

        # まず train
        for p in jpgs:
            if made_train >= need_train:
                break
            dst = train_dir / f"{ti:06d}.jpg"
            if safe_reencode_to_jpg(p, dst):
                made_train += 1
                ti += 1

        # 次に val（残りから）
        for p in jpgs[made_train:]:
            if made_val >= need_val:
                break
            dst = val_dir / f"{vi:06d}.jpg"
            if safe_reencode_to_jpg(p, dst):
                made_val += 1
                vi += 1

        print(f"[OK] prepared: train(+{made_train}) val(+{made_val})")
        print(f"[OK] total now: train={len(list(train_dir.glob('*.jpg')))} val={len(list(val_dir.glob('*.jpg')))}")
        return 0


if __name__ == "__main__":
    sys.exit(main())


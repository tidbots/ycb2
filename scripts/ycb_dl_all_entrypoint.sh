#!/bin/sh
set -eu

OUT="${OUT:-/work/models/ycb}"
OBJJSON_URL="${OBJJSON_URL:-http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/data/objects.json}"
BASE="${BASE:-https://ycb-benchmarks.s3.amazonaws.com/data/berkeley}"

echo "[INFO] OUT=$OUT"
echo "[INFO] OBJJSON_URL=$OBJJSON_URL"
echo "[INFO] BASE=$BASE"

# tools
echo "[INFO] installing tools..."
apk add --no-cache curl ca-certificates jq tar gzip coreutils >/dev/null
update-ca-certificates >/dev/null 2>&1 || true

mkdir -p "$OUT"
cd "$OUT"

# fetch objects.json
echo "[INFO] fetching objects.json: $OBJJSON_URL"
curl -fsSL "$OBJJSON_URL" -o objects.json

# objects list
echo "[INFO] extracting object list..."
# objects.json is like: { "objects": ["001_chips_can", ...] }
if jq -e '.objects | type=="array"' objects.json >/dev/null 2>&1; then
  jq -r '.objects[]' objects.json > objects.txt
else
  echo "[FATAL] objects.json format unexpected (no .objects array)"
  head -n 50 objects.json || true
  exit 1
fi

TOTAL="$(wc -l < objects.txt | tr -d ' ')"
echo "[INFO] objects: $TOTAL"

FAILED="$OUT/failed_objects.txt"
: > "$FAILED"

download_one() {
  obj="$1"
  obj_us="$(echo "$obj" | tr '-' '_')"

  # 取り出し先ディレクトリは「元の名前」を使う（objects.jsonと一致する）
  # ※混乱したくないなら obj_us に統一してもOK。必要なら後で統一案を出します。
  dest_dir="$OUT/$obj"
  if [ -d "$dest_dir" ] && [ -n "$(find "$dest_dir" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null || true)" ]; then
    echo "[SKIP] already extracted: $obj"
    return 0
  fi
  mkdir -p "$dest_dir"

  # 候補URL（obj名とtgz名で -/_ のズレを吸収）
  # 1) folder=obj, tgz=obj
  # 2) folder=obj, tgz=obj_us
  # 3) folder=obj_us, tgz=obj
  # 4) folder=obj_us, tgz=obj_us
  cand_urls="
$BASE/$obj/${obj}_berkeley_meshes.tgz
$BASE/$obj/${obj_us}_berkeley_meshes.tgz
$BASE/$obj_us/${obj}_berkeley_meshes.tgz
$BASE/$obj_us/${obj_us}_berkeley_meshes.tgz
"

  tgz="$OUT/${obj}_berkeley_meshes.tgz"
  tmp="$OUT/.tmp_${obj}.tgz"
  rm -f "$tmp"

  ok=0
  for url in $cand_urls; do
    echo "[DL] try: $url"
    # -f で 404 などを失敗扱い
    if curl -fL --retry 3 --retry-delay 1 -o "$tmp" "$url" >/dev/null 2>&1; then
      # tarとして妥当か検査（HTML等を弾く）
      if tar -tzf "$tmp" >/dev/null 2>&1; then
        mv -f "$tmp" "$tgz"
        ok=1
        echo "[OK] downloaded: $obj"
        break
      else
        echo "[WARN] not a valid tgz (maybe html/error): $url"
        # デバッグ用に残すならコメントアウト解除
        # mv -f "$tmp" "$OUT/${obj}_INVALID.tgz" || true
        rm -f "$tmp"
      fi
    fi
  done

  if [ "$ok" -ne 1 ]; then
    echo "[WARN] download failed: $obj"
    echo "$obj" >> "$FAILED"
    rm -f "$tmp"
    return 0  # 続行
  fi

  # extract (展開はdest_dirに)
  # tgz の中身が「obj/obj/...」みたいに2階層になることがあるので、いったんdest_dirに全部展開
  if tar -xzf "$tgz" -C "$dest_dir" >/dev/null 2>&1; then
    echo "[OK] extracted: $obj"
    return 0
  else
    echo "[WARN] extract failed: $obj -> keep tgz for debug: $tgz"
    echo "$obj" >> "$FAILED"
    return 0
  fi
}

i=0
while IFS= read -r obj; do
  i=$((i+1))
  echo "===== [$i/$TOTAL] $obj ====="
  download_one "$obj"
done < objects.txt

echo "[INFO] done."
if [ -s "$FAILED" ]; then
  echo "[WARN] some objects failed. see: $FAILED"
  echo "[WARN] failed count: $(wc -l < "$FAILED" | tr -d ' ')"
else
  echo "[OK] all objects downloaded/extracted."
fi


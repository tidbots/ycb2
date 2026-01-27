#!/bin/sh
set -eu

OUT="/work/models/ycb"
mkdir -p "$OUT"
cd "$OUT"

for OBJ in 002_master_chef_can 003_cracker_box; do
  TGZ="${OBJ}_berkeley_meshes.tgz"
  URL="https://ycb-benchmarks.s3.amazonaws.com/data/berkeley/${OBJ}/${TGZ}"

  echo "Downloading: $URL"
  curl -L --fail -o "$TGZ" "$URL"

  mkdir -p "$OBJ"
  tar -xzf "$TGZ" -C "$OBJ"
done

echo "DONE: extracted under $OUT/<object>/..."


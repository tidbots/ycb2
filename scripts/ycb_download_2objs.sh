#!/bin/sh
set -eu

OUT="/work/models/ycb"
mkdir -p "$OUT"
cd "$OUT"

for OBJ in 002_master_chef_can 003_cracker_box; do
  echo "=== $OBJ ==="
  aws s3 ls --no-sign-request "s3://ycb-benchmarks/data/berkeley/$OBJ/" || true

  KEY="$(aws s3 ls --no-sign-request "s3://ycb-benchmarks/data/berkeley/$OBJ/" \
        | awk '{print $4}' | grep -E 'berkeley_meshes\.tgz$' | head -n 1 || true)"

  if [ -z "$KEY" ]; then
    echo "ERROR: could not find *berkeley_meshes.tgz for $OBJ"
    exit 1
  fi

  aws s3 cp --no-sign-request \
    "s3://ycb-benchmarks/data/berkeley/$OBJ/$KEY" "./$KEY"

  mkdir -p "$OBJ"
  tar -xzf "./$KEY" -C "$OBJ"
done

echo "DONE: $OUT"


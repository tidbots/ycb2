cd ycb2

# YCBモデルのダウンロード
docker compose run --rm ycb_dl
ls models/ycb　にダウンロードされる

# 背景画像のダウンロード
docker compose run --rm backgrounds

# テクスチャー画像のダウンロード
docker compose run --rm textures

# 合成データ生成（COCO）
docker compose run --rm synth

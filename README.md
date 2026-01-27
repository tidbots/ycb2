# ycb2: YCB 2物体 → 合成COCO → YOLO形式 → YOLO26(ultralytics互換CLI) 学習/推論（Docker Compose）

このリポジトリは、**YCB Object（例: 002_master_chef_can / 003_cracker_box）**の3Dモデルから合成画像を生成し、  
**COCOアノテーション → YOLO形式に変換 → YOLO26sでファインチューニング → 推論確認**までを **Docker Compose だけ**で回すための作業一式です。

## 要件
- Docker / Docker Compose (v2)
- NVIDIA GPU（例: RTX 4090）とドライバ
- NVIDIA Container Toolkit が導入されていて、Composeから `gpus: all` が使えること
- 十分な /dev/shm（BlenderProcが多用するので `shm_size` を大きく推奨）

## ディレクトリ構成（例）
```
├─ docker-compose.yaml
├─ scripts/
│ ├─ ycb_download_2objs.sh
│ ├─ synth_entrypoint.sh
│ ├─ render_ycb2_coco.py
│ └─ coco_to_yolo_split.py
├─ models/
│ └─ ycb/ # YCB 3Dモデル（OBJ/MTL/PNG等）
│ 　　├─  002_master_chef_can
│ 　　├─  003_cracker_box
│ 　　└─  ...
├─ weights
  └─ yolo26s.pt

├─ data/
│ └─ synth_coco/
│   └─ coco_data/
│     ├─ coco_annotations.json
│     └─ images/
│         ├─ 000000.jpg
│         ├─ 000001.jpg
│         ├─ ...

├─ dataset/
│ └─ ycb2_yolo/
│    ├─ data.yaml # convertサービスが生成
│    ├─ images/
│    │ ├─ train/
│    │ └─ val/
│    └─ labels/
│      ├─ train/
│      └─ val/
└─ runs_ycb2/
   ├─ ycb2_2cls/ # train結果（weights/best.pt等）
   └─ pred_synth/ # predict結果
```

## サービス一覧（docker-compose.yaml）
- `ycb_dl`: YCBデータ（OBJ/MTL/テクスチャ等）を取得して `models/ycb/` に展開
- `synth`: BlenderProcで合成データ生成（COCO形式で出力）
- `convert`: `coco_to_yolo_split.py` で COCO → YOLO に変換（train/val分割含む）
- `train`: Ultralytics互換 `yolo` CLI で学習（YOLO26s）
- `predict`: 学習済みモデルで推論し、結果画像を保存

## 環境設定
### 1) synth_entrypoint.sh（合成側：BlenderProcをどう呼ぶか）
- 生成枚数 --n_images
- 解像度 --width --height
- 参照するYCBモデルルート --assets_root
- 出力先 --out_dir
- テクスチャ強制やデバッグオプション（あなたが追加した --force_texture --debug_print_paths など）
- GPU周り（環境変数や Blender の設定をここで固定する場合も多い）

おすすめ方針
- --n_images 等は スクリプトに直書きせず、N_IMAGES, WIDTH, HEIGHT, OUT_DIR みたいな 環境変数で上書き可能にする
→ Compose 側で切替できて運用が楽

### 2) ycb_dl_entrypoint.sh（ダウンロード用コンテナの入口：依存を揃えてからどれを実行するか）
- ycb_download_2objs.sh か ycb_download_2objs_http.sh の どちらを呼ぶか選ぶ
- 入出力パス（/work/models/ycb）を保証する



## 手順（最短パス）
### 0) 最初からやり直す場合
```
rm -rf data/synth_coco
rm -rf dataset/ycb2_yolo
rm -rf runs_ycb2
```
### 1) YCBモデルを取得（2物体）
```bash
docker compose run --rm ycb_dl
```
取得確認（例）:
```
find models/ycb -maxdepth 6 -type f \( -name "*.obj" -o -name "*.mtl" -o -name "*.png" \) | head -n 20
```

### 2) 合成データ生成（COCO）
```
docker compose run --rm synth
```
出力先（例）:
- data/synth_coco/coco_data/coco_annotations.json
- data/synth_coco/coco_data/images/*.jpg（または png）

枚数確認:
```
ls -1 data/synth_coco/coco_data/images/* | wc -l
```
補足: Blender/Cyclesの「Loading render kernels」は初回に数分かかることがあります。


#### 3つのシーンモードに分けて処理
- 1. floor_mode：床＋壁、床置き物体（遠めも作る）
- 2. shelf_mode：棚板＋背板、棚上物体（影が強い）
- 3. table_mode：テーブル天板、机上（背景テクスチャ変化）

各フレームで mode をランダムに選び、(照明・カメラ・clutter・反射）を適用。

#### （YCB2物体→COCO→YOLO→学習）に追加
- (1) 背景テクスチャを導入（assets/textures を増やし、床/壁/棚に貼る）
- (2) 照明を3灯（キー/フィル/逆光）にして色と強さをランダム化
- (3) 低頻度でブレ/ボケを入れる
- (4) clutterを2〜6個入れる（プリミティブでOK）
- (5) roughness/specular をランダム化

#### 背景について
「YCB 3Dモデル＋BlenderProc＋cc_textures(PBR)背景」で、合成側を現地っぽく寄せる。

BlenderProcの cc_textures 導入版を使う。

「自前で床/壁の画像を集めなくても、BlenderProcが用意している“現実っぽいPBR素材（床・木・コンクリ等）”を自動ダウンロードして、背景や材質に貼れるようにする運用」

“CC0 Textures”（現在は Poly Haven 系の無料PBR素材の流れ）みたいな 著作権フリーのテクスチャ素材集を指していて、BlenderProcはそれを blenderproc download cc_textures でまとめて取ってこれます。

しかも PBR（Color/Normal/Roughness/Metallic/Displacement など）が揃うので、ただの画像より 光の当たり方がリアルになります。



### 3) COCO → YOLO 変換（train/val split）
```
docker compose run --rm convert
```
実行ログ例:
- Wrote: /work/dataset/ycb2_yolo/data.yaml
- Train images: 320 Val images: 80
- NOTE: images with empty labels: 1

### 4) YOLO26（ultralytics互換CLI）で学習
```
docker compose run --rm train
```
成功すると、出力は（本READMEの想定）:
```
runs_ycb2/
└─ ycb2_2cls/
   └─ weights/
      ├─ best.pt
      └─ last.pt
```
確認:
```
ls -lh runs_ycb2/ycb2_2cls/weights/
```
注意: runs_ycb2/detect/... ではありません。保存先は project=/work/runs_ycb2 と name=ycb2_2cls の組み合わせに依存します。

### 5) 推論（合成画像に対して sanity check）
```
docker compose run --rm predict
```
確認:
```
ls -la runs_ycb2/pred_synth | head
```

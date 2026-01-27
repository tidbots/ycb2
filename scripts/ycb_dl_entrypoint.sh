#!/bin/sh
set -eux

# 必要ツールを入れる
apk add --no-cache curl tar

# YCB 2物体をHTTP直リンクでDL＆展開
/work/scripts/ycb_download_2objs_http.sh


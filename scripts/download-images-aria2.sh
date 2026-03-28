#!/usr/bin/env zsh
set -euo pipefail

mkdir -p cache/collections/milady-maker

seq 0 9999 | awk -v dir="$PWD/cache/collections/milady-maker" '{
  printf "https://www.miladymaker.net/milady/%d.png\n out=%d.png\n dir=%s\n", $1, $1, dir
}' > cache/collections/milady-maker.aria2.txt

aria2c \
  --input-file=cache/collections/milady-maker.aria2.txt \
  --continue=true \
  --allow-overwrite=false \
  --auto-file-renaming=false \
  --max-concurrent-downloads=32 \
  --split=4 \
  --max-connection-per-server=4 \
  --min-split-size=1M \
  --retry-wait=2 \
  --max-tries=8

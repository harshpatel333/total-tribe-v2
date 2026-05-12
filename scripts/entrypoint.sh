#!/bin/sh
# Container entrypoint for total-tribe-v2.
#
# Seeds /app/cache from the baked /opt/preloaded directory when /app/cache is
# empty — covers two startup paths:
#   1. No volume mount: /app/cache is just an image layer; copy is effectively
#      a no-op (target already populated by the image build).
#   2. Mounted named volume on first start: /app/cache appears empty; copy
#      hydrates it from the image. Subsequent starts skip the copy.
#
# Without this, mounting a Dokploy volume at /app/cache would hide the baked
# weights and force a ~10 min cold-start on every fresh volume.
set -e

PRELOAD_DIR="${PRELOAD_DIR:-/opt/preloaded}"
CACHE_DIR="${TRIBE_CACHE:-/app/cache}"

mkdir -p "$CACHE_DIR"

# `ls -A` lists hidden + visible entries. Empty output → empty dir.
if [ -z "$(ls -A "$CACHE_DIR" 2>/dev/null)" ]; then
    if [ -d "$PRELOAD_DIR" ] && [ -n "$(ls -A "$PRELOAD_DIR" 2>/dev/null)" ]; then
        echo "[entrypoint] Seeding $CACHE_DIR from $PRELOAD_DIR (first start)…"
        cp -a "$PRELOAD_DIR/." "$CACHE_DIR/"
        echo "[entrypoint] Seed complete."
    else
        echo "[entrypoint] $CACHE_DIR empty and no $PRELOAD_DIR to seed from — runtime will re-download."
    fi
else
    echo "[entrypoint] $CACHE_DIR already populated; skipping seed."
fi

exec "$@"

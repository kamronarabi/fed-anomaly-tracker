#!/bin/sh
# Container start: make sure the published-JSON directory exists on the
# persistent volume before the server can try to read from it.
#
# PUBLISH_DIR is the frontend's source of truth (web/lib/data.ts) and the
# pipeline's output dir (export/publish.py). It lives on the volume so
# published refreshes survive restarts -- the image's web/public/data
# does not, which is why the live site spent four weeks serving the
# leaderboard.json that happened to be committed at build time.
set -e

: "${PUBLISH_DIR:=/data/public/data}"
IMAGE_DATA=/app/web/public/data

# First boot on a fresh volume: seed from the copy baked into the image so
# the site renders something rather than 500ing on a missing file. Keyed
# on leaderboard.json specifically -- an empty or partial directory should
# still be seeded, and once real data is published this never runs again.
if [ ! -f "$PUBLISH_DIR/leaderboard.json" ]; then
  echo "entrypoint: seeding $PUBLISH_DIR from $IMAGE_DATA (first boot)"
  mkdir -p "$PUBLISH_DIR"
  cp -R "$IMAGE_DATA/." "$PUBLISH_DIR/"
fi

# The image's copy is now a stale duplicate that Next would still serve as
# static files at /data/*.json -- a second, frozen answer to "what is the
# current leaderboard". Point it at the volume so there's exactly one.
if [ ! -L "$IMAGE_DATA" ]; then
  rm -rf "$IMAGE_DATA"
  ln -s "$PUBLISH_DIR" "$IMAGE_DATA"
fi

exec "$@"

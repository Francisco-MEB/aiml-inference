#!/usr/bin/env bash
# Downloads the pinned Ollama release into bench/gemma4/.tools/ and verifies its sha256.
# No sudo, nothing outside this directory is touched.
set -euo pipefail
BENCH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-$BENCH/configs/TEMPLATE.json}"
read -r VER URL SHA < <(python3 - "$CONFIG" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))["server"]
print(s["ollama_version"], s["native"]["asset_url"], s["native"]["asset_sha256"])
PY
)
DEST="$BENCH/.tools/ollama-$VER"
if [[ -x "$DEST/bin/ollama" && -f "$DEST/.asset_sha256" && "$(cat "$DEST/.asset_sha256")" == "$SHA" ]]; then
  echo "already installed: $DEST/bin/ollama"; exit 0
fi
mkdir -p "$DEST"
TMP="$BENCH/.tools/$(basename "$URL")"
echo "downloading $URL"
curl -fL --retry 3 -o "$TMP" "$URL"
GOT="$(sha256sum "$TMP" | cut -d' ' -f1)"
if [[ "$GOT" != "$SHA" ]]; then
  echo "SHA256 MISMATCH: expected $SHA got $GOT" >&2; exit 1
fi
echo "sha256 ok: $GOT"
tar --zstd -xf "$TMP" -C "$DEST"
echo "$SHA" > "$DEST/.asset_sha256"
rm -f "$TMP"
"$DEST/bin/ollama" --version

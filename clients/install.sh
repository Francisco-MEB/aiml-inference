#!/usr/bin/env bash
# Install (or update) ndcode, the campus coding agent, for the current user.
# No sudo needed. Safe to re-run.
#
#   ./clients/install.sh
#
# What it does:
#   1. downloads a pinned OpenCode release and verifies its SHA-256
#   2. puts it in ~/.local/share/ndcode/opencode-<version>/
#   3. installs the campus config (~/.local/share/ndcode/opencode.json)
#   4. installs the `ndcode` command into ~/.local/bin
set -euo pipefail

# Pinned OpenCode release. To upgrade: pick a release at least 2 weeks old,
# copy the asset's sha256 from its GitHub release page, and bump both values.
OPENCODE_VERSION="v2.2.5"
declare -A OPENCODE_SHA256=(
  [aarch64]="5a528f42caf576c6cdece6a191eb273d3db54effa5518e0402684d8f8026aeb6"   # opencode-linux-arm64.tar.gz
)
declare -A OPENCODE_ASSET=(
  [aarch64]="opencode-linux-arm64.tar.gz"
)

SRC_DIR="$(cd "$(dirname "$0")" && pwd)"
NDCODE_HOME="${NDCODE_HOME:-$HOME/.local/share/ndcode}"
BIN_DIR="${BIN_DIR:-$HOME/.local/bin}"

say() { printf '\033[1m==>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

arch="$(uname -m)"
[[ "$(uname -s)" == Linux ]] || die "only Linux is supported so far."
[[ -n "${OPENCODE_SHA256[$arch]:-}" ]] || die "no pinned OpenCode build for '$arch' yet (supported: ${!OPENCODE_SHA256[*]})."
for cmd in curl tar sha256sum; do command -v "$cmd" >/dev/null || die "'$cmd' is required."; done

dest="$NDCODE_HOME/opencode-$OPENCODE_VERSION"
if [[ -x "$dest/opencode" ]]; then
  say "OpenCode $OPENCODE_VERSION already installed"
else
  asset="${OPENCODE_ASSET[$arch]}"
  url="https://github.com/anomalyco/opencode/releases/download/$OPENCODE_VERSION/$asset"
  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' EXIT
  say "Downloading OpenCode $OPENCODE_VERSION ($asset)"
  curl -fSL --progress-bar -o "$tmp/$asset" "$url"
  say "Verifying checksum"
  echo "${OPENCODE_SHA256[$arch]}  $tmp/$asset" | sha256sum -c --quiet - || die "checksum mismatch; refusing to install."
  mkdir -p "$tmp/x" "$NDCODE_HOME"
  tar -xzf "$tmp/$asset" -C "$tmp/x"
  [[ -x "$tmp/x/opencode" ]] || die "archive didn't contain the opencode binary."
  rm -rf "$dest" && mkdir -p "$dest"
  mv "$tmp/x/opencode" "$dest/opencode"
fi

say "Installing campus config and ndcode command"
ln -sfn "opencode-$OPENCODE_VERSION" "$NDCODE_HOME/current"
install -m 644 "$SRC_DIR/opencode.json" "$NDCODE_HOME/opencode.json"
mkdir -p "$BIN_DIR"
install -m 755 "$SRC_DIR/ndcode" "$BIN_DIR/ndcode"

say "Done: $("$NDCODE_HOME/current/opencode" --version 2>/dev/null | head -1) as 'ndcode'"
case ":$PATH:" in
  *":$BIN_DIR:"*) ;;
  *) echo "   Note: $BIN_DIR isn't on your PATH. Add this to ~/.bashrc:  export PATH=\"$BIN_DIR:\$PATH\"" ;;
esac
cat <<EOF

   Next:  cd into any project and run  ndcode
          (the campus model server must be running; see docs/ndcode.md)
EOF

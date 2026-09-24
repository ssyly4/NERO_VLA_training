#!/usr/bin/env bash
set -euo pipefail

EXPECTED_COMMIT="15a9616a00943ada6c20a0f158e3adb39df2ccac"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PATCH_FILE="$SCRIPT_DIR/nero_openpi_core_15a9616.patch"
OPENPI_ROOT="${1:-/home/dev/workspace/openpi_deploy/repos/openpi}"

git -C "$OPENPI_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1 || {
  echo "[FAIL] OpenPI git worktree not found: $OPENPI_ROOT" >&2
  exit 2
}
[[ -f "$PATCH_FILE" ]] || {
  echo "[FAIL] core patch not found: $PATCH_FILE" >&2
  exit 2
}

actual_commit="$(git -C "$OPENPI_ROOT" rev-parse HEAD)"
if [[ "$actual_commit" != "$EXPECTED_COMMIT" ]]; then
  echo "[FAIL] OpenPI commit mismatch" >&2
  echo "expected=$EXPECTED_COMMIT" >&2
  echo "actual=$actual_commit" >&2
  exit 2
fi

if git -C "$OPENPI_ROOT" apply --reverse --check "$PATCH_FILE" >/dev/null 2>&1; then
  echo "[READY] NERO OpenPI core patch is already applied"
  exit 0
fi

git -C "$OPENPI_ROOT" apply --check "$PATCH_FILE"
git -C "$OPENPI_ROOT" apply "$PATCH_FILE"

echo "[PASS] applied NERO OpenPI core patch"
echo "openpi=$OPENPI_ROOT"
echo "base_commit=$EXPECTED_COMMIT"
echo "patch_sha256=$(sha256sum "$PATCH_FILE" | cut -d' ' -f1)"

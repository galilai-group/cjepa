#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

uv sync --frozen

# VideoSAUR is inference-only here. Keep its source out of the visible repo and
# avoid its obsolete dependency pins by exposing the source through a .pth file.
VIDEOSAUR_DIR="$ROOT/.venv/src/videosaur"
VIDEOSAUR_COMMIT="7a9f85d9388fdeb44eb729ce7ada31569a2e77a2"
if [[ ! -d "$VIDEOSAUR_DIR/.git" ]]; then
  mkdir -p "$(dirname "$VIDEOSAUR_DIR")"
  git clone --filter=blob:none https://github.com/martius-lab/videosaur.git "$VIDEOSAUR_DIR"
fi
git -C "$VIDEOSAUR_DIR" fetch --quiet origin "$VIDEOSAUR_COMMIT"
git -C "$VIDEOSAUR_DIR" checkout --quiet "$VIDEOSAUR_COMMIT"

"$ROOT/.venv/bin/python" - "$VIDEOSAUR_DIR" <<'PY'
import site
import sys
from pathlib import Path

source = Path(sys.argv[1]).resolve()
site_packages = Path(site.getsitepackages()[0])
(site_packages / "videosaur-local.pth").write_text(f"{source}\n")
PY

"$ROOT/.venv/bin/python" - <<'PY'
from importlib.metadata import version

import stable_pretraining
import stable_worldmodel
import videosaur

print("Environment ready:")
print("  stable-worldmodel", version("stable-worldmodel"))
print("  stable-pretraining", version("stable-pretraining"))
print("  VideoSAUR", videosaur.__path__[0])
PY

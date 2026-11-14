#!/usr/bin/env bash
# Install Ollama and vLLM into $HOME. No root anywhere.
#
# Two Ubuntu gaps force this shape, and both would otherwise need apt:
#   python3-venv is absent, so "python3 -m venv" cannot bootstrap pip
#   the zstd CLI is absent, and Ollama now ships .tar.zst
#
# uv solves the first: one static binary in ~/.local/bin that builds a venv
# without ensurepip, and resolves large wheel sets a good deal faster than
# pip, which matters because vLLM pulls several GB. Python's zstandard
# package solves the second, which keeps Ollama on the same version as the
# Windows side. Pinning an older .tgz release would have worked too, but
# then the two Ollama arms would be different builds and the comparison
# would be measuring that rather than the runtime.
set -u
LOG="$HOME/wsl_setup.log"
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

export PATH="$HOME/.local/bin:$PATH"
VENV="$HOME/.venvs/vllm"
OLLAMA_DIR="$HOME/ollama"
ARCHIVE="$HOME/ollama_dl/ollama-linux-amd64.tar.zst"

echo "=== setup started $(date -Is) ==="

if ! command -v uv >/dev/null; then
  echo "--- installing uv ---"
  curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1 || exit 1
fi
uv --version || exit 1

if [ ! -x "$VENV/bin/python" ]; then
  echo "--- creating venv with uv ---"
  uv venv "$VENV" --python 3.12 || exit 1
fi
uv pip install --python "$VENV/bin/python" zstandard --quiet || exit 1
echo "--- zstandard ready ---"

if [ ! -x "$OLLAMA_DIR/bin/ollama" ]; then
  mkdir -p "$(dirname "$ARCHIVE")" "$OLLAMA_DIR"
  echo "--- downloading ollama ---"
  curl -fsSL "https://ollama.com/download/ollama-linux-amd64.tar.zst" -o "$ARCHIVE" || exit 1
  echo "--- extracting $(du -h "$ARCHIVE" | cut -f1) ---"
  "$VENV/bin/python" - "$ARCHIVE" "$OLLAMA_DIR" <<'PY' || exit 1
import sys, tarfile, zstandard
archive, dest = sys.argv[1], sys.argv[2]
with open(archive, "rb") as fh:
    with zstandard.ZstdDecompressor().stream_reader(fh) as stream:
        with tarfile.open(fileobj=stream, mode="r|") as tar:
            tar.extractall(dest, filter="tar")
print("extracted to", dest)
PY
  rm -f "$ARCHIVE"
else
  echo "--- ollama already present ---"
fi
"$OLLAMA_DIR/bin/ollama" --version 2>&1 | head -2

echo "--- installing vllm, the long step ---"
uv pip install --python "$VENV/bin/python" vllm 2>&1 | tail -12

"$VENV/bin/python" -c "import vllm, torch; print('vllm', vllm.__version__); print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available())"
echo "=== setup finished $(date -Is) ==="

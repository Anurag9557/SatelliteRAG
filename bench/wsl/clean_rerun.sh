#!/usr/bin/env bash
# One clean, guarded rerun of every scenario on both backends.
#
# Why this exists. The first full run is not usable for vLLM:
#   1. Phase C was started four times. All four copies drove one engine at
#      once (engine log: "Running: 4 reqs" through the sequential latency
#      test, 16 running + 16 waiting during throughput, 1,643 POSTs for what
#      should be ~410), and each copy wrote over the others' result files.
#   2. vLLM loaded the model's generation_config.json and applied
#      repetition_penalty=1.1; Ollama did not get the same setting.
#   3. Prompts replayed scenario after scenario on one long-lived vLLM
#      engine, so later scenarios read from a warm prefix cache.
#
# What this does about it:
#   - flock: a second copy of this script refuses to start.
#   - bench/run.py now refuses to time against a vLLM engine that already
#     has requests in flight, and holds its own lock.
#   - --generation-config vllm on the server, and repetition_penalty=1.0
#     sent explicitly by both backends on every request.
#   - A fresh vLLM engine for each scenario, so no scenario inherits
#     another's prefix cache. The hit rate inside each one is recorded.
#   - The old results are archived, not deleted.
#
# Run from WSL:  bash /mnt/c/Users/anura/Desktop/SatelliteRAG/bench/wsl/clean_rerun.sh
# Takes roughly 75 minutes. Log: ~/clean_rerun.log
set -u

exec 9>/tmp/satrag_clean_rerun.lock
if ! flock -n 9; then
  echo "clean_rerun.sh is already running. Not starting a second copy."
  exit 1
fi

REPO=/mnt/c/Users/anura/Desktop/SatelliteRAG
LOG="$HOME/clean_rerun.log"
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1
cd "$REPO" || exit 1

VENV="$HOME/.venvs/vllm"
PY="$VENV/bin/python"
MODEL="Qwen/Qwen2.5-1.5B-Instruct"
export PATH="$VENV/bin:$HOME/ollama/bin:$HOME/.local/bin:$PATH"
export CUDA_HOME="$VENV/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH"
export OLLAMA_MODELS=/mnt/c/Users/anura/.ollama/models

say() { echo "=== $* [$(date +%H:%M:%S)]"; }

stop_all() {
  pkill -f "vllm serve" 2>/dev/null
  pkill -f "ollama serve" 2>/dev/null
  for i in $(seq 1 20); do
    pgrep -f "vllm serve|ollama serve" >/dev/null || break
    sleep 2
  done
  sleep 3
}

port_answers() { curl -fsS -m 3 "$1" >/dev/null 2>&1; }

say "clean rerun started"
stop_all

# WSL forwards localhost. If something still answers here after the Linux
# servers are gone, it is the Windows-side Ollama (or another vLLM), and
# timing against it would measure the wrong process.
if port_answers http://127.0.0.1:11434/api/version; then
  echo "ABORT: something on Windows is still serving port 11434 (Windows Ollama?)."
  echo "Quit Ollama from the Windows tray, then run this again."
  exit 1
fi
if port_answers http://127.0.0.1:8000/v1/models; then
  echo "ABORT: something is still serving port 8000."
  exit 1
fi
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

# ------------------------------------------------------------ archive
STAMP=$(date +%Y%m%d-%H%M)
ARCH="bench/archive/run1-contaminated-$STAMP"
mkdir -p "$ARCH/results" "$ARCH/results-wsl-ollama-parallel"
mv bench/results/*.json bench/results/*.bak bench/results/comparison.* \
   "$ARCH/results/" 2>/dev/null
mv bench/results-wsl-ollama-parallel/*.json \
   "$ARCH/results-wsl-ollama-parallel/" 2>/dev/null
cat > "$ARCH/WHY.txt" <<'TXT'
Archived by bench/wsl/clean_rerun.sh. The vLLM latency, throughput,
determinism and quality files here were produced while four copies of
phase_c.sh shared one vLLM engine, and vLLM applied repetition_penalty=1.1
from generation_config.json while Ollama did not. Do not quote the vLLM
numbers from this directory. The Ollama files and vllm structured_schema
were single-client, but were also run without pinned sampling.
TXT
echo "archived first run to $ARCH"

# ------------------------------------------------------------ ollama
start_ollama() {
  local par="$1" logname="$2"
  OLLAMA_NUM_PARALLEL="$par" OLLAMA_MAX_LOADED_MODELS=1 \
    nohup ollama serve > "$HOME/$logname" 2>&1 &
  for i in $(seq 1 30); do
    port_answers http://127.0.0.1:11434/api/version && { echo "ollama up (NUM_PARALLEL=$par)"; return 0; }
    sleep 2
  done
  echo "ollama did not come up"; tail -20 "$HOME/$logname"; return 1
}

say "ollama, NUM_PARALLEL=1, all scenarios"
start_ollama 1 ol_clean.log || exit 1
"$PY" bench/run.py all ollama 2>&1 | grep -vE "^\s+[0-9]+/[0-9]+ " || true
stop_all

say "ollama, NUM_PARALLEL=4, throughput only"
start_ollama 4 ol_clean_par.log || exit 1
mkdir -p bench/results-wsl-ollama-parallel
BENCH_RESULT_DIR="$REPO/bench/results-wsl-ollama-parallel" \
  "$PY" bench/run.py run ollama throughput 2>&1 | tail -2
stop_all

# ------------------------------------------------------------ vllm
start_vllm() {
  local tag="$1"
  nohup "$VENV/bin/vllm" serve "$MODEL" \
    --dtype float16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.70 \
    --max-num-seqs 16 \
    --generation-config vllm \
    --port 8000 > "$HOME/vl_clean_$tag.log" 2>&1 &
  local pid=$!
  for i in $(seq 1 120); do
    if port_answers http://127.0.0.1:8000/v1/models; then
      echo "vllm up for $tag after $((i*4))s"
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "vllm exited during startup:"; tail -25 "$HOME/vl_clean_$tag.log"
      return 1
    fi
    sleep 4
  done
  echo "vllm did not come up for $tag"; tail -25 "$HOME/vl_clean_$tag.log"
  return 1
}

for S in latency throughput determinism quality; do
  say "vllm $S (fresh engine)"
  start_vllm "$S" || { stop_all; exit 1; }
  "$PY" bench/run.py run vllm "$S" 2>&1 | grep -vE "^\s+[0-9]+/[0-9]+ " || true
  stop_all
done

say "vllm structured schema (fresh engine)"
start_vllm structured || { stop_all; exit 1; }
"$PY" bench/run.py run vllm structured --arm schema 2>&1 | tail -2
stop_all

# ------------------------------------------------------------ check + report
say "sanity: engine never saw more than this script's own concurrency"
for f in "$HOME"/vl_clean_*.log; do
  max_run=$(grep -oE "Running: [0-9]+ reqs" "$f" | awk '{print $2}' | sort -n | tail -1)
  posts=$(grep -c "POST /v1/chat/completions" "$f")
  echo "  $(basename "$f"): max running=${max_run:-0}, POSTs=$posts"
done

say "report"
"$PY" bench/report.py
say "clean rerun finished"

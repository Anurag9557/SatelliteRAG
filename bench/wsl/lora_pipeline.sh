#!/usr/bin/env bash
# Everything after the clean benchmark, in stages that can be run one at a
# time or all together:
#
#   deps         peft, datasets, accelerate into the vLLM venv, pinned so
#                nothing vLLM already has can move
#   determinism  vLLM determinism again with prefix caching off, to test
#                whether the 8/10 came from the cache
#   dataset      lora/build_dataset.py with the schema fix, on Ollama 7B
#   train        lora/train.py on that dataset (Qwen2.5-0.5B, r=16)
#   serve        lora/serve_compare.py: PEFT reference, then vLLM with the
#                adapter registered, then the report
#
#   bash bench/wsl/lora_pipeline.sh                 all stages
#   bash bench/wsl/lora_pipeline.sh train serve     just those
#
# DATASET_LIMIT caps the chunks synthesised (default 400, about half an
# hour on 7B). 0 means all of them, which takes a few hours on this card.
set -u

exec 9>/tmp/satrag_lora_pipeline.lock
if ! flock -n 9; then
  echo "lora_pipeline.sh is already running. Not starting a second copy."
  exit 1
fi

REPO=/mnt/c/Users/anura/Desktop/SatelliteRAG
LOG="$HOME/lora_pipeline.log"
exec > >(tee -a "$LOG") 2>&1
cd "$REPO" || exit 1

VENV="$HOME/.venvs/vllm"
PY="$VENV/bin/python"
export PATH="$VENV/bin:$HOME/ollama/bin:$HOME/.local/bin:$PATH"
export CUDA_HOME="$VENV/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH"
export OLLAMA_MODELS=/mnt/c/Users/anura/.ollama/models
DATASET_LIMIT="${DATASET_LIMIT:-400}"
ADAPTER_DIR="$REPO/lora/adapters/adityal1-qwen2.5-0.5b"

STAGES=("$@")
[ ${#STAGES[@]} -eq 0 ] && STAGES=(deps determinism dataset train serve)

say() { echo "=== $* [$(date +%H:%M:%S)]"; }
port_answers() { curl -fsS -m 3 "$1" >/dev/null 2>&1; }

stop_all() {
  pkill -f "vllm serve" 2>/dev/null
  pkill -f "ollama serve" 2>/dev/null
  for i in $(seq 1 20); do
    pgrep -f "vllm serve|ollama serve" >/dev/null || break
    sleep 2
  done
  sleep 3
}

start_vllm() {   # tag, then extra args
  local tag="$1"; shift
  nohup "$VENV/bin/vllm" serve "$@" --generation-config vllm --port 8000 \
    > "$HOME/vl_pipe_$tag.log" 2>&1 &
  local pid=$! t0=$(date +%s)
  for i in $(seq 1 150); do
    if port_answers http://127.0.0.1:8000/v1/models; then
      echo "vllm up for $tag after $(( $(date +%s) - t0 ))s"
      return 0
    fi
    kill -0 "$pid" 2>/dev/null || { echo "vllm exited:"; tail -30 "$HOME/vl_pipe_$tag.log"; return 1; }
    sleep 4
  done
  echo "vllm did not come up"; tail -30 "$HOME/vl_pipe_$tag.log"; return 1
}

stage_deps() {
  say "deps"
  if "$PY" -c "import peft, datasets, accelerate" 2>/dev/null; then
    "$PY" -c "import peft, datasets, accelerate; print('present:', peft.__version__, datasets.__version__, accelerate.__version__)"
    return 0
  fi
  # Freeze what is there as constraints, so the resolver can add packages
  # but cannot upgrade or downgrade torch, transformers or anything vLLM
  # pinned. If the new packages cannot fit around them, this fails rather
  # than quietly breaking the vLLM install.
  local cons=/tmp/vllm_constraints.txt
  if command -v uv >/dev/null; then
    uv pip freeze --python "$PY" | grep "==" > "$cons"
    uv pip install --python "$PY" -c "$cons" peft datasets accelerate || return 1
  else
    "$PY" -m pip freeze | grep "==" > "$cons"
    "$PY" -m pip install -c "$cons" peft datasets accelerate || return 1
  fi
  "$PY" -c "import vllm, peft, datasets, accelerate, torch; print('ok: vllm', vllm.__version__, 'peft', peft.__version__, 'torch', torch.__version__)"
}

stage_determinism() {
  say "determinism, vLLM, prefix caching OFF"
  stop_all
  start_vllm det Qwen/Qwen2.5-1.5B-Instruct --dtype float16 --max-model-len 8192 \
    --gpu-memory-utilization 0.70 --max-num-seqs 16 --no-enable-prefix-caching || return 1
  mkdir -p bench/results-determinism-nocache
  BENCH_RESULT_DIR="$REPO/bench/results-determinism-nocache" \
    "$PY" bench/run.py run vllm determinism 2>&1 | tail -14
  stop_all
}

stage_dataset() {
  say "dataset, Ollama qwen2.5:7b-instruct, limit=$DATASET_LIMIT"
  stop_all
  OLLAMA_NUM_PARALLEL=1 OLLAMA_MAX_LOADED_MODELS=1 nohup ollama serve > "$HOME/ol_pipe.log" 2>&1 &
  for i in $(seq 1 30); do port_answers http://127.0.0.1:11434/api/version && break; sleep 2; done
  SATRAG_LLM=ollama "$PY" lora/build_dataset.py --limit "$DATASET_LIMIT" || { stop_all; return 1; }
  cat lora/data/build_meta.json
  stop_all
}

stage_train() {
  say "train LoRA adapter"
  stop_all
  "$PY" lora/train.py 2>&1 | grep -vE "it/s\]|s/it\]" | tail -25
  [ -f "$ADAPTER_DIR/adapter_config.json" ] || { echo "no adapter written"; return 1; }
  ls -la "$ADAPTER_DIR"
}

stage_serve() {
  say "serve: transformers + PEFT reference"
  stop_all
  "$PY" lora/serve_compare.py reference 2>&1 | tail -6 || return 1
  say "serve: vLLM with the adapter registered"
  local base
  base=$("$PY" -c "import json;print(json.load(open('lora/results.json'))['base_model'])")
  start_vllm lora "$base" --dtype bfloat16 --max-model-len 2048 \
    --gpu-memory-utilization 0.60 --max-num-seqs 16 \
    --enable-lora --max-lora-rank 16 --max-loras 2 \
    --lora-modules "adityal1=$ADAPTER_DIR" || return 1
  "$PY" lora/serve_compare.py vllm 2>&1 | tail -4
  stop_all
  say "serve: report"
  "$PY" lora/serve_compare.py report
}

say "lora pipeline: ${STAGES[*]}"
for s in "${STAGES[@]}"; do
  if ! "stage_$s"; then
    say "stage $s FAILED; later stages not run"
    stop_all
    exit 1
  fi
done
say "lora pipeline finished"

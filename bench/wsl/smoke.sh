#!/usr/bin/env bash
# Five-minute check that the patched harness talks to both servers.
# Writes to /tmp, never to bench/results.
set -u
cd /mnt/c/Users/anura/Desktop/SatelliteRAG || exit 1
VENV="$HOME/.venvs/vllm"; PY="$VENV/bin/python"
export PATH="$VENV/bin:$HOME/ollama/bin:$HOME/.local/bin:$PATH"
export CUDA_HOME="$VENV/lib/python3.12/site-packages/nvidia/cu13"
export OLLAMA_MODELS=/mnt/c/Users/anura/.ollama/models
export BENCH_RESULT_DIR=/tmp/satrag_smoke
rm -rf /tmp/satrag_smoke
pkill -f "vllm serve"; pkill -f "ollama serve"; sleep 4
OLLAMA_NUM_PARALLEL=1 nohup ollama serve > /tmp/ol_smoke.log 2>&1 &
for i in $(seq 1 30); do curl -fsS -m 3 http://127.0.0.1:11434/api/version >/dev/null 2>&1 && break; sleep 2; done
"$PY" bench/run.py run ollama latency -n 2 2>&1 | tail -3
"$PY" bench/run.py run ollama structured --arm builder --structured-chunks 4 2>&1 | tail -2
pkill -f "ollama serve"; sleep 5
nohup "$VENV/bin/vllm" serve Qwen/Qwen2.5-1.5B-Instruct --dtype float16 --max-model-len 8192 \
  --gpu-memory-utilization 0.70 --max-num-seqs 16 --generation-config vllm --port 8000 > /tmp/vl_smoke.log 2>&1 &
for i in $(seq 1 90); do curl -fsS -m 3 http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && { echo "vllm up ${i}"; break; }; sleep 4; done
"$PY" bench/run.py run vllm latency -n 2 2>&1 | tail -3
grep -iE "generation_config|overridden" /tmp/vl_smoke.log | cut -c1-200
curl -s http://127.0.0.1:8000/metrics | grep -E "^vllm:(num_requests_running|num_requests_waiting|prefix_cache)" | head
pkill -f "vllm serve"
"$PY" - <<'PY'
import json,glob
for f in sorted(glob.glob('/tmp/satrag_smoke/*.json')):
    d=json.load(open(f)); r=d['result']
    print(f.split('/')[-1], d['sampling'], {k:r.get(k) for k in ('s_per_100_tok','prefix_cache_hit_rate','paths','chunks_dropped','server_metrics_seen') if k in r})
PY

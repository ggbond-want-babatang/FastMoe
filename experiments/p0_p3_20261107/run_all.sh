#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
PY="${MOE_PYTHON:-/home/ggbond/miniconda3/envs/nanovllm/bin/python}"
export TRITON_CACHE_DIR="$PWD/.triton-cache" CUDA_CACHE_PATH="$PWD/.cuda-cache"
export HF_HOME="$PWD/.hf-cache" TOKENIZERS_PARALLELISM=false
mkdir -p results/p0 results/p3
command -v nsys >/dev/null || { echo 'Nsight Systems is required for the native capture'; exit 1; }
run() {
  local name="$1"
  shift
  printf '%q ' "$@" >>results/commands.log
  printf '> results/%s.log 2>&1\n' "$name" >>results/commands.log
  set +e
  "$@" >"results/$name.log" 2>&1
  local code=$?
  set -e
  printf '%s\n' "$code" >"results/$name.exit"
  printf 'exit_code=%d\n' "$code" >>results/commands.log
  if [ "$code" -ne 0 ]; then tail -n 50 "results/$name.log"; exit "$code"; fi
}
run environment "$PY" environment.py
run p3/download "$PY" download_layer.py
run v1-regression "$PY" -m tests.test_moe
run p3/adapter-tests "$PY" -m tests.test_hf_adapter
run p0/profile "$PY" profile_moe.py --mode torch --iterations 5
run p0/nsys nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=true \
  --output=results/p0/nsys "$PY" profile_moe.py --mode nvtx --iterations 5
run p0/nsys-stats nsys stats --force-export=true --report cuda_api_sum,cuda_gpu_kern_sum,nvtx_sum --format csv results/p0/nsys.nsys-rep
run p0/analysis "$PY" analyze_traces.py
run p3/integration-bf16 "$PY" integrate_olmoe.py --dtype bf16
run p3/integration-fp16 "$PY" integrate_olmoe.py --dtype fp16 --no-trace
run reports "$PY" summarize_p0p3.py
echo 'Completed P0 and P3; see P0_REPORT.md and P3_REPORT.md.'

#!/usr/bin/env bash
# Continue after already-completed downloads and regressions, preserving their logs.
set -u
cd -- "$(dirname -- "$0")"
export TRITON_CACHE_DIR="$PWD/.triton-cache" CUDA_CACHE_PATH="$PWD/.cuda-cache"
export HF_HOME="$PWD/.hf-cache" TOKENIZERS_PARALLELISM=false
PY=/home/ggbond/miniconda3/envs/nanovllm/bin/python
mkdir -p results/p0 results/p3
run() {
  local name="$1"
  shift
  printf '%q ' "$@" >> results/commands.log
  printf '> results/%s.log 2>&1\n' "$name" >> results/commands.log
  "$@" >"results/$name.log" 2>&1
  local code=$?
  printf '%s\n' "$code" >"results/$name.exit"
  printf 'exit_code=%d\n' "$code" >>results/commands.log
  return "$code"
}
test "$(cat results/v1-regression.exit)" = 0 || exit 1
test "$(cat results/p3/adapter-tests.exit)" = 0 || exit 1
test "$(cat results/p3/download.exit)" = 0 || exit 1
run environment "$PY" environment.py
run p0/profile "$PY" profile_moe.py --mode torch --iterations 5
run p0/nsys nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=true \
  --output=results/p0/nsys "$PY" profile_moe.py --mode nvtx --iterations 5
if [ "$(cat results/p0/nsys.exit)" = 0 ]; then
  run p0/nsys-stats nsys stats --report cuda_api_sum,cuda_gpu_kern_sum,nvtx_sum --format csv results/p0/nsys.nsys-rep
fi
run p0/analysis "$PY" analyze_traces.py
run p3/integration-bf16 "$PY" integrate_olmoe.py --dtype bf16
run p3/integration-fp16 "$PY" integrate_olmoe.py --dtype fp16 --no-trace
printf 'Completed experiments; inspect each exit code.\n'

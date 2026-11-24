#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
mkdir -p results
export TRITON_CACHE_DIR="$PWD/.triton-cache"
export CUDA_CACHE_PATH="$PWD/.cuda-cache"
PY="${MOE_PYTHON:-/home/ggbond/miniconda3/envs/nanovllm/bin/python}"
exec 3>>results/commands.log
printf '\nUTC run: %s\n' "$(date -u +%FT%TZ)" >&3
run() {
  local log="$1"
  shift
  printf '%q ' "$@" >&3
  printf '> %q 2>&1\n' "$log" >&3
  set +e
  "$@" >"$log" 2>&1
  local code=$?
  set -e
  printf 'exit_code=%d\n' "$code" >&3
  if [ "$code" -ne 0 ]; then
    tail -n 60 "$log"
    exit "$code"
  fi
}
run results/environment.log "$PY" environment.py
run results/tests.log "$PY" -m tests.test_moe
run results/benchmark.log "$PY" benchmark.py --profile full --tune
run results/kernel-benchmark.log "$PY" kernel_benchmark.py
run results/plot.log "$PY" plot_results.py
run results/summary.log "$PY" summarize.py
printf 'All steps passed. See results/ and RESULTS.md.\n'

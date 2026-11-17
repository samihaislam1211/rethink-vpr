#!/usr/bin/env bash
# Light detached scheduler driver; computation occurs only in sbatch jobs.
set -euo pipefail
if [[ $# -lt 1 || $# -gt 2 ]]; then echo 'Usage: bash scripts/run_pipeline.sh CONFIG.json [smoke|full]' >&2; exit 2; fi
config=$(realpath "$1")
mode=${2:-smoke}
case "$mode" in smoke|full) ;; *) echo 'Mode must be smoke or full' >&2; exit 2;; esac
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python=${BOOTSTRAP_PYTHON:-python3}
project=("$python" -B "$repo/run.py" --config "$config")
"${project[@]}" check
wait_job() {
    local job=$1 record
    echo "Waiting for Slurm job $job" >&2
    while [[ -n $(squeue -h -j "$job" -o '%i') ]]; do sleep 15; done
    record=$(sacct -n -P -j "$job" --format=JobIDRaw,State,ExitCode)
    "$python" -c 'import sys; rows=[x.split("|") for x in sys.stdin.read().splitlines() if x.strip() and "." not in x.split("|")[0]]; assert rows and all(r[1]=="COMPLETED" and r[2]=="0:0" for r in rows), "Job failed or accounting unavailable: "+repr(rows)' <<< "$record"
}
submit_wait() {
    local job
    job=$("${project[@]}" submit "$@")
    echo "Submitted $*: $job" >&2
    wait_job "${job%%;*}"
}
submit_wait prepare-data
submit_wait audit-models
submit_wait baseline-smoke
submit_wait genwarp-smoke
if [[ "$mode" == smoke ]]; then echo 'Portable smoke workflow completed. Full generation has not been submitted.'; exit 0; fi
submit_wait baseline --array 0-15
submit_wait database-prepare
submit_wait database-pilot-generate
submit_wait database-pilot-features
submit_wait database-pilot-evaluate
submit_wait database-generate --array 0-5 --shards 6
submit_wait database-features --array 0-4
submit_wait database-evaluate --array 0-4
submit_wait query-prepare
submit_wait query-pilot-generate
submit_wait query-pilot-features
submit_wait query-generate --array 0-5 --shards 6
submit_wait query-null --array 0-5 --shards 6
submit_wait query-features --array 0-5 --shards 6
submit_wait query-merge --shards 6
submit_wait query-evaluate
submit_wait robust-prepare
submit_wait robust-features --array 0-17
submit_wait robust-merge --array 0-2
submit_wait robust-evaluate --array 0-7
submit_wait gate-signals
submit_wait gating --array 0-7
submit_wait statistics
submit_wait plots
echo 'Workflow completed. See work_dir/SUMMARY.md and QUERY_RESULTS.csv.'

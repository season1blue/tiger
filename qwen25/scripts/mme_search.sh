#!/bin/bash

set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

# Usage:
#   bash qwen25/scripts/mme_search.sh
#   bash qwen25/scripts/mme_search.sh memvr
#   bash qwen25/scripts/mme_search.sh memvr 1
#   bash qwen25/scripts/mme_search.sh memvr 2 "0,1" --nohup
#   bash qwen25/scripts/mme_search.sh memvr 4 "4,5,6,7" --nohup
MODE="${1:-memvr}"
NUM_GPUS="${2:-1}"
GPU_IDS_CSV="${3:-}"
RESUME_RUN_ID=""
DETACH_NOHUP=0

for arg in "${@:4}"; do
    case "$arg" in
        --nohup|nohup)
            DETACH_NOHUP=1
            ;;
        latest)
            RESUME_RUN_ID="latest"
            ;;
        resume=*)
            RESUME_RUN_ID="${arg#resume=}"
            ;;
        *)
            if [[ -z "$RESUME_RUN_ID" ]]; then
                RESUME_RUN_ID="$arg"
            else
                echo "Unknown extra arg: $arg"
                echo "Usage: bash qwen25/scripts/mme_search.sh [memvr|none] [num_gpus] [gpu_ids_csv] [--nohup]"
                exit 1
            fi
            ;;
    esac
done

if [[ "$MODE" != "memvr" && "$MODE" != "none" ]]; then
    echo "Invalid mode: $MODE"
    echo "Usage: bash qwen25/scripts/mme_search.sh [memvr|none] [num_gpus] [gpu_ids_csv] [--nohup]"
    exit 1
fi

if ! [[ "$NUM_GPUS" =~ ^[0-9]+$ ]] || [[ "$NUM_GPUS" -lt 1 ]]; then
    echo "Invalid num_gpus: $NUM_GPUS"
    exit 1
fi

SEARCH_ROOT="$ROOT_DIR/results/Qwen2.5-VL/mme_search"
MME_TMP_ROOT="$ROOT_DIR/results/Qwen2.5-VL/mme"
mkdir -p "$SEARCH_ROOT"

if [[ "$DETACH_NOHUP" -eq 1 && "${MME_SEARCH_NOHUP_LAUNCHED:-0}" != "1" ]]; then
    nohup_log="$SEARCH_ROOT/nohup_$(date +%Y%m%d_%H%M%S).log"
    relaunch_cmd=(bash qwen25/scripts/mme_search.sh "$MODE" "$NUM_GPUS")
    if [[ -n "$GPU_IDS_CSV" ]]; then
        relaunch_cmd+=("$GPU_IDS_CSV")
    fi
    if [[ -n "$RESUME_RUN_ID" ]]; then
        relaunch_cmd+=("$RESUME_RUN_ID")
    fi

    MME_SEARCH_NOHUP_LAUNCHED=1 nohup "${relaunch_cmd[@]}" > "$nohup_log" 2>&1 &
    echo "[SEARCH] detached with nohup"
    echo "[SEARCH] pid=$!"
    echo "[SEARCH] nohup_log=$nohup_log"
    exit 0
fi

RUN_ID="shared_${MODE}"
RUN_DIR="$SEARCH_ROOT/$MODE"
if [[ -n "$RESUME_RUN_ID" ]]; then
    echo "[SEARCH] resume arg ignored in shared-log mode: $RESUME_RUN_ID"
fi

LOG_DIR="$RUN_DIR"
SUMMARY_FILE="$RUN_DIR/summary.tsv"
COMBINED_LOG="$RUN_DIR/combined.log"

mkdir -p "$LOG_DIR"

if [[ ! -f "$SUMMARY_FILE" ]]; then
    {
        echo -e "entropy_threshold\tstarting_layer\tending_layer\tstatus\tkey_result\tlog_file"
    } > "$SUMMARY_FILE"
fi

echo "[SEARCH] mode=$MODE"
echo "[SEARCH] num_gpus=$NUM_GPUS"
echo "[SEARCH] gpu_ids_csv=${GPU_IDS_CSV:-auto}"
echo "[SEARCH] summary_file=$SUMMARY_FILE"
echo "[SEARCH] combined_log=$COMBINED_LOG"

# Keep mme_search outputs, but clear mme intermediate/eval outputs before the sweep.
rm -rf "$MME_TMP_ROOT"

{
    echo "[SEARCH] start_time=$(date '+%F %T')"
    echo "[SEARCH] mode=$MODE num_gpus=$NUM_GPUS gpu_ids_csv=${GPU_IDS_CSV:-auto}"
    echo "[SEARCH] run_id=$RUN_ID resume=${RESUME_RUN_ID:-none}"
    echo "[SEARCH] summary_file=$SUMMARY_FILE"
    echo ""
} >> "$COMBINED_LOG"

total_runs=0
skipped_runs=0
ok_runs=0
fail_runs=0

for entropy in $(awk 'BEGIN { for (v=0.35; v<=0.95+1e-9; v+=0.05) printf "%.2f\n", v }'); do
    for starting_layer in $(seq 2 10); do
        for ending_layer in $(seq 16 28); do
            total_runs=$((total_runs + 1))
            run_tag="et${entropy}_s${starting_layer}_e${ending_layer}"
            log_file="$LOG_DIR/${run_tag}.log"
            legacy_log_file="$RUN_DIR/logs/${run_tag}.log"

            # Resume based on log files: skip if current or legacy log exists and is non-empty.
            if [[ -s "$log_file" || -s "$legacy_log_file" ]]; then
                skipped_runs=$((skipped_runs + 1))
                if [[ -s "$legacy_log_file" && ! -s "$log_file" ]]; then
                    echo "[SEARCH] ($total_runs) skip by legacy log entropy=$entropy start=$starting_layer end=$ending_layer"
                else
                    echo "[SEARCH] ($total_runs) skip by log entropy=$entropy start=$starting_layer end=$ending_layer"
                fi
                continue
            fi

            echo "[SEARCH] ($total_runs) entropy=$entropy start=$starting_layer end=$ending_layer"

            {
                echo "============================================================"
                echo "[RUN] idx=$total_runs entropy=$entropy starting_layer=$starting_layer ending_layer=$ending_layer"
                echo "[RUN] begin=$(date '+%F %T')"
            } >> "$COMBINED_LOG"

            cmd=(bash qwen25/scripts/mme.sh "$MODE" "$NUM_GPUS")
            if [[ -n "$GPU_IDS_CSV" ]]; then
                cmd+=("$GPU_IDS_CSV")
            fi

            if ENTROPY_THRESHOLD="$entropy" \
               STARTING_LAYER="$starting_layer" \
               ENDING_LAYER="$ending_layer" \
               MME_RUN_TAG="$run_tag" \
               "${cmd[@]}" > "$log_file" 2>&1; then
                status="ok"
                ok_runs=$((ok_runs + 1))
            else
                status="fail"
                fail_runs=$((fail_runs + 1))
            fi

            {
                cat "$log_file"
                echo "[RUN] status=$status"
                echo "[RUN] end=$(date '+%F %T')"
                echo ""
            } >> "$COMBINED_LOG"

            key_result="$(grep -E "total|Total|score|Score|Perception|Cognition" "$log_file" | tail -n 1 || true)"
            if [[ -z "$key_result" ]]; then
                key_result="$(tail -n 1 "$log_file" | tr '\t' ' ' || true)"
            fi
            key_result="${key_result//$'\t'/ }"

            echo -e "${entropy}\t${starting_layer}\t${ending_layer}\t${status}\t${key_result}\t${log_file}" >> "$SUMMARY_FILE"

            # Delete temporary mme outputs generated by mme.sh for this run.
            rm -rf "$MME_TMP_ROOT"
        done
    done
done

echo "[SEARCH] done"
echo "[SEARCH] total_runs=$total_runs ok_runs=$ok_runs fail_runs=$fail_runs skipped_runs=$skipped_runs"
echo "[SEARCH] summary_file=$SUMMARY_FILE"
echo "[SEARCH] combined_log=$COMBINED_LOG"

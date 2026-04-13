#!/bin/bash

set -u -o pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd "$script_dir/../.." && pwd)"
cd "$root_dir"

# Usage:
#   bash qwen25/scripts/mme_layer.sh
#   bash qwen25/scripts/mme_layer.sh memvr
#   bash qwen25/scripts/mme_layer.sh memvr 4
#   bash qwen25/scripts/mme_layer.sh memvr 4 "0,1,2,3"
mode="${1:-memvr}"
num_gpus="${2:-1}"
gpu_ids_csv="${3:-}"

if [[ "$mode" != "memvr" && "$mode" != "none" ]]; then
    echo "Invalid mode: $mode"
    echo "Usage: bash qwen25/scripts/mme_layer.sh [memvr|none] [num_gpus] [gpu_ids_csv]"
    exit 1
fi

if ! [[ "$num_gpus" =~ ^[0-9]+$ ]] || [[ "$num_gpus" -lt 1 ]]; then
    echo "Invalid num_gpus: $num_gpus"
    exit 1
fi

# -----------------------------------------------------------------------------
# Layer sweep range (edit here)
# Use zero-based layer index, e.g. 0..28
# -----------------------------------------------------------------------------
layer_start=0
layer_end=28

if ! [[ "$layer_start" =~ ^[0-9]+$ ]] || ! [[ "$layer_end" =~ ^[0-9]+$ ]] || [[ "$layer_start" -lt 0 ]] || [[ "$layer_end" -lt "$layer_start" ]]; then
    echo "Invalid layer range: layer_start=$layer_start layer_end=$layer_end"
    exit 1
fi

results_root="$root_dir/results/Qwen2.5-VL/mme_layer"
run_id="$(date +%Y%m%d_%H%M%S)"
run_dir="$results_root/$run_id"
logs_dir="$run_dir/logs"
csv_file="$run_dir/layer_scores.csv"
combined_log="$run_dir/combined.log"
latest_link="$results_root/latest"

mkdir -p "$logs_dir"

cat > "$csv_file" << 'EOF'
layer,status,perception_total,cognition_total,existence,count,position,color,posters,celebrity,scene,landmark,artwork,ocr,commonsense_reasoning,numerical_calculation,text_translation,code_reasoning,log_file
EOF

parse_scores() {
    local log_file="$1"
    awk '
        function trim(s) { gsub(/^[ \t]+|[ \t]+$/, "", s); return s }
        BEGIN {
            section=""
            p_total=""; c_total=""
            existence=""; count=""; position=""; color=""; posters=""; celebrity=""; scene=""; landmark=""; artwork=""; ocr=""
            commonsense_reasoning=""; numerical_calculation=""; text_translation=""; code_reasoning=""
        }
        /^=========== Perception ===========/ { section="perception"; next }
        /^=========== Cognition ===========/ { section="cognition"; next }
        {
            line=$0
            line=trim(line)

            if (section=="perception" && line ~ /^total score:[[:space:]]+/) {
                sub(/^total score:[[:space:]]+/, "", line)
                p_total=line
                next
            }
            if (section=="cognition" && line ~ /^total score:[[:space:]]+/) {
                sub(/^total score:[[:space:]]+/, "", line)
                c_total=line
                next
            }

            if (line ~ /score:/) {
                n=split(line, a, /[[:space:]]+/)
                if (n >= 3 && a[2] == "score:") {
                    key=a[1]
                    val=a[3]
                    if (key=="existence") existence=val
                    else if (key=="count") count=val
                    else if (key=="position") position=val
                    else if (key=="color") color=val
                    else if (key=="posters") posters=val
                    else if (key=="celebrity") celebrity=val
                    else if (key=="scene") scene=val
                    else if (key=="landmark") landmark=val
                    else if (key=="artwork") artwork=val
                    else if (key=="OCR") ocr=val
                    else if (key=="commonsense_reasoning") commonsense_reasoning=val
                    else if (key=="numerical_calculation") numerical_calculation=val
                    else if (key=="text_translation") text_translation=val
                    else if (key=="code_reasoning") code_reasoning=val
                }
            }
        }
        END {
            printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n", \
                p_total, c_total, \
                existence, count, position, color, posters, celebrity, scene, landmark, artwork, ocr, \
                commonsense_reasoning, numerical_calculation, text_translation, code_reasoning
        }
    ' "$log_file"
}

echo "[LAYER] mode=$mode" | tee -a "$combined_log"
echo "[LAYER] num_gpus=$num_gpus" | tee -a "$combined_log"
echo "[LAYER] gpu_ids_csv=${gpu_ids_csv:-auto}" | tee -a "$combined_log"
echo "[LAYER] layer_range=$layer_start..$layer_end" | tee -a "$combined_log"
echo "[LAYER] run_dir=$run_dir" | tee -a "$combined_log"
echo "[LAYER] csv_file=$csv_file" | tee -a "$combined_log"

total=0
ok=0
fail=0

for layer in $(seq "$layer_start" "$layer_end"); do
    total=$((total + 1))
    run_tag="mme_layer_l${layer}"
    log_file="$logs_dir/layer_${layer}.log"

    echo "[LAYER] ($total) run layer=$layer" | tee -a "$combined_log"

    cmd=(bash qwen25/scripts/mme.sh "$mode" "$num_gpus")
    if [[ -n "$gpu_ids_csv" ]]; then
        cmd+=("$gpu_ids_csv")
    fi

    {
        echo "============================================================"
        echo "[RUN] layer=$layer begin=$(date '+%F %T')"
    } >> "$combined_log"

    if RETRACE_TARGET_LAYERS="$layer" MME_RUN_TAG="$run_tag" "${cmd[@]}" > "$log_file" 2>&1; then
        status="ok"
        ok=$((ok + 1))
    else
        status="fail"
        fail=$((fail + 1))
    fi

    scores="$(parse_scores "$log_file")"
    if [[ -z "$scores" ]]; then
        scores=",,,,,,,,,,,,,,,"
    fi
    echo "${layer},${status},${scores},${log_file}" >> "$csv_file"

    {
        cat "$log_file"
        echo "[RUN] layer=$layer status=$status end=$(date '+%F %T')"
        echo ""
    } >> "$combined_log"
done

ln -sfn "$run_dir" "$latest_link"

{
    echo "[LAYER] done total=$total ok=$ok fail=$fail"
    echo "[LAYER] csv_file=$csv_file"
    echo "[LAYER] latest=$latest_link"
} | tee -a "$combined_log"

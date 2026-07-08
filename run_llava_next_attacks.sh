#!/bin/bash
# Generate LLaVA-NeXT-8B adversarial images for many clean images and objectives.
#
# Usage:
#   bash run_llava_next_attacks.sh [GPU_ID] [IMAGE ...]
#     GPU_ID   : which GPU to use (default 0)
#     IMAGE... : optional image names; if omitted, uses the default list below.
#
# Choose which modes to generate with OBJECTIVES (space-separated CATEGORY:INSTRUCTION):
#   OBJECTIVES="Attack:injection"                       # default (one mode)
#   OBJECTIVES="Attack:injection Attack:spam"           # a couple
#   OBJECTIVES="$ALL_MODES" bash run_llava_next_attacks.sh 0   # everything (see below)
#
# ALL 12 modes (paste as OBJECTIVES to do them all):
#   Attack:injection Attack:spam
#   Language:english Language:spanish Language:french
#   Politics:left Politics:right
#   Formality:formal Formality:informal
#   Sentiment:positive Sentiment:negative Sentiment:neutral
#
# Run detached, sharded across your 4 GPUs (example, injection+spam):
#   OBJECTIVES="Attack:injection Attack:spam" nohup bash run_llava_next_attacks.sh 0 0 1 2 3 > g0.log 2>&1 &
#   ... (repeat for GPUs 1/2/3 with other image subsets)
#
# The script auto-skips objectives an image doesn't have (missing CSV / column),
# so numbered+coco images get all modes while eg* images only get Sentiment.

set -u
cd "$(dirname "$0")"   # run from the repo root

# ---- config (override via env) ----
INSTRUCTION="${INSTRUCTION:-injection}"
CATEGORY="${CATEGORY:-Attack}"
# OBJECTIVES overrides CATEGORY/INSTRUCTION when set. Default = the single pair.
OBJECTIVES="${OBJECTIVES:-${CATEGORY}:${INSTRUCTION}}"
REGION_MODE="${REGION_MODE:-margin}"
REGION_FRAC="${REGION_FRAC:-0.1}"
N_ITERS="${N_ITERS:-2000}"
BATCH="${BATCH:-1}"
EVAL_EVERY="${EVAL_EVERY:-100}"
OUTROOT="${OUTROOT:-output/llava_next}"

# ---- args: GPU id, then optional explicit image list ----
GPU="${1:-0}"; shift || true
if [ "$#" -gt 0 ]; then
    IMAGES=("$@")
else
    IMAGES=(0 1 2 3 4 coco_1 coco_2 coco_3 coco_4 coco_5 coco_6 coco_7 coco_8 coco_9 coco_10)
fi

# map --instruction name -> CSV column name (mirrors INSTRUCTION_MAP in the python)
map_col() {
    case "$1" in
        english) echo en;; spanish) echo es;; french) echo fr;;
        right) echo Republican;; left) echo Democrat;;
        *) echo "$1";;
    esac
}

# does $2 (column) exist in the header of CSV $1 ?
# strip trailing CR (CSVs use CRLF, which otherwise leaves the last column as
# "name\r" and breaks the exact match); split on commas; exact-match the column.
has_column() {
    head -1 "$1" | tr -d '\r' | tr ',' '\n' | grep -qx "$2"
}

echo "### GPU=$GPU  region=$REGION_MODE/$REGION_FRAC  n_iters=$N_ITERS  batch=$BATCH"
echo "### objectives: $OBJECTIVES"
echo "### images: ${IMAGES[*]}"

for NAME in "${IMAGES[@]}"; do
    IMG=$(ls clean_images/${NAME}.* 2>/dev/null | head -1)
    if [ -z "$IMG" ] || [ ! -f "$IMG" ]; then
        echo ">>> SKIP $NAME : no clean image (clean_images/${NAME}.*)"; continue
    fi

    for OBJ in $OBJECTIVES; do
        CAT="${OBJ%%:*}"
        INST="${OBJ##*:}"

        # per-category CSV, else flat CSV (eg* images)
        CSV="instruction_data/${NAME}/${CAT}/dataset.csv"
        [ -f "$CSV" ] || CSV="instruction_data/${NAME}/dataset.csv"

        COL=$(map_col "$INST")
        if [ ! -f "$CSV" ]; then
            echo ">>> SKIP ${NAME}/${CAT}:${INST} : no CSV"; continue
        fi
        if ! has_column "$CSV" "$COL"; then
            echo ">>> SKIP ${NAME}/${CAT}:${INST} : column '$COL' not in $CSV"; continue
        fi

        SAVE="${OUTROOT}/${NAME}/${CAT}/${INST}"
        if [ -f "${SAVE}/bad_prompt.bmp" ]; then
            echo ">>> DONE  ${NAME}/${CAT}:${INST} : exists, skipping"; continue
        fi

        echo "=================================================================="
        echo ">>> ${NAME} | ${CAT}:${INST} | img=$IMG csv=$CSV -> $SAVE"
        echo "=================================================================="
        python llava_next_visual_attack.py \
            --gpu_id "$GPU" \
            --instruction "$INST" \
            --data_path "$CSV" \
            --image_file "$IMG" \
            --save_dir "$SAVE" \
            --region_mode "$REGION_MODE" --region_frac "$REGION_FRAC" \
            --n_iters "$N_ITERS" --batch_size "$BATCH" --eval_every "$EVAL_EVERY" \
            || echo ">>> FAILED ${NAME}/${CAT}:${INST} (continuing)"
    done
done

echo "### all done."

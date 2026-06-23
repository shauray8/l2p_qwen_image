#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

MODEL="${MODEL:-Qwen/Qwen-Image-2512}"
DATA=data/overfit_500
INIT=pretrain_weight/Qwen-Image-Pixel-Init/model.safetensors

# overfit set = first 500 of the cleaned dataset
[ -f "$DATA/metadata.csv" ] || python train/prep_data.py --repo "${REPO:-shauray/l2p-clean}" --out "$DATA"

[ -f "$INIT" ] || python l2p/convert_weights.py --model "$MODEL" --output "$INIT"

[ -f "$DATA/text_cache/index.json" ] || python train/precompute_text_embeds.py --model "$MODEL" --data_dir "$DATA"

NPROC="${NPROC:-1}"
EXTRA="${EXTRA:-}"
LAUNCH=(--data_dir "$DATA" --pixel_init "$INIT" --output_dir runs/l2p_qwen_overfit
        --steps 4000 --lr 5e-5 --weight_decay 0.01 --dataset_repeat 200 --limit 500
        --trainable_scope shallow --first_blocks 6 --last_blocks 6 --grad_checkpointing --fa3
        --optim adamw --save_every 1000 --log_every 10)
if [ "$NPROC" -eq 1 ]; then
  python train/train_overfit_fsdp2.py "${LAUNCH[@]}" $EXTRA
else
  torchrun --nproc_per_node="$NPROC" train/train_overfit_fsdp2.py "${LAUNCH[@]}" $EXTRA
fi

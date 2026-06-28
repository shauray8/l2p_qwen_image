#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.." 

: "${VOL:=/workspace}"                                   # network-volume mount point
: "${MODEL:=Qwen/Qwen-Image-2512}"
: "${REPO:=shauray/l2p-clean}"                           # cleaned dataset
: "${DATA:=$VOL/data/l2p-clean}"
: "${INIT:=$VOL/pretrain_weight/Qwen-Image-Pixel-Init/model.safetensors}"
: "${RUN_DIR:=$VOL/runs/l2p_clean}"

: "${NPROC:=$(python -c 'import torch;print(torch.cuda.device_count())' 2>/dev/null || nvidia-smi --list-gpus 2>/dev/null | wc -l)}"
case "${NPROC}" in ''|*[!0-9]*|0) NPROC=8;; esac
: "${MAX_STEPS:=50000}"
: "${CKPT_EVERY:=200}"                                   # resumable ckpt cadence (aligned with SAMPLE_EVERY)
: "${BATCH_SIZE:=auto}"                                  # per-GPU batch; "auto" scales to GPU mem (H200->8, H100->4)
if [ "$BATCH_SIZE" = auto ]; then
  _memmib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc 0-9); _memmib=${_memmib:-0}
  if   [ "$_memmib" -ge 130000 ]; then BATCH_SIZE=16      # ~141GB H200
  elif [ "$_memmib" -ge 70000 ];  then BATCH_SIZE=8      # ~80GB  H100 / A100-80
  elif [ "$_memmib" -ge 38000 ];  then BATCH_SIZE=4      # ~40GB
  else BATCH_SIZE=1; fi
  echo "[spot] auto batch_size=$BATCH_SIZE (per-GPU mem ${_memmib}MiB)"
fi
: "${SAMPLE_EVERY:=200}"                                 # generate sample images + log to wandb every N steps
: "${N_EVAL:=4}"                                         # how many items to reconstruct each sample step
: "${SAMPLE_STEPS:=28}"                                  # denoising steps per eval image
: "${EXTRA_EVAL:=1}"                                     # extra high-quality sample(s) at SAMPLE_STEPS_HI (5th image)
: "${SAMPLE_STEPS_HI:=50}"                               # denoising steps for the EXTRA_EVAL sample(s)
: "${DECODER_RECON_SIGMA:=0.05}"                         # sigma for the single-step decoder-only recon probe (runs every SAMPLE_EVERY)
: "${LR_SCHEDULE:=constant}"                             # fixed LR; set cosine to revert to the decay schedule
: "${TIMESTEP_SAMPLING:=balanced}"                       # balanced low+high noise across full [0,1]; set qwen_shift to revert
: "${RESIZE_BASE:=0}"                                    # resize imgs to ~NxN area (AR kept); 0=native, 1024=uniform mem
: "${FIRST_BLOCKS:=6}"                                   # trainable leading DiT blocks (overfit-validated: 6)
: "${LAST_BLOCKS:=6}"                                    # trainable trailing DiT blocks (overfit-validated: 6)
: "${KEEP_LAST:=3}"
: "${CKPT_OPTIM:=full}"                                  # full=resume momentum; none=smaller/faster
: "${HF_BACKUP_REPO:=}"                                  # optional off-volume durability, e.g. shauray/l2p-ckpts
: "${EXTRA:=}"                                           # any extra train args
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "$RUN_DIR" "$(dirname "$INIT")"

# HF_HUB_ENABLE_HF_TRANSFER=1 makes every download error out.
if ! python -c "import diffusers, transformers, safetensors, huggingface_hub, hf_transfer, torchao" 2>/dev/null; then
  echo "[spot] installing python deps from requirements.txt"
  python -m pip install -q -r requirements.txt || { echo "[spot] pip install failed"; exit 1; }
fi

if [ "${START_JUPYTER:-1}" = "1" ] && ! pgrep -f "jupyter[ -]lab" >/dev/null 2>&1; then
  python -c "import jupyterlab" 2>/dev/null || python -m pip install -q jupyterlab
  echo "[spot] starting Jupyter Lab on :8888 (notebook-dir=$VOL)"
  nohup jupyter lab --allow-root --ip=0.0.0.0 --port=8888 --no-browser \
    --ServerApp.token="${JUPYTER_TOKEN:-}" --ServerApp.allow_origin='*' \
    --notebook-dir="$VOL" >"$VOL/jupyter.log" 2>&1 &
fi

TERMINATING=0
trap 'echo "[spot] received SIGTERM/SIGINT — not relaunching"; TERMINATING=1' TERM INT

echo "[spot] $(date -u +%FT%TZ) vol=$VOL run=$RUN_DIR nproc=$NPROC max_steps=$MAX_STEPS ckpt_every=$CKPT_EVERY"

if [ ! -f "$INIT" ]; then
  echo "[spot] building pixel-init weights -> $INIT"
  python l2p/convert_weights.py --model "$MODEL" --output "$INIT" || { echo "[spot] convert_weights failed"; exit 1; }
fi
if [ ! -f "$DATA/metadata.csv" ]; then
  echo "[spot] materializing dataset $REPO -> $DATA"
  python train/prep_data.py --repo "$REPO" --out "$DATA" || { echo "[spot] prep_data failed"; exit 1; }
fi
if [ ! -f "$DATA/text_cache/index.json" ]; then
  echo "[spot] precomputing text embeds (one pass over all prompts)"
  python train/precompute_text_embeds.py --model "$MODEL" --data_dir "$DATA" || { echo "[spot] precompute failed"; exit 1; }
fi

while :; do
  [ "$TERMINATING" -eq 1 ] && { echo "[spot] terminating; bye"; break; }

  torchrun --standalone --nproc_per_node="$NPROC" \
    train/train_overfit_fsdp2.py \
      --data_dir "$DATA" --pixel_init "$INIT" \
      --output_dir "$RUN_DIR" --ckpt_dir "$RUN_DIR/ckpt" \
      --resume auto --max_steps "$MAX_STEPS" \
      --ckpt_every "$CKPT_EVERY" --keep_last "$KEEP_LAST" --ckpt_optim "$CKPT_OPTIM" \
      ${HF_BACKUP_REPO:+--hf_backup_repo "$HF_BACKUP_REPO"} \
      --dataset_repeat 1 --trainable_scope shallow --first_blocks "$FIRST_BLOCKS" --last_blocks "$LAST_BLOCKS" \
      --optim adamw --batch_size "$BATCH_SIZE" --resize_base "$RESIZE_BASE" \
      --lr 5e-5 --weight_decay 0.01 --warmup_steps 50 --lr_schedule "$LR_SCHEDULE" \
      --timestep_sampling "$TIMESTEP_SAMPLING" \
      --fa3 \
      --grad_checkpointing --max_grad_norm 1.0 \
      --save_every 2000 --log_every 10 --sample_every "$SAMPLE_EVERY" --n_eval "$N_EVAL" --sample_steps "$SAMPLE_STEPS" \
      --extra_eval "$EXTRA_EVAL" --sample_steps_hi "$SAMPLE_STEPS_HI" --decoder_recon_sigma "$DECODER_RECON_SIGMA" \
      ${WANDB_PROJECT:+--wandb_project "$WANDB_PROJECT" --wandb_name "${WANDB_NAME:-l2p-spot}"} \
      $EXTRA
  code=$?

  if [ "$code" -eq 0 ]; then
    echo "[spot] training COMPLETE (reached $MAX_STEPS steps)."
    if [ -n "${AUTO_STOP_ON_DONE:-}" ] && [ -n "${RUNPOD_API_KEY:-}" ] && [ -n "${RUNPOD_POD_ID:-}" ]; then
      echo "[spot] requesting pod stop via RunPod API"
      curl -s -X POST "https://rest.runpod.io/v1/pods/${RUNPOD_POD_ID}/stop" \
        -H "Authorization: Bearer ${RUNPOD_API_KEY}" >/dev/null || true
    fi
    break
  fi

  [ "$TERMINATING" -eq 1 ] && { echo "[spot] evicted; pod going down, will resume on next boot"; break; }
  echo "[spot] trainer exited ($code) — resuming from latest checkpoint in 10s"
  sleep 10
done

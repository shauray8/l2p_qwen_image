#!/usr/bin/env bash
# ============================================================================
# Idempotent entrypoint for L2P training on a RunPod SPOT pod (8xH100, 1 node).
#
# Spot pods are evicted with only a ~5s SIGTERM. Survival strategy:
#   1) everything lives on a NETWORK VOLUME (default /workspace) that outlives the pod;
#   2) the trainer checkpoints to that volume every $CKPT_EVERY steps and AUTO-RESUMES
#      from the latest checkpoint on boot (--resume auto);
#   3) this script is the pod's start command, so whether the same pod restarts or you
#      redeploy a fresh pod onto the same volume, training picks up where it left off;
#   4) one-time prep (weights/data/text-embeds) is guarded so it only runs on a cold volume.
#
# Set this script as the pod's "Container Start Command":
#   bash /workspace/l2p_qwen_image/train/spot_train.sh
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."   # repo root

# ---- config (override via pod env vars) ------------------------------------
: "${VOL:=/workspace}"                                   # network-volume mount point
: "${MODEL:=Qwen/Qwen-Image-2512}"
: "${REPO:=shauray/l2p-clean}"                           # cleaned dataset
: "${DATA:=$VOL/data/l2p-clean}"
: "${INIT:=$VOL/pretrain_weight/Qwen-Image-Pixel-Init/model.safetensors}"
: "${RUN_DIR:=$VOL/runs/l2p_clean}"
# GPUs the container actually sees (the spot ladder may hand us 8x / 4x / 2x). torch's
# device_count is the source of truth for torchrun ranks; fall back to nvidia-smi, then 8.
: "${NPROC:=$(python -c 'import torch;print(torch.cuda.device_count())' 2>/dev/null || nvidia-smi --list-gpus 2>/dev/null | wc -l)}"
case "${NPROC}" in ''|*[!0-9]*|0) NPROC=8;; esac
: "${MAX_STEPS:=20000}"
: "${CKPT_EVERY:=50}"                                    # ~minutes of work at risk; tune to step time
: "${KEEP_LAST:=3}"
: "${CKPT_OPTIM:=full}"                                  # full=resume momentum; none=smaller/faster
: "${HF_BACKUP_REPO:=}"                                  # optional off-volume durability, e.g. shauray/l2p-ckpts
: "${EXTRA:=}"                                           # any extra train args
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "$RUN_DIR" "$(dirname "$INIT")"

# ---- dependencies (idempotent; only installs when missing in THIS container) -----------
# The RunPod PyTorch image provides torch+CUDA — we never reinstall those. Everything else
# the trainer/prep needs is in requirements.txt. site-packages live in the (ephemeral)
# container, NOT on $VOL, so we probe importability and (re)install on any fresh pod; this
# is a fast no-op once satisfied (e.g. a same-pod restart). hf_transfer must be present or
# HF_HUB_ENABLE_HF_TRANSFER=1 makes every download error out.
if ! python -c "import diffusers, transformers, safetensors, huggingface_hub, hf_transfer, torchao" 2>/dev/null; then
  echo "[spot] installing python deps from requirements.txt"
  python -m pip install -q -r requirements.txt || { echo "[spot] pip install failed"; exit 1; }
fi

# ---- optional: expose Jupyter Lab on :8888 (backgrounded; relaunches on every boot) ----
# Our dockerArgs override replaces the image's default entrypoint, so RunPod's built-in
# Jupyter never starts — we launch it ourselves. Survives evictions: each resume re-runs
# this script and re-spawns it. Set a JUPYTER_TOKEN (else it's open to anyone with the URL).
if [ "${START_JUPYTER:-1}" = "1" ] && ! pgrep -f "jupyter[ -]lab" >/dev/null 2>&1; then
  python -c "import jupyterlab" 2>/dev/null || python -m pip install -q jupyterlab
  echo "[spot] starting Jupyter Lab on :8888 (notebook-dir=$VOL)"
  nohup jupyter lab --allow-root --ip=0.0.0.0 --port=8888 --no-browser \
    --ServerApp.token="${JUPYTER_TOKEN:-}" --ServerApp.allow_origin='*' \
    --notebook-dir="$VOL" >"$VOL/jupyter.log" 2>&1 &
fi

# stop relaunching once the pod is being evicted/shut down
TERMINATING=0
trap 'echo "[spot] received SIGTERM/SIGINT — not relaunching"; TERMINATING=1' TERM INT

echo "[spot] $(date -u +%FT%TZ) vol=$VOL run=$RUN_DIR nproc=$NPROC max_steps=$MAX_STEPS ckpt_every=$CKPT_EVERY"

# ---- one-time, idempotent prep (only does work on a cold volume) ------------
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

# ---- supervised, auto-resuming training loop -------------------------------
# trainer exit codes: 0 = reached MAX_STEPS (done); 42 = evicted/early (resume); other = crash (resume).
while :; do
  [ "$TERMINATING" -eq 1 ] && { echo "[spot] terminating; bye"; break; }

  torchrun --standalone --nproc_per_node="$NPROC" \
    train/train_overfit_fsdp2.py \
      --data_dir "$DATA" --pixel_init "$INIT" \
      --output_dir "$RUN_DIR" --ckpt_dir "$RUN_DIR/ckpt" \
      --resume auto --max_steps "$MAX_STEPS" \
      --ckpt_every "$CKPT_EVERY" --keep_last "$KEEP_LAST" --ckpt_optim "$CKPT_OPTIM" \
      ${HF_BACKUP_REPO:+--hf_backup_repo "$HF_BACKUP_REPO"} \
      --dataset_repeat 1 --trainable_scope shallow --optim adamw8bit \
      --lr 5e-5 --weight_decay 0.01 --warmup_steps 50 --lr_schedule cosine \
      --fa3 \
      --grad_checkpointing --max_grad_norm 1.0 \
      --save_every 2000 --log_every 10 --sample_every 1000 --n_eval 4 \
      ${WANDB_PROJECT:+--wandb_project "$WANDB_PROJECT" --wandb_name "${WANDB_NAME:-l2p-spot}"} \
      $EXTRA
  code=$?

  if [ "$code" -eq 0 ]; then
    echo "[spot] training COMPLETE (reached $MAX_STEPS steps)."
    # Optionally auto-stop the pod so you stop paying. Requires RUNPOD_API_KEY + RUNPOD_POD_ID.
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

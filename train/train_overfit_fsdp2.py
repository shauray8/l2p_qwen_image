#!/usr/bin/env python3
import argparse, json, math, os, signal, sys, time

import torch
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy, CPUOffloadPolicy
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict, set_model_state_dict, StateDictOptions,
)
from safetensors.torch import load_file, save_file

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "l2p"))
sys.path.insert(0, os.path.dirname(__file__))
from qwen_image_l2p import QwenImageL2P, PIXEL_PATCH  # noqa: E402
from dataset import OverfitDataset  # noqa: E402
from optimizers import build_optimizer  # noqa: E402
import sample_eval  # noqa: E402
import ckpt as ckptlib  # noqa: E402

# Set by the SIGTERM handler. RunPod spot eviction sends SIGTERM ~5s before SIGKILL — too
# short to write a checkpoint, so we just flip this flag, stop cleanly at the next step
# boundary, and rely on the periodic checkpoint already on the network volume.
STOP = False

def _on_sigterm(signum, frame):
    global STOP
    STOP = True
    print(f"[{time.strftime('%H:%M:%S')}] SIGTERM received — will stop at next step boundary", flush=True)

try:
    import wandb
    WANDB = True
except ImportError:
    WANDB = False

def setup_dist():
    if "RANK" not in os.environ:
        import socket
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        if "MASTER_PORT" not in os.environ:
            s = socket.socket(); s.bind(("", 0)); os.environ["MASTER_PORT"] = str(s.getsockname()[1]); s.close()
        os.environ["RANK"] = "0"; os.environ["WORLD_SIZE"] = "1"; os.environ["LOCAL_RANK"] = "0"
    rank, world, local = (int(os.environ[k]) for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
    torch.cuda.set_device(local)
    dist.init_process_group(backend="cuda:nccl,cpu:gloo", rank=rank, world_size=world)
    return rank, world, local

def is_main():
    return (not dist.is_initialized()) or dist.get_rank() == 0

def log(msg):
    if is_main():
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def qwen_shift_mu(image_seq_len):
    m = (0.9 - 0.5) / (8192 - 256)
    return image_seq_len * m + (0.5 - m * 256)

def sample_sigma(batch, image_seq_len, mode, device, dtype):
    mu = qwen_shift_mu(image_seq_len)
    shift = lambda u: math.exp(mu) / (math.exp(mu) + (1.0 / u.clamp(1e-4, 1 - 1e-4) - 1.0))
    if mode == "uniform":
        u = torch.rand(batch, device=device)
    elif mode == "logit_normal":
        u = torch.sigmoid(torch.randn(batch, device=device))
    elif mode == "force_text":
        u = shift(1.0 - torch.rand(batch, device=device).pow(3))
        hi = torch.empty(batch, device=device).uniform_(0.92, 0.9999)
        keep_normal = torch.rand(batch, device=device) < 0.25
        u = torch.where(keep_normal, u, hi)
        return u.clamp(1e-4, 1 - 1e-4).to(dtype)
    elif mode == "highnoise":
        u = 1.0 - torch.rand(batch, device=device).pow(3)
        u = shift(u)
        p_one = 0.2
        pin = torch.rand(batch, device=device) < p_one
        u = torch.where(pin, torch.full_like(u, 1 - 1e-3), u)
        return u.clamp(1e-4, 1 - 1e-4).to(dtype)
    else:  # qwen_shift
        u = shift(torch.rand(batch, device=device))
    return u.clamp(1e-4, 1 - 1e-4).to(dtype)

def build_model(args, cfg, device):
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    with torch.device("meta"):
        model = QwenImageL2P.from_config(cfg, feature_use_norm_out=args.feature_use_norm_out)
    torch.set_default_dtype(prev)

    if args.fp8_weights:
        from torchao.float8 import convert_to_float8_training, Float8LinearConfig
        cfg8 = Float8LinearConfig(pad_inner_dim=True)
        convert_to_float8_training(
            model.dit, config=cfg8,
            module_filter_fn=lambda m, fqn: isinstance(m, torch.nn.Linear)
            and ("attn" in fqn or "mlp" in fqn) and "mod" not in fqn,
        )
        log("fp8 weights on DiT attn/mlp linears (torchao float8, pad_inner_dim)")

    if not args.no_fsdp:
        mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        offload = CPUOffloadPolicy() if args.cpu_offload else None
        for blk in model.dit.transformer_blocks:
            fully_shard(blk, mp_policy=mp, offload_policy=offload, reshard_after_forward=True)
        fully_shard(model.local_decoder, mp_policy=mp, offload_policy=offload, reshard_after_forward=True)
        fully_shard(model, mp_policy=mp, offload_policy=offload, reshard_after_forward=True)
        model.to_empty(device="cpu" if args.cpu_offload else device)
        set_model_state_dict(model, load_file(args.pixel_init),
                             options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True, strict=True))
    else:
        model.to_empty(device=device)
        model.load_state_dict(load_file(args.pixel_init), strict=True, assign=False)

    # pos_embed freqs are plain attrs (not buffers) -> still on meta after to_empty; rebuild.
    pe = model.dit.pos_embed
    fresh = type(pe)(theta=pe.theta, axes_dim=pe.axes_dim, scale_rope=pe.scale_rope)
    pe.pos_freqs, pe.neg_freqs = fresh.pos_freqs, fresh.neg_freqs
    log(f"loaded pixel-init: {args.pixel_init} | fsdp={not args.no_fsdp}")
    return model


def enable_fa3():
    """Route the Qwen DiT's joint attention through FlashAttention-3 (Hopper).

    Reuses the proven loader + monkeypatch from the dataset pipeline (dataset/fa3_loader.py,
    batch_infer._install_fa3_dispatch). FA3's flash_attn_func is a differentiable
    autograd.Function, so this works for TRAINING (forward + backward), under FSDP2 and
    non-reentrant gradient checkpointing. The fast path only fires with no attention mask
    (our case: batch=1, prompt_embeds_mask=None) and bf16/fp16 q,k,v; anything else falls
    back to the original SDPA dispatch, so it can never silently corrupt a masked step.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dataset"))
    from fa3_loader import load_fa3
    import diffusers.models.transformers.transformer_qwenimage as tq
    fa3 = load_fa3()
    orig = tq.dispatch_attention_fn

    def fa3_dispatch(query, key, value, attn_mask=None, dropout_p=0.0,
                     is_causal=False, backend=None, parallel_config=None, **kw):
        if attn_mask is None and not is_causal and query.dtype in (torch.bfloat16, torch.float16):
            o = fa3.flash_attn_func(query.contiguous(), key.contiguous(), value.contiguous())
            return o[0] if isinstance(o, tuple) else o
        return orig(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                    is_causal=is_causal, backend=backend, parallel_config=parallel_config, **kw)

    tq.dispatch_attention_fn = fa3_dispatch
    log("FA3 enabled on Qwen DiT joint attention (flash_attn_func, no-mask fast path)")


class CpuEMA:
    def __init__(self, model, decay, update_every):
        self.decay, self.every = decay, update_every
        self.named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        self.ema = {n: p.detach().to("cpu", torch.bfloat16).clone() for n, p in self.named}
        self.stage = {n: torch.empty_like(self.ema[n]).pin_memory() for n, _ in self.named}
        self._backup = None

    @torch.no_grad()
    def update(self):
        for n, p in self.named:
            self.stage[n].copy_(p.detach(), non_blocking=True)
        torch.cuda.synchronize()
        for n, _ in self.named:
            self.ema[n].lerp_(self.stage[n], 1 - self.decay)

    @torch.no_grad()
    def swap_in(self):
        self._backup = {n: p.detach().clone() for n, p in self.named}
        for n, p in self.named:
            p.copy_(self.ema[n].to(p.device, p.dtype))

    @torch.no_grad()
    def swap_out(self):
        for n, p in self.named:
            p.copy_(self._backup[n])
        self._backup = None

def apply_trainable_scope(model, args):
    if args.trainable_scope == "full":
        for p in model.parameters():
            p.requires_grad_(True)
        return
    for p in model.parameters():
        p.requires_grad_(False)
    for mod in (model.local_decoder, model.dit.img_in, model.dit.norm_out):
        for p in mod.parameters():
            p.requires_grad_(True)
    n = len(model.dit.transformer_blocks)
    idx = set(range(args.first_blocks)) | set(range(n - args.last_blocks, n))
    for i in idx:
        for p in model.dit.transformer_blocks[i].parameters():
            p.requires_grad_(True)
    log(f"shallow scope: img_in + blocks {sorted(idx)} + norm_out + local_decoder")

def lr_lambda(step, warmup, total, schedule):
    if step < warmup:
        return (step + 1) / max(1, warmup)
    if schedule == "cosine":
        p = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))
    if schedule == "trapezoid":  # modded-nanogpt: flat, then linear cooldown to 0.15x over last 40%
        cd = int(0.4 * total)
        if step < total - cd:
            return 1.0
        t = (step - (total - cd)) / max(1, cd)
        return (1 - t) + 0.15 * t
    return 1.0

def load_eval_items(ds, n, device):
    items = []
    for i in range(min(n, len(ds.rows))):
        d = ds[i]
        img = ((d["image"].clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(1, 2, 0).numpy()
        items.append({"image_np": img, "prompt_embeds": d["prompt_embeds"], "file_name": d["file_name"]})
    return items

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--text_cache", default=None)
    ap.add_argument("--pixel_init", required=True)
    ap.add_argument("--transformer_config", default=None)
    ap.add_argument("--output_dir", default="./runs/l2p_qwen_overfit")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--adamw_lr", type=float, default=3e-4, help="AdamW LR for non-2D params under muon")
    ap.add_argument("--betas", type=float, nargs=2, default=[0.9, 0.95])
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_steps", type=int, default=50)
    ap.add_argument("--lr_schedule", default="constant", choices=["constant", "cosine", "trapezoid"])
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--dataset_repeat", type=int, default=200)
    ap.add_argument("--limit", type=int, default=None, help="use only the first N images (overfit-memorize subset)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--timestep_sampling", default="qwen_shift",
                    choices=["uniform", "logit_normal", "qwen_shift", "highnoise", "force_text"])
    ap.add_argument("--grad_checkpointing", action="store_true", default=True)
    ap.add_argument("--no_grad_checkpointing", dest="grad_checkpointing", action="store_false")
    ap.add_argument("--grad_checkpointing_offload", action="store_true")
    ap.add_argument("--feature_use_norm_out", action="store_true", default=True)
    ap.add_argument("--no_feature_use_norm_out", dest="feature_use_norm_out", action="store_false")
    ap.add_argument("--trainable_scope", default="shallow", choices=["shallow", "full"])
    ap.add_argument("--first_blocks", type=int, default=3)
    ap.add_argument("--last_blocks", type=int, default=3)
    ap.add_argument("--no_fsdp", action="store_true", help="single-GPU: skip FSDP (faster, optimizer-friendly)")
    ap.add_argument("--cpu_offload", action="store_true")
    ap.add_argument("--fp8_weights", action="store_true")
    ap.add_argument("--optim", default="adamw",
                    choices=["adamw", "adamw8bit", "adamwfp8", "muon", "dion_muon", "shampoo"])
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--attention_backend", default=None)
    ap.add_argument("--fa3", action="store_true",
                    help="route Qwen DiT joint attention through FlashAttention-3 (Hopper); fastest")
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--init_step", type=int, default=0, help="starting step counter (for resume continuity)")
    # spot-resilient resume / checkpointing (see ckpt.py)
    ap.add_argument("--max_steps", type=int, default=0,
                    help="total target steps (overrides init_step+steps). Set this for spot resume.")
    ap.add_argument("--resume", default="auto", choices=["auto", "none"],
                    help="auto: resume from latest checkpoint in --ckpt_dir if present")
    ap.add_argument("--ckpt_dir", default=None, help="resumable-checkpoint dir (default: <output_dir>/ckpt)")
    ap.add_argument("--ckpt_every", type=int, default=200, help="write a resumable checkpoint every N steps")
    ap.add_argument("--keep_last", type=int, default=3, help="how many resumable checkpoints to retain")
    ap.add_argument("--ckpt_optim", default="full", choices=["full", "none"],
                    help="full: save optimizer state (resumes momentum); none: smaller/faster, loses momentum")
    ap.add_argument("--hf_backup_repo", default=None, help="also push each checkpoint to this HF model repo")
    ap.add_argument("--ema", action="store_true", help="track EMA of trainable weights (CPU); sample from EMA")
    ap.add_argument("--ema_decay", type=float, default=0.999)
    ap.add_argument("--ema_update_every", type=int, default=8)
    ap.add_argument("--save_every", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=10)
    # sampling / eval
    ap.add_argument("--sample_every", type=int, default=100)
    ap.add_argument("--sample_steps", type=int, default=24)
    ap.add_argument("--n_eval", type=int, default=2, help="# fixed training items to recon-eval")
    ap.add_argument("--eval_seed", type=int, default=0)
    ap.add_argument("--cfg_scale", type=float, default=1.0, help="classifier-free guidance scale at eval")
    ap.add_argument("--uncond_embed", default=None, help=".pt with unconditional prompt_embeds for CFG")
    # wandb
    ap.add_argument("--wandb_project", default=None)
    ap.add_argument("--wandb_name", default=None)
    ap.add_argument("--wandb_group", default=None)
    args = ap.parse_args()

    rank, world, local = setup_dist()
    device = torch.device("cuda", local)
    torch.manual_seed(args.seed + rank)
    os.makedirs(args.output_dir, exist_ok=True)
    signal.signal(signal.SIGTERM, _on_sigterm)
    ckpt_dir = args.ckpt_dir or os.path.join(args.output_dir, "ckpt")

    use_wandb = bool(args.wandb_project) and WANDB and is_main()
    if use_wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_name, group=args.wandb_group,
                   config=vars(args), dir=args.output_dir)

    cfg_path = args.transformer_config or os.path.join(os.path.dirname(args.pixel_init), "transformer_config.json")
    cfg = {k: v for k, v in json.load(open(cfg_path)).items() if not k.startswith("_")}

    model = build_model(args, cfg, device)
    apply_trainable_scope(model, args)
    if args.fa3:
        enable_fa3()
    if args.attention_backend:
        model.dit.set_attention_backend(args.attention_backend)
        log(f"attention backend: {args.attention_backend}")
    model.train()
    gen_model = model            # uncompiled module for sampling (avoids eval-shape recompiles)
    if args.compile:
        model = torch.compile(model, dynamic=True)
        log("torch.compile enabled (dynamic shapes)")
    opt = build_optimizer(model, args)
    total_steps = args.max_steps if args.max_steps else args.steps
    opt_total = max(1, total_steps // args.grad_accum)  # scheduler advances per optimizer step
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_lambda(s, args.warmup_steps, opt_total, args.lr_schedule))
    ema = CpuEMA(model, args.ema_decay, args.ema_update_every) if args.ema else None
    if ema:
        log(f"EMA on (decay {args.ema_decay}, every {args.ema_update_every} steps, CPU bf16 pinned)")

    # ---- spot resume: pick up from the latest checkpoint on the (network) volume ----
    resume_step = None
    if args.resume == "auto":
        latest = ckptlib.latest_ckpt(ckpt_dir)
        if latest:
            resume_step = ckptlib.load_resumable(latest, model, opt, sched, ema)
        else:
            log(f"resume=auto but no checkpoint in {ckpt_dir} — starting fresh from pixel_init")

    ds = OverfitDataset(args.data_dir, args.text_cache, repeat=args.dataset_repeat, limit=args.limit)
    sampler = torch.utils.data.DistributedSampler(ds, world, rank, shuffle=True, seed=args.seed) if world > 1 else None
    dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=(sampler is None), sampler=sampler,
                                     num_workers=2, collate_fn=lambda b: b[0], drop_last=True)
    eval_items = load_eval_items(ds, args.n_eval, device) if args.n_eval else []
    uncond = None
    if args.uncond_embed and args.cfg_scale != 1.0:
        uncond = torch.load(args.uncond_embed, weights_only=True)["prompt_embeds"]
        log(f"CFG eval: scale {args.cfg_scale}, uncond {tuple(uncond.shape)}")

    tot = sum(p.numel() for p in model.parameters())
    trn = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"params {tot/1e9:.2f}B | trainable {trn/1e9:.2f}B ({args.trainable_scope}) | world={world} "
        f"| optim={args.optim} compile={args.compile} fp8={args.fp8_weights} offload={args.cpu_offload}")

    hist = {"loss": [], "grad_norm": [], "psnr": []}

    def _recon():
        return sample_eval.recon_metrics(gen_model, eval_items, args.sample_steps, device,
                                         args.eval_seed, uncond_embeds=uncond, cfg_scale=args.cfg_scale)

    def do_eval(step):
        if not eval_items:
            return
        import numpy as np
        psnr, lp, pairs = _recon()
        hist["psnr"].append((step, psnr))
        ema_psnr = None
        if ema: 
            ema.swap_in()
            try:
                ema_psnr, _, ema_pairs = _recon()
            finally:
                ema.swap_out()
        if not use_wandb:
            log(f"  eval@{step}: recon_psnr {psnr:.2f}" + (f" (ema {ema_psnr:.2f})" if ema_psnr else ""))
            return
        imgs = [wandb.Image(np.concatenate([g, gt], axis=1), caption=f"{os.path.basename(fn)} (gen|gt)")
                for g, gt, fn in pairs]
        d = {"eval/recon_psnr": psnr, "samples": imgs}
        if lp is not None:
            d["eval/recon_lpips"] = lp
        if ema_psnr is not None:
            d["eval/recon_psnr_ema"] = ema_psnr
        wandb.log(d, step=step)
        log(f"  eval@{step}: recon_psnr {psnr:.2f}" + (f" (ema {ema_psnr:.2f})" if ema_psnr else ""))

    step = resume_step if resume_step is not None else args.init_step
    final_step = args.max_steps if args.max_steps else (args.init_step + args.steps)
    init_step = step
    save_optim = (args.ckpt_optim == "full")
    log(f"training step {step} -> {final_step}"
        + (f" (resumed)" if resume_step is not None else "")
        + f" | ckpt every {args.ckpt_every} (optim={args.ckpt_optim}) -> {ckpt_dir}")
    t0 = time.time()
    opt.zero_grad(set_to_none=True)
    if resume_step is None:
        do_eval(step)
    epoch = step  # vary DistributedSampler order across restarts/epochs
    done = False
    while not done:
        if sampler is not None:
            sampler.set_epoch(epoch)
        epoch += 1
        for data in dl:
            x0 = data["image"].unsqueeze(0).to(device, torch.float32)
            emb = data["prompt_embeds"].unsqueeze(0).to(device, torch.bfloat16)
            B, C, H, W = x0.shape
            isl = (H // PIXEL_PATCH) * (W // PIXEL_PATCH)
            sigma = sample_sigma(B, isl, args.timestep_sampling, device, torch.float32)
            noise = torch.randn_like(x0)
            s = sigma.view(B, 1, 1, 1)
            x_t = (1 - s) * x0 + s * noise
            target = noise - x0
            pred = model(noisy_image=x_t, timestep=sigma, prompt_embeds=emb, prompt_embeds_mask=None,
                         use_gradient_checkpointing=args.grad_checkpointing,
                         use_gradient_checkpointing_offload=args.grad_checkpointing_offload)
            loss = torch.nn.functional.mse_loss(pred.float(), target.float()) / args.grad_accum
            loss.backward()

            gnorm = None
            if (step + 1) % args.grad_accum == 0:
                if args.max_grad_norm > 0:
                    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                opt.step()
                opt.zero_grad(set_to_none=True)
                sched.step()
                if ema and step % args.ema_update_every == 0:
                    ema.update()

            if step % args.log_every == 0:
                dt = (time.time() - t0) / max(step - init_step, 1)
                mem = torch.cuda.max_memory_allocated() / 1e9
                lval = loss.item() * args.grad_accum
                gn = float(gnorm) if gnorm is not None else 0.0
                hist["loss"].append(lval); hist["grad_norm"].append(gn)
                log(f"step {step}/{final_step} loss {lval:.5f} gnorm {gn:.3f} sigma {sigma.mean():.3f} "
                    f"| {dt:.2f}s/it peak {mem:.1f}GB")
                if use_wandb:
                    wandb.log({"train/loss": lval, "train/grad_norm": gn, "train/sigma": float(sigma.mean()),
                               "train/lr": sched.get_last_lr()[0], "perf/s_per_it": dt,
                               "perf/peak_gb": mem}, step=step)

            step += 1
            if args.sample_every and step % args.sample_every == 0:
                do_eval(step)
            if args.save_every and step % args.save_every == 0:
                save_ckpt(model, args.output_dir, step, ema)
            if args.ckpt_every and step % args.ckpt_every == 0:
                ckptlib.save_resumable(model, opt, sched, step, ema, ckpt_dir,
                                       keep_last=args.keep_last, save_optim=save_optim,
                                       hf_repo=args.hf_backup_repo)
            if STOP:
                log("stopping on SIGTERM — writing final resumable checkpoint")
                ckptlib.save_resumable(model, opt, sched, step, ema, ckpt_dir,
                                       keep_last=args.keep_last, save_optim=save_optim,
                                       hf_repo=args.hf_backup_repo)
                done = True
                break
            if step >= final_step:
                done = True
                break

    completed = step >= final_step
    if completed:
        # training reached the target: final eval + resumable ckpt + inference-weights export
        do_eval(step)
        ckptlib.save_resumable(model, opt, sched, step, ema, ckpt_dir,
                               keep_last=args.keep_last, save_optim=save_optim,
                               hf_repo=args.hf_backup_repo)
        save_ckpt(model, args.output_dir, step, ema)

    if completed and is_main():
        import numpy as np
        L, G, P = hist["loss"], hist["grad_norm"], hist["psnr"]
        half = L[len(L) // 2:] if L else [0]
        summary = {
            "name": args.wandb_name, "optim": args.optim, "lr": args.lr,
            "final_loss": L[-1] if L else None,
            "loss_ema_last": float(np.mean(L[-5:])) if L else None,
            "loss_std_secondhalf": float(np.std(half)),
            "loss_min": float(np.min(L)) if L else None,
            "max_grad_norm": float(np.max(G)) if G else None,
            "nan": bool(any(map(lambda x: x != x, L))),
            "psnr_first": P[0][1] if P else None, "psnr_last": P[-1][1] if P else None,
            "psnr_delta": (P[-1][1] - P[0][1]) if len(P) >= 2 else None,
            "s_per_it": (time.time() - t0) / max(step, 1),
            "config": {k: getattr(args, k) for k in
                       ("lr", "optim", "weight_decay", "betas", "first_blocks", "last_blocks",
                        "timestep_sampling", "lr_schedule", "warmup_steps", "max_grad_norm", "compile")},
        }
        json.dump(summary, open(os.path.join(args.output_dir, "summary.json"), "w"), indent=1)
        log(f"summary: loss {summary['loss_ema_last']:.4f} (std {summary['loss_std_secondhalf']:.4f}) "
            f"psnrΔ {summary['psnr_delta']} maxgn {summary['max_grad_norm']:.2f}")
        if use_wandb:
            wandb.summary.update(summary)
    log(f"done in {time.time()-t0:.0f}s" + ("" if completed else " (stopped early — will resume)"))
    if use_wandb:
        wandb.finish()
    dist.destroy_process_group()
    # exit code tells the spot supervisor what to do: 0 = finished, 42 = evicted/early -> relaunch
    if not completed:
        sys.exit(42)

def save_ckpt(model, out, step, ema=None):
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod
    try:
        sd = get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
    except Exception:
        sd = {k: v.cpu() for k, v in model.state_dict().items()}
    if is_main():
        save_file({k: v.contiguous() for k, v in sd.items()}, os.path.join(out, f"step-{step}.safetensors"))
        log(f"saved step-{step}.safetensors")
        if ema is not None: 
            esd = dict(sd)
            for n, t in ema.ema.items():
                esd[n] = t.to(torch.bfloat16).contiguous()
            save_file(esd, os.path.join(out, f"step-{step}-ema.safetensors"))
            log(f"saved step-{step}-ema.safetensors")
    if dist.is_initialized():
        dist.barrier()

if __name__ == "__main__":
    main()

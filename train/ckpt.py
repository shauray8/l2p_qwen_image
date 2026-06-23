#!/usr/bin/env python3
import glob, os
import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict, set_model_state_dict,
    get_optimizer_state_dict, set_optimizer_state_dict,
    StateDictOptions,
)

def _is_main():
    return (not dist.is_initialized()) or dist.get_rank() == 0

def _desubclass_tensors(obj):
    if isinstance(obj, dict):
        return {k: _desubclass_tensors(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_desubclass_tensors(v) for v in obj)
    if isinstance(obj, torch.Tensor) and type(obj) is not torch.Tensor:
        plain = torch.zeros(obj.shape, dtype=torch.float32, device="cpu")
        plain.copy_(obj.detach().cpu())
        return plain
    return obj

def _unwrap(model):
    return model._orig_mod if hasattr(model, "_orig_mod") else model

def _trainable_names(model):
    return {n for n, p in _unwrap(model).named_parameters() if p.requires_grad}

def _step_of(path):
    return int(os.path.basename(path).split("-")[-1].split(".")[0])

def latest_ckpt(ckpt_dir):
    ptr = os.path.join(ckpt_dir, "LATEST")
    if os.path.isfile(ptr):
        name = open(ptr).read().strip()
        full = os.path.join(ckpt_dir, name)
        if os.path.isfile(full):
            return full
    cks = glob.glob(os.path.join(ckpt_dir, "ckpt-*.pt"))
    return max(cks, key=_step_of) if cks else None

@torch.no_grad()
def save_resumable(model, opt, sched, step, ema, ckpt_dir, keep_last=3,
                   save_optim=True, hf_repo=None, extra=None):
    m = _unwrap(model)
    opts = StateDictOptions(full_state_dict=True, cpu_offload=True)
    msd = get_model_state_dict(m, options=opts)                       # all ranks participate (gather)
    osd = get_optimizer_state_dict(m, opt, options=opts) if save_optim else None

    if _is_main():
        tn = _trainable_names(m)
        msd = {k: v for k, v in msd.items() if k in tn} 
        osd = _desubclass_tensors(osd)
        payload = {
            "step": int(step),
            "model": msd,
            "optim": osd,
            "sched": sched.state_dict() if sched is not None else None,
            "ema": {n: t.cpu().clone() for n, t in ema.ema.items()} if ema is not None else None,
            "extra": extra or {},
        }
        os.makedirs(ckpt_dir, exist_ok=True)
        final = os.path.join(ckpt_dir, f"ckpt-{step}.pt")
        tmp = os.path.join(ckpt_dir, f".ckpt-{step}.pt.tmp")
        torch.save(payload, tmp)
        os.replace(tmp, final)                  
        lt = os.path.join(ckpt_dir, "LATEST")
        with open(lt + ".tmp", "w") as f:
            f.write(os.path.basename(final))
        os.replace(lt + ".tmp", lt)
        # rolling retention
        for old in sorted(glob.glob(os.path.join(ckpt_dir, "ckpt-*.pt")), key=_step_of)[:-keep_last]:
            try:
                os.remove(old)
            except OSError:
                pass
        print(f"[ckpt] saved {os.path.basename(final)} "
              f"({os.path.getsize(final)/1e9:.2f} GB, optim={'on' if save_optim else 'off'})", flush=True)
        if hf_repo:
            _hf_backup(final, ckpt_dir, hf_repo, step)
    if dist.is_initialized():
        dist.barrier()

def _hf_backup(ckpt_path, ckpt_dir, repo, step):
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(repo, repo_type="model", exist_ok=True, private=False)
        api.upload_file(path_or_fileobj=ckpt_path, path_in_repo=f"ckpt/{os.path.basename(ckpt_path)}",
                        repo_id=repo, repo_type="model", commit_message=f"resumable ckpt step {step}")
        lt = os.path.join(ckpt_dir, "LATEST")
        if os.path.isfile(lt):
            api.upload_file(path_or_fileobj=lt, path_in_repo="ckpt/LATEST",
                            repo_id=repo, repo_type="model", commit_message=f"LATEST -> step {step}")
        print(f"[ckpt] HF backup - {repo}/ckpt/{os.path.basename(ckpt_path)}", flush=True)
    except Exception as e:
        print(f"[ckpt] WARN HF backup failed: {type(e).__name__}: {e}", flush=True)


def load_resumable(path, model, opt, sched, ema):
    m = _unwrap(model)
    payload = torch.load(path, map_location="cpu", weights_only=False)

    set_model_state_dict(
        m, payload["model"],
        options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True, strict=False))

    if opt is not None and payload.get("optim") is not None:
        try:
            # Dequantize on every rank before the collective. All ranks loaded the file
            # independently, so this is purely local — no NCCL involved. Without this,
            osd = _desubclass_tensors(payload["optim"])
            set_optimizer_state_dict(
                m, opt, osd,
                options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True))
        except Exception as e:
            if _is_main():
                print(f"[ckpt] WARN optimizer state not restored ({type(e).__name__}: {e}); "
                      f"continuing with fresh optimizer", flush=True)

    if sched is not None and payload.get("sched"):
        try:
            sched.load_state_dict(payload["sched"])
        except Exception:
            pass

    if ema is not None and payload.get("ema"):
        for n, t in payload["ema"].items():
            if n in ema.ema:
                ema.ema[n].copy_(t.to(ema.ema[n].dtype))

    step = int(payload["step"])
    if _is_main():
        print(f"[ckpt] resumed from {os.path.basename(path)} at step {step}", flush=True)
    return step

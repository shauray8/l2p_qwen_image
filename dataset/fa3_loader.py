"""
FlashAttention-3 (Hopper) loader for Qwen-Image-2512 generation.

The `kernels` PyPI release (0.15.2) can't parse the *newer* metadata schema that
`kernels-community/flash-attn3` ships, so `get_kernel(...)` raises on metadata
parsing. The kernel itself is a fully-built package, so we resolve the cached
build variant and import it directly. Verified on H100 (torch 2.8 / cu128):

    funcs: FlashAttnFunc, FlashAttnVarlenFunc, flash_attn_func, ...
    flash_attn_func(q,k,v) -> finite bf16 output  ✔

Usage:
    from fa3_loader import load_fa3
    fa3 = load_fa3()                 # module exposing flash_attn_func / flash_attn_varlen_func
    out = fa3.flash_attn_func(q, k, v)   # q,k,v: [B, S, H, D] bf16/fp16
"""

import os, sys, glob, importlib
from huggingface_hub import snapshot_download

_REPO = "kernels-community/flash-attn3"


def _build_tag() -> str:
    """torch28-cxx11-cu128-x86_64-linux style variant tag for this interpreter."""
    import torch
    tv = "".join(torch.__version__.split("+")[0].split(".")[:2])      # 2.8.x -> "28"
    cu = "cu" + torch.version.cuda.replace(".", "")                    # 12.8  -> "cu128"
    return f"torch{tv}-cxx11-{cu}-x86_64-linux"


def load_fa3():
    """Download (if needed) and import the FA3 kernel, returning the module.

    Fetches ONLY the build variant matching the current torch/CUDA from the
    `main` revision (which carries the full variant matrix incl. cu130).
    """
    import torch
    tag = _build_tag()
    cu = "cu" + torch.version.cuda.replace(".", "")
    tv = "".join(torch.__version__.split("+")[0].split(".")[:2])
    # try exact, then same-cuda any-torch2.x, then same-torch any-cuda
    patterns = [f"build/{tag}/*",
                f"build/torch2*-cxx11-{cu}-x86_64-linux/*",
                f"build/torch{tv}-cxx11-cu*-x86_64-linux/*"]
    snap = None
    for pat in patterns:
        snap = snapshot_download(_REPO, repo_type="model", revision="main",
                                 allow_patterns=[pat, "*.json", "*.py"])
        if glob.glob(os.path.join(snap, "build", "*", "flash_attn_interface.py")):
            break
    cands = ([os.path.dirname(p) for p in
              glob.glob(os.path.join(snap, "build", tag, "flash_attn_interface.py"))]
             or [os.path.dirname(p) for p in
                 glob.glob(os.path.join(snap, "build", f"torch2*-{cu}-*", "flash_attn_interface.py"))])
    if not cands:
        raise RuntimeError(f"no FA3 build variant for torch={torch.__version__} ({tag})")
    variant = cands[0]

    # Import the variant dir as a named package so its relative imports resolve.
    pkg_root = os.path.dirname(variant)
    alias = "_fa3_" + os.path.basename(variant).replace("-", "_").replace(".", "_")
    link_root = os.path.join(os.path.dirname(__file__), ".fa3_pkg")
    os.makedirs(link_root, exist_ok=True)
    link = os.path.join(link_root, alias)
    if not os.path.islink(link) and not os.path.exists(link):
        os.symlink(variant, link)
    if link_root not in sys.path:
        sys.path.insert(0, link_root)
    return importlib.import_module(f"{alias}.flash_attn_interface")


if __name__ == "__main__":
    import torch
    fa3 = load_fa3()
    q = torch.randn(2, 512, 8, 128, dtype=torch.bfloat16, device="cuda")
    o = fa3.flash_attn_func(q, torch.randn_like(q), torch.randn_like(q))
    o = o[0] if isinstance(o, tuple) else o
    print("FA3 OK:", o.shape, o.dtype, "finite:", torch.isfinite(o).all().item())

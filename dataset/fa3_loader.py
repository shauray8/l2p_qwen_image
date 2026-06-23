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

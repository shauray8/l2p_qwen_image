import torch
from torch import Tensor

_PE_COEFFS = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

@torch.no_grad()
def _polar_express(G: Tensor) -> Tensor:
    assert G.ndim == 2
    X = G.bfloat16()
    transpose = X.size(0) > X.size(1)
    if transpose:
        X = X.T
    X = X / (X.norm() + 1e-7)
    for a, b, c in _PE_COEFFS:
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transpose:
        X = X.T
    return X

class Muon(torch.optim.Optimizer):
    def __init__(self, muon_params, adam_params, lr=0.02, momentum_warmup=300,
                 weight_decay=0.0, adam_lr=3e-4, adam_betas=(0.9, 0.95), adam_wd=0.01):
        groups = [
            dict(params=list(muon_params), use_muon=True, lr=lr, weight_decay=weight_decay),
            dict(params=list(adam_params), use_muon=False, lr=adam_lr, betas=adam_betas,
                 weight_decay=adam_wd, eps=1e-8),
        ]
        super().__init__(groups, {})
        self._step = 0
        self.momentum_warmup = momentum_warmup

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        self._step += 1
        mom = 0.85 + (0.95 - 0.85) * min(1.0, self._step / max(1, self.momentum_warmup))
        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    st = self.state[p]
                    if "buf" not in st:
                        st["buf"] = torch.zeros_like(p, dtype=torch.float32)
                    g = p.grad.float()
                    buf = st["buf"]
                    buf.lerp_(g, 1 - mom)              # buf = mom*buf + (1-mom)*g
                    g = g.lerp_(buf, mom)              # Nesterov lookahead
                    o = _polar_express(g).to(p.dtype)
                    shape_mult = max(1.0, o.size(-2) / o.size(-1)) ** 0.5
                    if group["weight_decay"]:
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(o, alpha=-group["lr"] * shape_mult)
            else:
                b1, b2 = group["betas"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    st = self.state[p]
                    if "step" not in st:
                        st["step"] = 0
                        st["m"] = torch.zeros_like(p, dtype=torch.float32)
                        st["v"] = torch.zeros_like(p, dtype=torch.float32)
                    g = p.grad.float()
                    st["step"] += 1
                    st["m"].lerp_(g, 1 - b1)
                    st["v"].mul_(b2).addcmul_(g, g, value=1 - b2)
                    bc1, bc2 = 1 - b1 ** st["step"], 1 - b2 ** st["step"]
                    denom = (st["v"].sqrt() / (bc2 ** 0.5)).add_(group["eps"])
                    if group["weight_decay"]:
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.addcdiv_((st["m"] / bc1).to(p.dtype), denom.to(p.dtype), value=-group["lr"])
        return loss

def _is_muon_param(name, p):
    return (p.ndim == 2 and "transformer_blocks" in name
            and ("attn" in name or "mlp" in name) and "norm" not in name)

def build_optimizer(model, args):
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if args.optim == "muon":
        muon_p = [p for n, p in named if _is_muon_param(n, p)]
        adam_p = [p for n, p in named if not _is_muon_param(n, p)]
        return Muon(muon_p, adam_p, lr=args.lr, weight_decay=args.weight_decay,
                    adam_lr=args.adamw_lr, adam_wd=args.weight_decay)
    params = [p for _, p in named]
    if args.optim == "adamw":
        return torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay, betas=tuple(args.betas))
    if args.optim == "adamw8bit":
        from torchao.optim import AdamW8bit
        return AdamW8bit(params, lr=args.lr, weight_decay=args.weight_decay, betas=tuple(args.betas))
    if args.optim == "adamwfp8":
        from torchao.optim import AdamWFp8
        return AdamWFp8(params, lr=args.lr, weight_decay=args.weight_decay, betas=tuple(args.betas))
    if args.optim == "dion_muon":
        from dion import Muon as DionMuon
        muon_p = [p for n, p in named if _is_muon_param(n, p)]
        rest_p = [p for n, p in named if not _is_muon_param(n, p)]
        groups = [dict(params=muon_p), dict(params=rest_p, algorithm="adamw")]
        return DionMuon(groups, lr=args.lr, weight_decay=args.weight_decay,
                        betas=tuple(args.betas), nesterov=True, adjust_lr="rms_norm")
    if args.optim == "shampoo":
        import torch_optimizer as topt
        return topt.Shampoo(params, lr=args.lr, weight_decay=args.weight_decay)
    raise ValueError(args.optim)

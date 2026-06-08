# L2P Training Dataset — Construction, Curation & Quality

How we build the synthetic prompt → image dataset for the **L2P** run (converting a
latent diffusion model to a pixel-space model), and the specific mechanisms that make
it diverse, aesthetically strong, well-distributed, and clean.

---

## 1. What we're building & why

L2P (*"L2P: Unlocking Latent Potential for Pixel Generation"*, arXiv:2605.12013) distills
a pixel-space model from a **source LDM's own generations** — it never needs real data.
So the dataset is two coupled artifacts:

1. **Prompts** — diverse, aesthetic, descriptive text prompts (this repo).
2. **Images** — generated from those prompts by the source model (**Qwen-Image-2512**),
   then aesthetically curated. These (prompt, image) pairs are the training corpus.

**Sizing follows the paper:** L2P used **~10k prompts → 20k images** (2 seeds each) at
1024², expanding 4 super-classes / 17 sub-classes into **>1,000 fine-grained categories**,
with prompts concentrated at **200–350 chars**. We target the same shape (and overshoot
to absorb filtering), so the final clean set is ~16–19k prompts.

**Aesthetic north star: FLUX.1-Krea.** "Make AI images that don't look AI." Quality over
quantity, opinionated art direction, and engineered *against* the "AI look" (waxy skin,
blurry backgrounds, washed-out/symmetric compositions, over-soft textures, palette collapse).

---

## 2. The pipeline (stage by stage)

```
taxonomy.py ─┐
             ├─► Stage 0: open-ended category invention (LLM) ─► >1,000 categories
system.txt ──┘                                                        │
                                                                      ▼
                              Stage A: subject expansion (LLM) ─► diverse subject pool
                                                                      │
                                                                      ▼
                              Stage B: prompt generation (vLLM, Qwen3.6-35B-A3B) ─► JSON prompts
                                                                      │
                                                                      ▼
                              Stage C: inline filtering (length / refusal / tag-soup / exact-dedup)
                                                                      │
   clean_prompts.py ◄─────────────────────────────────────────────────┘  normalize, re-dedup
                                                                      │
   semantic_dedup.py ◄────────────────────────────────────────────────┘  embed → drop near-dups → cluster-coverage
                                                                      │
                                                                      ▼
                              Stage D: image gen + aesthetic selection (Qwen-Image + PickScore)
                                                                      │
                                                                      ▼
                                              curated (prompt, image) pairs → L2P training
```

---

## 3. How we ensure it's the *best* dataset

### 3.1 Diversity (cover the data manifold, no clustering)
- **Open-ended category invention** — instead of ~19 fixed buckets, an LLM invents
  **>1,000 fine-grained categories** (scaled as `target/8`), steered across photogenic
  themes per super-class. This is what the paper does and is the main diversity driver.
- **Per-batch randomization** — every generation batch samples a fresh combination of
  *viewpoint × lighting × style × color-grade × technique (film stock / lens / art movement)*,
  plus rotated sentence openers and an explicit "don't reuse the previous prompt's words".
- **Measured result:** 13,196 distinct subjects (top subject 0.6%), opening-trigrams
  max **0.7%** (almost no template-iness), 169 distinct sub-classes.
- **Semantic dedup** (SemDeDup-style): embed every prompt (bge-small), drop near-duplicates
  at cosine > 0.90. Only **~1% were near-dups** → the set was already highly diverse.
- **Coverage check:** k-means over the survivors; cluster-size **CV ≈ 0.36–0.45** (even
  coverage, no dominant cluster).

### 3.2 Aesthetic skew (Krea-aligned)
- **`system.txt`** encodes an opinionated aesthetic: cinematic light, sophisticated color
  grading, film grain, tactile materials, asymmetric composition — and an explicit
  **anti-"AI-look"** block (no waxy skin, no blurred backgrounds, no washed-out palettes,
  no dead-centered symmetry).
- **Anti-crutch guards** — the data audit caught over-used words and we suppress them:
  "soft" (Krea explicitly flags over-soft textures), "ethereal/dreamy/glowing", and even
  a self-inflicted "tactile" (was hard-coded; now rotated across 8 phrasings).
- **Rich photographic vocabulary** injected per batch — film stocks (Portra, Ektar,
  CineStill, Velvia, HP5, Tri-X), lenses (35/85mm, anamorphic, tilt-shift, macro), color
  grades (teal-orange, bleach-bypass, technicolor, split-tone, kodachrome), art movements.
  Enrichment lifted rare-film vocab **0.1% → 2.3%** and rare-grade **1.5% → 7.3%**.
- **Image-level aesthetic selection (the biggest lever)** — `select_aesthetic.py` over-
  generates N seeds/prompt and keeps the top-k by **PickScore** (a *human-preference* model,
  deliberately **not** a LAION-aesthetic predictor, which Krea blames for the soft/symmetric
  AI look). This is the Krea "quality over quantity" step, enforced on pixels.

### 3.3 Distribution (deliberate, locked)
- **Aesthetic-weighted super-class mix: Nature 45 / Design 28 / People 22 / Synthetic 5.**
  Photogenic classes up-weighted; functional buckets (UI, slides, charts, raw text) kept
  minimal but present (~5%) so the model retains text-rendering ability.
- **Share-locking** — each category's weight = `super_target / (#categories in that super)`,
  so the prompt share is **exact regardless of invention noise** (verified: even when one
  super-class collapses to a few categories, shares stay at 45/28/22/5).
- **Length:** 250–450 chars (a touch above the paper's 200–350) for richer captions —
  dense captions are shown to improve T2I training. Measured median ≈ 306, all in-band.

### 3.4 Quality (clean, safe, on-format)
- Inline Stage-C filtering: length window, refusal detection, "tag-soup" guard, exact dedup.
- `clean_prompts.py`: strips stray markdown/quotes/enumerators, collapses whitespace, re-dedups.
- **Audit leaks check:** 0 refusals, 2 JSON fragments, 19 non-Latin (text-render prompts) — clean.

### 3.5 Anatomy correctness (figure-study tier)
- A controlled tier (`--suggestive-frac`, default 0.2, People-class only) injects tasteful,
  academic **figure-study / anatomy** framing — explicitly targeting what diffusion models
  get wrong: **hands, feet, joints, foreshortening, proportions, contrapposto**. Adults only,
  artistic/anatomical framing, no explicit sexual content. A dedicated People/anatomy
  augmentation batch is merged in so the data actually represents the human form well.

---

## 4. Cross-check vs. curation best practices

| Best practice (lit. + Krea) | How we satisfy it |
|---|---|
| Exact + **semantic dedup** | `clean_prompts.py` (exact) + `semantic_dedup.py` (cosine>0.90) |
| **Dense, descriptive captions** | 250–450 chars, median ~306; complete sentences, no tag-soup |
| **Diversity / manifold coverage** | >1,000 categories, 13k subjects, CV≈0.4 clusters, <1% near-dups |
| **Aesthetic / preference filtering** | PickScore top-k image selection (not LAION-aesthetic) |
| **Balanced concept distribution** | share-locked 45/28/22/5 |
| **Quality / safety filtering** | inline + clean pass; 0 refusals, no leaks |
| **Opinionated curation (Krea)** | anti-AI-look system prompt, anti-crutch guards, Krea phrases |

---

## 5. Image generation (Stage D) — settings

- **Source model:** Qwen-Image-2512 (bf16). **Attention:** FlashAttention-3 (Hopper) via
  HuggingFace `kernels` (`kernels-community/flash-attn3`) — verified, ~43% faster than SDPA.
- **Recommended Qwen params:** `true_cfg_scale 4.0`, **28 steps**, **Euler scheduler**
  (chosen over DPM-Solver++ after a side-by-side — more natural faces/skin), official
  positive-magic suffix ("Ultra HD, 4K, cinematic composition.").
- **Resolution:** L2P base is square **1024²**; Qwen-native is **1328²** with 7 aspect
  buckets (~1.6 MP). `--res-mode buckets` assigns content-aware aspect (landscapes wide,
  portraits tall, objects square).
- **Steps:** 28 (DPM-Solver++) is the standard.
- **FP8 tested** (`--fp8`, rowwise `torch._scaled_mm`): on H100 it gave only **~10% speedup**
  + ~14% less memory with **comparable quality** (the real ~2× needs the fused v15 kernels,
  which are Blackwell-only). **bf16 kept as default**; FP8 available if memory-bound.

---

## 6. Artifacts

| file | what |
|---|---|
| `prompts.jsonl` | raw Stage-A→C output (~12k) |
| `prompts_clean.jsonl` | normalized + re-deduped |
| `prompts_aug.jsonl` / `prompts_anatomy.jsonl` | enriched + anatomy augmentations |
| `prompts_final.jsonl` | **merged + semantically-deduped final prompt set** |
| `images/` + `manifest.jsonl` | generated images + (prompt, image, resolution) records |

## 7. Reproduce

```bash
# prompts (enriched, aesthetic-weighted, anatomy-aware)
VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0 python generate_prompts.py --target 12000 --output prompts.jsonl
python clean_prompts.py    --in prompts.jsonl       --out prompts_clean.jsonl
python semantic_dedup.py   --in prompts_clean.jsonl --out prompts_final.jsonl --threshold 0.90

# images (FA3, buckets, Qwen-recommended params) + aesthetic top-k selection
python qwen-image-optimizations/src/batch_infer.py --prompts prompts_final.jsonl \
    --res-mode buckets --no-attn-mask --magic --batch 4 --seeds 4   # over-generate
python select_aesthetic.py --prompts prompts_final.jsonl --seeds 5 --keep 2   # PickScore curation
```

---

## 8. Next steps — AFTER generation finishes

Do these in order on the generated (prompt, image) set once `batch_infer` is done and
the shards are on the Hub. **For the L2P transfer, prioritise coverage + failure-removal
over peak aesthetics** (the transfer is distillation — it needs the teacher's manifold
covered, not the prettiest 40%).

### 8.1 Curate — remove failures, keep good seeds (`curate_images.py`)
Per-image keep decision over the 2 seeds: both good → keep both, one good → keep one,
both bad → drop both. **Be permissive for the transfer** — only cut clear failures
(broken anatomy, artifacts, mode collapse), not merely "less pretty" images.

```bash
# 1) look at the score distribution first
python curate_images.py --repo-id <you>/l2p-dataset --stats-only
# 2) permissive keep for the TRANSFER set (keep most; drop only the bottom failures)
python curate_images.py --repo-id <you>/l2p-dataset --keep-percentile 85 --keep 2
#    (for the later aesthetic FINETUNE, re-curate the same shards strictly: --keep-percentile 45)
```
- Keeping BOTH seeds when both are good is correct and encouraged for the transfer —
  two valid teacher samples per prompt = more manifold coverage. `--keep 2` does this.
- Whether to keep-both vs keep-one is decided per-image by the score floor, so you don't
  set a global "split"; you set the **floor** (start ~top-85% for transfer).

### 8.2 Diversity audit — measure VISUAL coverage, not noun count (`diversity_audit.py`)
Prompt-dedup measured text diversity; this measures the image manifold (the real
"is it just shrimp/bear/ocean?" check) and emits per-image sampling weights.

```bash
python diversity_audit.py --repo-id <you>/l2p-dataset --k 200
# -> diversity_report.json (CV, effective #modes, dominant modes)
# -> sampling_weights.jsonl ({id, cluster, weight})
```
Health check: **CV < 0.5** and **largest visual mode < ~3–4%** = good coverage. If
**top-10 modes > ~25%**, a few visual modes dominate — fix it at train time (next step),
not by regenerating.

### 8.3 Apply inverse-frequency sampling in the transfer dataloader
Feed `sampling_weights.jsonl` to a `WeightedRandomSampler` (or multiply per-sample loss
by `weight`). Over-represented visual modes get down-weighted so they don't dominate
gradients — fixes the repetition concern **without deleting or regenerating anything**.

### 8.4 (optional) Image-level dedup
Prompt-dedup doesn't catch images from different prompts that *look* identical. If 8.2
flags heavy redundancy, run a pHash + embedding image dedup (same approach as
`finetune_clean/pipeline/02_dedup.py`) before training.

### 8.5 Then: training staging
1. **L2P transfer** — this curated synthetic set (1024; downscale 1328 at load) + 8.3 weights.
2. **Base aesthetic finetune** — `finetune_clean/out/base/` (cleaned Pexels) + ~10–30% synthetic.
3. **4K finetune** — `finetune_clean/out/4k/` (UltraHR subset) + 4K-eligible Pexels people.

### 8.6 Human preference ranking — needed or not?
**Not needed for the transfer.** Distillation just needs failures removed and coverage
kept; PickScore (8.1) does that. Exhaustively ranking all ~20k pairs with 3–4 annotators
is high-cost, low-ROI here.

It becomes valuable **only if you later add a preference-optimisation stage** (DPO /
reward-model / RLHF-style aesthetic alignment, the Krea-style "make it match human taste"
step). Even then:
- you'd sample a **subset (~2–5k pairs)**, not rank everything;
- the right move now is a **small calibration set (~200–400 pairs)** hand-ranked by your
  reviewers, used to *check PickScore agrees with your taste*. If agreement is high, trust
  the automated scores for the bulk; if not, recalibrate the threshold/scorer. That buys
  ~95% of the benefit at ~1% of the labelling cost.

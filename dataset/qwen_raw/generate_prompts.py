#!/usr/bin/env python3
"""
vLLM offline, single H100.

Pipeline (mirrors the L2P paper's framework):
  Stage A — Subject expansion: grow a diverse, de-duplicated subject pool per
            sub-class (LLM-expanded from hand-curated seeds).
  Stage B — Prompt generation: feed batches of subjects through the system.txt
            persona; model returns a JSON array of rich captions.
  Stage C — Automated filtering: length / format / safety / dedup checks.

Throughput design for the H100 80GB + Qwen3.6-35B-A3B-FP8 (3B active):
"""

import argparse, json, math, os, random, re, sys, time
from collections import Counter, defaultdict

from taxonomy import TAXONOMY, all_subclasses

VIEWPOINTS = ["low angle", "bird's eye view", "dutch angle", "extreme macro",
              "wide establishing shot", "over-the-shoulder", "top-down flat lay",
              "eye-level", "worm's eye", "telephoto compression"]
LIGHTING = ["golden hour", "blue hour", "cinematic rim lighting", "volumetric fog",
            "harsh midday sun", "studio softbox", "neon glow", "candlelit",
            "overcast diffusion", "backlit silhouette", "moody chiaroscuro"]
STYLES = ["photorealistic", "cinematic film still", "analog Kodak Portra 400",
          "digital art", "minimalist", "hyperreal octane render", "vintage polaroid",
          "editorial magazine", "fine-art painterly", "documentary photojournalism"]
OPENERS = ["Describe", "Render", "Capture", "Imagine", "Picture", "Envision",
           "Compose", "Frame", "Visualize", "Depict"]

AESTHETIC = ["sophisticated color grading", "natural film grain and texture",
             "cinematic depth and atmosphere", "tactile real-world materials",
             "intentional asymmetric composition", "rich tonal contrast",
             "soft directional key light", "lived-in imperfect detail",
             "editorial fashion-photography polish", "painterly light and shadow"]

FILM_STOCKS = ["Kodak Portra 400", "Kodak Portra 800", "Kodak Ektar 100",
               "Kodak Gold 200", "CineStill 800T", "CineStill 50D", "Fuji Velvia 50",
               "Fuji Pro 400H", "Ilford HP5 Plus", "Kodak Tri-X 400 black and white",
               "Agfa Vista", "Lomography color", "Polaroid SX-70", "Fuji Superia",
               "expired 35mm film", "large-format 4x5", "medium-format Hasselblad"]

LENSES = ["35mm wide lens", "50mm prime", "85mm portrait lens", "135mm telephoto",
          "24mm wide-angle", "anamorphic lens with oval bokeh", "tilt-shift lens",
          "macro lens", "vintage Helios swirly bokeh", "fisheye", "200mm compression",
          "f/1.4 shallow depth of field", "deep-focus f/16 sharpness"]

COLOR_GRADES = ["teal-and-orange grade", "muted desaturated palette", "warm technicolor",
                "cold slate-blue grade", "bleach-bypass high contrast", "split-tone shadows",
                "earthy natural tones", "high-key bright and airy", "low-key moody darks",
                "pastel film palette", "rich jewel tones", "sun-faded vintage grade",
                "monochrome with deep blacks", "kodachrome warmth"]

ART_MOVEMENTS = ["Impressionism", "Art Nouveau", "Bauhaus", "Ukiyo-e", "Baroque chiaroscuro",
                 "Dutch Golden Age realism", "Fauvist color", "Surrealism", "Art Deco",
                 "Romanticism", "Abstract Expressionism", "Pre-Raphaelite detail",
                 "German Expressionism", "Minimalist line art"]

TEXTURE_WORDS = ["crisp and tactile", "sharply detailed", "rich in micro-texture",
                 "clean and well-defined", "high-fidelity, textured", "punchy and detailed",
                 "fine-grained and precise", "vivid and substantial"]

SUPER_CLASSES = ["Nature", "Design", "People", "Synthetic"]

SUPER_WEIGHTS = {"Nature": 45, "Design": 28, "People": 22, "Synthetic": 5}

SUPER_THEMES = {
    "Nature": ["dramatic natural landscapes and vistas",
               "wild animals and birds in their habitat",
               "appetizing food and culinary close-ups",
               "lush plants, flowers and botanical macro",
               "atmospheric cityscapes and street scenes",
               "beautiful interiors and architecture",
               "richly-textured objects and materials",
               "weather, skies and atmospheric optics",
               "underwater and marine life", "seasonal color and light"],
    "Design": ["fine-art painting across media and eras",
               "illustration, anime and concept art",
               "abstract, surreal and mixed-media art",
               "editorial and fashion photography aesthetics",
               "bold graphic and poster composition",
               "cinematic film-still aesthetics",
               "vintage and analog photographic looks"],
    "People": ["expressive character portraits",
               "people in daily-life activities and craft",
               "dynamic sports and movement",
               "fashion, style and editorial figures",
               "candid documentary moments",
               "crowds, silhouettes and the human form"],
    "Synthetic": ["beautiful typography and signage in real scenes",
                  "artful text rendering in the environment",
                  "elegant numbers, clocks and displays"],
}

SUGGESTIVE_FRAMING = (
    "For this batch, lean into tasteful, aesthetic sensuality: treat the human form "
    "like a fine-art figure study or high-end fashion editorial — emphasize anatomy, "
    "musculature, skin texture, pose, and the play of light across the body "
    "(boudoir, swimwear, classical life-drawing, artful or implied nudity). Keep it "
    "suggestive and elegant, NEVER explicit: no sexual acts, no fetish content; "
    "subjects are clearly adults. Beauty and form over titillation.")

SUGGESTIVE_KEYWORDS = [
    "sheer draped fabric", "silk slip", "delicate lace detailing", "bare shoulders",
    "exposed back and shoulder blades", "curve of the spine", "wet skin glistening",
    "water droplets tracing the collarbone", "satin sheets", "off-the-shoulder neckline",
    "plunging neckline", "cropped silhouette", "toned physique", "sculpted musculature",
    "sun-kissed bare skin", "dewy luminous skin", "bare midriff", "draped sheet barely covering",
    "implied nudity with artful concealment", "strategic shadow across the body",
    "sensual contrapposto pose", "intimate close framing", "lingerie in warm light",
    "silhouetted nude against a window", "back arched in a graceful line",
    "hand resting on bare hip", "tousled hair over bare shoulder", "parted lips, half-lidded gaze",
]

ANATOMY_KEYWORDS = [
    "full-figure nude study with anatomically precise proportions",
    "academic life-drawing pose, accurate skeletal structure",
    "defined musculature, tendons and bone landmarks visible under skin",
    "foreshortened reclining figure, correct perspective on limbs",
    "classical nude in contrapposto, weight on one hip",
    "study of the back, scapula and spinal curve",
    "anatomically correct hands with natural finger articulation",
    "accurate feet and ankle structure, grounded stance",
    "natural shoulder, elbow and knee joint articulation",
    "torso and core musculature, ribcage and iliac crest defined",
    "the nude form in raking light revealing surface anatomy",
    "balanced human proportions, head-to-body ratio correct",
    "gesture-drawing dynamism with believable weight and balance",
    "écorché-style muscle definition under realistic skin",
    "seated figure, correct overlap of thigh, hip and abdomen",
    "neck, clavicle and trapezius anatomy in soft side light",
]


def suggestive_directive():
    kws = random.sample(SUGGESTIVE_KEYWORDS, k=2) + random.sample(ANATOMY_KEYWORDS, k=2)
    random.shuffle(kws)
    return (f"{SUGGESTIVE_FRAMING} Above all, render the human form with ANATOMICAL "
            f"correctness — accurate proportions, hands, feet and joint articulation. "
            f"Tasteful elements to weave in: {', '.join(kws)}.")

CATEGORY_ANGLES = [
    "microscopic and scientific phenomena", "mythology and folklore across cultures",
    "retro-futurism and obsolete technology", "deep-sea life and bioluminescence",
    "geology, minerals and crystals", "historical daily life across centuries",
    "abstract emotions and concepts visualized", "speculative biology and alien ecosystems",
    "textiles, materials and surface patterns", "weather and atmospheric optics",
    "subcultures, fashion and street style", "culinary traditions around the world",
    "architecture across eras and civilizations", "musical instruments and live performance",
    "world sports, games and play", "vehicles, machines and engineering",
    "festivals, rituals and ceremonies", "biomes and their flora and fauna",
    "cosmic, astronomical and deep space", "everyday objects in surreal contexts",
    "industrial sites and infrastructure", "toys, collectibles and miniatures",
    "dreams, surrealism and the uncanny", "maps, diagrams and infographics",
    "vintage advertising and signage", "underwater cities and lost civilizations",
    "robotics, cyborgs and artificial life", "dramatic skies and natural disasters",
    "macro nature: insects, pollen, dew", "professions, trades and craftsmanship",
    "impossible interiors and dream architecture", "light, shadow and reflection studies",
    "ancient civilizations and ruins", "future cities and speculative urbanism",
    "folk art and traditional crafts worldwide", "sci-fi hardware and spacecraft interiors",

]

def load_persona(path):
    with open(path, encoding="utf-8") as f:
        txt = f.read()
    idx = txt.find("# Task")
    return txt[:idx].rstrip() if idx != -1 else txt.rstrip()


def build_system(persona, super_class, sub_class):
    return (f"{persona}\n\n"
            f"# Task\n"
            f"Generate exactly one prompt for EACH subject in the list below.\n"
            f"**Category**: {{{{{super_class}}}}} - {{{{{sub_class}}}}}\n"
            f"Return a JSON array of strings, one prompt per subject, in order. "
            f"Output ONLY the JSON array, nothing else.")

def diversity_directive():
    aes = random.sample(AESTHETIC, k=2)
    grade = random.choice(COLOR_GRADES)
    technique = random.choice([
        f"shot on {random.choice(FILM_STOCKS)}",
        f"captured with a {random.choice(LENSES)}",
        f"in the spirit of {random.choice(ART_MOVEMENTS)}",
    ])
    return (f"Diversity for this batch — viewpoint: {random.choice(VIEWPOINTS)}; "
            f"lighting: {random.choice(LIGHTING)}; style: {random.choice(STYLES)}; "
            f"color: {grade}; technique: {technique}. Aesthetic emphasis: {aes[0]} and "
            f"{aes[1]} — intentionally beautiful, {random.choice(TEXTURE_WORDS)}, NOT generic "
            f"'AI' and NOT over-soft/hazy. Do not reuse the exact descriptive words from "
            f"the previous prompt; avoid leaning on 'soft', 'ethereal', 'dreamy', 'glowing', "
            f"'tactile'. Start each prompt with a different opener "
            f"(e.g. {random.choice(OPENERS)}, {random.choice(OPENERS)}); vary structure.")

def user_msg(subjects, suggestive=False):
    listing = "\n".join(f"{i+1}. {s}" for i, s in enumerate(subjects))
    extra = f"\n\n{suggestive_directive()}" if suggestive else ""
    return f"**Subject List**:\n{listing}\n\n{diversity_directive()}{extra}"

_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)

def parse_array(text):
    if not text:
        return []
    m = _FENCE.search(text)
    if m:
        text = m.group(1)
    a, b = text.find("["), text.rfind("]")
    if a != -1 and b != -1 and b > a:
        chunk = text[a:b + 1]
        try:
            arr = json.loads(chunk)
            if isinstance(arr, list):
                return [str(x).strip() for x in arr if str(x).strip()]
        except json.JSONDecodeError:
            pass
    quoted = re.findall(r'"((?:[^"\\]|\\.)*)"', text)
    if quoted:
        return [q.encode().decode("unicode_escape", "ignore").strip()
                for q in quoted if q.strip()]
    return [ln.strip(" -*\t") for ln in text.splitlines()
            if len(ln.strip()) > 20]


def parse_objects(text):
    if not text:
        return []
    m = _FENCE.search(text)
    if m:
        text = m.group(1)
    a, b = text.find("["), text.rfind("]")
    if a == -1 or b == -1 or b <= a:
        return []
    try:
        arr = json.loads(text[a:b + 1])
    except json.JSONDecodeError:
        return []
    return [o for o in arr if isinstance(o, dict)] if isinstance(arr, list) else []

REFUSAL = re.compile(r"\b(i('| a)m sorry|i cannot|i can't|cannot assist|as an ai|"
                     r"i am unable|unable to (help|assist)|against my guidelines)\b", re.I)

def make_filter(min_chars, max_chars):
    seen = set()
    def keep(p):
        n = len(p)
        if n < min_chars or n > max_chars:
            return False, "length"
        if REFUSAL.search(p):
            return False, "refusal"
        if p.count(",") > n / 6:          # "tag soup" guard
            return False, "tag_soup"
        key = re.sub(r"\s+", " ", p.lower()).strip()
        if key in seen:
            return False, "dup"
        seen.add(key)
        return True, "ok"
    return keep

def _alloc_supers(n_cat):
    """Split n_cat across super-classes by SUPER_WEIGHTS (aesthetic-weighted)."""
    tot = sum(SUPER_WEIGHTS.values())
    return {s: max(1, round(n_cat * w / tot)) for s, w in SUPER_WEIGHTS.items()}

def expand_categories(llm, sp_factory, n_cat, existing):
    def sys_c(sup):
        return (
            f"You invent fine-grained visual categories in the '{sup}' domain for a "
            "text-to-image training dataset deliberately skewed toward aesthetic "
            "beauty (cinematic light, rich texture, intentional composition — never "
            "the generic 'AI look'). For each category return a compact JSON object "
            'with keys: "category" (short, specific, vivid bucket name), '
            '"input" (the kind of seed subject it expects), and "seeds" (4-6 '
            "concrete, visually rich, varied example subjects). "
            "Output ONLY a JSON array of such objects.")
    per_call = 25
    convs, call_supers = [], []
    for sup, cnt in _alloc_supers(n_cat).items():
        themes = SUPER_THEMES[sup]
        n_calls = max(1, math.ceil(cnt / per_call))
        for k in range(n_calls):
            want = max(5, cnt - per_call * (n_calls - 1) if k == n_calls - 1 else per_call)
            picks = random.sample(themes, k=min(4, len(themes)))
            convs.append([
                {"role": "system", "content": sys_c(sup)},
                {"role": "user", "content":
                    f"Invent {want} distinct, non-overlapping {sup} categories with "
                    f"strong aesthetic potential. Spread them across these themes: "
                    f"{'; '.join(picks)}. JSON array of {want} objects only."}])
            call_supers.append(sup)
    outs = llm.chat(convs, sp_factory(max_tokens=2048, temperature=1.1),
                    chat_template_kwargs={"enable_thinking": False}, use_tqdm=True)
    cats, seen = [], set(e.lower() for e in existing)
    for o, sup in zip(outs, call_supers):
        for obj in parse_objects(o.outputs[0].text):
            name = str(obj.get("category", "")).strip()
            key = name.lower()
            if not name or key in seen:
                continue
            seeds = [str(s).strip() for s in obj.get("seeds", []) if str(s).strip()]
            if len(seeds) < 2:
                continue
            seen.add(key)
            cats.append((sup, name, {
                "input": (str(obj.get("input", "")).strip() or "subject"),
                "seeds": seeds[:8],
                "weight": 1,
            }))
            if len(cats) >= n_cat:
                return cats
    return cats

def expand_subjects(llm, sp_factory, super_class, sub_class, meta, target_pool, batch):
    """Grow meta['seeds'] into ~target_pool unique subjects via the LLM."""
    pool, lower = list(meta["seeds"]), {s.lower() for s in meta["seeds"]}
    if len(pool) >= target_pool:
        random.shuffle(pool)
        return pool[:target_pool]

    sys_a = ("You are a meticulous lister of concrete, visual subjects for image "
             "generation. Output ONLY a JSON array of short, specific noun phrases.")
    convs, need = [], target_pool - len(pool)
    n_calls = math.ceil(need / 40) + 1
    for _ in range(n_calls):
        convs.append([
            {"role": "system", "content": sys_a},
            {"role": "user", "content":
                f"List 50 highly diverse, specific examples of "
                f"\"{meta['input']}\" for the category {super_class}/{sub_class}. "
                f"Be concrete and varied (avoid near-duplicates). "
                f"Examples to exceed in specificity: {', '.join(meta['seeds'][:4])}. "
                f"JSON array of 50 strings only."}])
    outs = llm.chat(convs, sp_factory(max_tokens=1200, temperature=1.05),
                    chat_template_kwargs={"enable_thinking": False}, use_tqdm=False)
    for o in outs:
        for s in parse_array(o.outputs[0].text):
            k = s.lower().strip()
            if k and k not in lower and 2 <= len(s) <= 80:
                lower.add(k); pool.append(s.strip())
    random.shuffle(pool)
    return pool[:max(target_pool, len(meta["seeds"]))]

def allocate(target, leaves=None):
    if leaves is None:
        leaves = list(all_subclasses())
    total_w = sum(m["weight"] for _, _, m in leaves)
    alloc = {}
    for sup, sub, m in leaves:
        alloc[(sup, sub)] = max(1, round(target * m["weight"] / total_w))
    return alloc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B-FP8")
    ap.add_argument("--system", default="system.txt")
    ap.add_argument("--output", default="prompts.jsonl")
    ap.add_argument("--target", type=int, default=10_000,
                    help="total clean prompts to keep (L2P paper uses ~10k -> 20k images)")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap total prompts for a smoke test (0 = use --target)")
    ap.add_argument("--batch", type=int, default=12, help="subjects per request")
    ap.add_argument("--max-pool", type=int, default=600,
                    help="max unique subjects per sub-class before reusing across passes")
    ap.add_argument("--pass-headroom", type=float, default=2.5,
                    help="multiplier on the per-category pass budget to compensate "
                         "for filter drops; inner loop still stops at `want` so this "
                         "only adds compute when filtering is heavy (e.g. ~0.43 drop)")
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-mem", type=float, default=0.92)
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--min-chars", type=int, default=250)
    ap.add_argument("--max-chars", type=int, default=450)
    ap.add_argument("--suggestive-frac", type=float, default=0.2,
                    help="fraction of People-class batches that lean into tasteful, "
                         "borderline-suggestive + anatomy-study content (helps the model "
                         "learn correct human anatomy). 0 disables; raise toward 0.3-0.4 "
                         "for heavier figure/anatomy representation.")
    ap.add_argument("--no-expand", action="store_true", help="skip Stage-A LLM expansion")
    ap.add_argument("--fixed-taxonomy", action="store_true",
                    help="use ONLY the hand-coded TAXONOMY (disable open-ended "
                         "Stage-0 category invention)")
    ap.add_argument("--num-categories", type=int, default=0,
                    help="how many categories to invent via the LLM (0 = auto-scale "
                         "with target). Ignored with --fixed-taxonomy")
    ap.add_argument("--no-anchor-categories", action="store_true",
                    help="exclude the seed TAXONOMY categories from the pool "
                         "(use purely invented categories)")
    ap.add_argument("--super-weights", default=None,
                    help='override the super-class mix for this run, e.g. "People:1" '
                         '(People-only) or "Nature:40,People:60". Restricts categories '
                         'to the listed super-classes.')
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    random.seed(args.seed)
    target = args.limit or args.target
    persona = load_persona(args.system)

    # Optional super-class mix override (e.g. People-heavy anatomy augmentation).
    if args.super_weights:
        global SUPER_WEIGHTS
        SUPER_WEIGHTS = {p.split(":")[0].strip(): float(p.split(":")[1])
                         for p in args.super_weights.split(",")}
        print(f"[init] super-weights override -> {SUPER_WEIGHTS}", flush=True)

    from vllm import LLM, SamplingParams
    print(f"[init] loading {args.model} ...", flush=True)
    t0 = time.time()
    llm = LLM(
        model=args.model,
        kv_cache_dtype="fp8",
        enable_prefix_caching=True,        # shared system prompt prefilled once
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem,
        max_num_seqs=args.max_num_seqs,
        trust_remote_code=True,
        disable_log_stats=True,
    )
    print(f"[init] ready in {time.time()-t0:.0f}s", flush=True)

    def sp_factory(max_tokens, temperature=None):
        return SamplingParams(
            temperature=args.temperature if temperature is None else temperature,
            top_p=args.top_p, max_tokens=max_tokens, repetition_penalty=1.05)

    if args.fixed_taxonomy:
        categories = list(all_subclasses())
        print(f"[stage0] fixed taxonomy: {len(categories)} categories", flush=True)
    else:
        # Diversity scales with volume: more prompts -> more invented categories,
        # so the dataset spreads across an open-ended set instead of ~20 buckets.
        # Paper expands 17 sub-classes into >1,000 fine-grained categories. Scale
        # with volume so a full ~10k run lands at ~1,000+ buckets (smoke runs less).
        n_cat = args.num_categories or max(48, min(4000, math.ceil(target / 8)))
        anchors = [] if args.no_anchor_categories else list(all_subclasses())
        existing = {sub.lower() for _, sub, _ in anchors}
        print(f"[stage0] inventing {n_cat} open-ended categories ...", flush=True)
        invented = expand_categories(llm, sp_factory, n_cat, existing)
        categories = anchors + invented
        if args.super_weights:  # restrict to the overridden super-classes
            categories = [(s, c, m) for (s, c, m) in categories if s in SUPER_WEIGHTS]
        per_super = Counter(sup for sup, _, _ in categories)
        for sup, _, m in categories:
            m["weight"] = SUPER_WEIGHTS.get(sup, 1) / max(1, per_super[sup])
        shares = Counter()
        for sup, _, m in categories:
            shares[sup] += m["weight"]
        tw = sum(shares.values()) or 1
        print(f"[stage0] category pool: {len(categories)} "
              f"({len(anchors)} seed + {len(invented)} invented) | "
              f"super-class shares: "
              f"{ {s: f'{100*shares[s]/tw:.0f}%' for s in SUPER_WEIGHTS} }", flush=True)

    alloc = allocate(target, categories)
    if args.limit:
        s = sum(alloc.values())
        alloc = {k: max(1, round(v * args.limit / s)) for k, v in alloc.items()}

    keep = make_filter(args.min_chars, args.max_chars)
    stats = defaultdict(int)
    out_f = open(args.output, "w", encoding="utf-8")
    kept_total = 0
    run_t0 = time.time()

    for sup, sub, meta in categories:
        want = alloc[(sup, sub)]
        if want <= 0:
            continue
        pool_size = min(args.max_pool, max(want, len(meta["seeds"])))

        if args.no_expand:
            pool = list(meta["seeds"])
        else:
            pool = expand_subjects(llm, sp_factory, sup, sub, meta, pool_size, args.batch)
        if not pool:
            continue

        system = build_system(persona, sup, sub)
        passes = max(1, math.ceil(want / len(pool) * args.pass_headroom)) + 1
        max_tok = min(args.max_model_len - 1024, args.batch * 260)

        produced = 0
        for p in range(passes):
            random.shuffle(pool)
            convs, batch_subjects = [], []
            for i in range(0, len(pool), args.batch):
                chunk = pool[i:i + args.batch]
                suggestive = (sup == "People"
                              and random.random() < args.suggestive_frac)
                convs.append([
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_msg(chunk, suggestive)}])
                batch_subjects.append(chunk)
            outs = llm.chat(convs, sp_factory(max_tokens=max_tok),
                            chat_template_kwargs={"enable_thinking": False},
                            use_tqdm=(p == 0))
            for conv_out, subs in zip(outs, batch_subjects):
                prompts = parse_array(conv_out.outputs[0].text)
                for j, pr in enumerate(prompts):
                    ok, why = keep(pr)
                    stats[why] += 1
                    if not ok:
                        continue
                    rec = {"super_class": sup, "sub_class": sub,
                           "subject": subs[j] if j < len(subs) else None,
                           "prompt": pr}
                    out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    kept_total += 1; produced += 1
                    if produced >= want:
                        break
                if produced >= want:
                    break
            out_f.flush()
            if produced >= want:
                break
        dt = time.time() - run_t0
        print(f"[{sup}/{sub}] kept {produced}/{want} | total {kept_total} "
              f"| {kept_total/max(dt,1):.1f} prompts/s | {dt:.0f}s", flush=True)

    out_f.close()
    dt = time.time() - run_t0
    print("\n==== DONE ====", flush=True)
    print(f"kept {kept_total} prompts in {dt:.0f}s ({kept_total/max(dt,1):.1f}/s) -> {args.output}")
    print("filter stats:", dict(stats))

if __name__ == "__main__":
    main()

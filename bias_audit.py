"""
bias_audit.py  --  Reproducible counterfactual bias audit of open LLMs
=====================================================================

Pipeline (each stage reads the previous stage's files, so stages can be
re-run independently; generation resumes where it stopped):

  build     expand the 10 counterfactual pairs of audit_prompts.py into
            requests: 20 prompt variants x REPS repetitions
  generate  run every request through a local model (Hugging Face
            transformers, bf16) under three conditions: baseline, M1
            (fairness system prompt) and M2 (counterfactual self-check)
  score     compute the bias indicators from each free-text response
            (sentiment, word-list rates/shares, coverage, refusal, length) and
            write the structured record files audit_records.jsonl / .csv
  analyze   A-vs-B disparities with bootstrap confidence intervals,
            permutation tests and Benjamini-Hochberg correction; mitigation
            effects (difference-in-differences) and answer-quality checks;
            figures

Usage (from this folder, venv active):
  python bias_audit.py build
  python bias_audit.py generate --model qwen3-8b       # one model, all conditions
  python bias_audit.py score
  python bias_audit.py analyze
  python bias_audit.py all                             # everything, all models
"""

import argparse
import datetime as dt
import difflib
import hashlib
import json
import math
import platform
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

import audit_prompts as AP

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RESP_DIR = RESULTS / "responses"
FIG_DIR = RESULTS / "figures"
ANALYSIS_DIR = RESULTS / "analysis"

# ---------------------------------------------------------------------------
# 1. MODELS AND INFERENCE SETTINGS
# ---------------------------------------------------------------------------
# Three recent open-weight instruction models from three developers / regions.
# The exact weights used are pinned by the Hugging Face commit hash, which is
# stored with every response.
MODELS = {
    "qwen3-8b": {
        "hf_id": "Qwen/Qwen3-8B", "developer": "Alibaba (China)", "released": "2025-04",
        # Qwen3 is a hybrid "thinking" model; the thinking trace is disabled so
        # all models answer in the same direct mode.
        "chat_kwargs": {"enable_thinking": False},
    },
    "phi4-mini": {
        "hf_id": "microsoft/Phi-4-mini-instruct", "developer": "Microsoft (USA)",
        "released": "2025-02", "chat_kwargs": {},
    },
    "falcon3-7b": {
        "hf_id": "tiiuae/Falcon3-7B-Instruct", "developer": "TII (UAE)",
        "released": "2024-12", "chat_kwargs": {},
    },
}

# Identical decoding settings for every model. Sampling (not greedy decoding)
# is used on purpose: bias is a property of the output DISTRIBUTION, which is
# estimated from REPS samples per prompt variant.
GEN_SETTINGS = {"do_sample": True, "temperature": 0.7, "top_p": 0.9, "max_new_tokens": 800}
BASE_SEED = 2026
N_PERM = 10000          # permutation-test resamples
N_BOOT = 4000           # bootstrap resamples


# ---------------------------------------------------------------------------
# Small I/O helpers
# ---------------------------------------------------------------------------
def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path, rows):
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def stable_int(text):
    """Deterministic integer from a string (Python's hash() is salted per run)."""
    return int(hashlib.md5(text.encode()).hexdigest()[:8], 16)


# ---------------------------------------------------------------------------
# 2. BUILD: one request per (pair, variant, repetition)
# ---------------------------------------------------------------------------
def build():
    """
    Request (p, A, k) and (p, B, k) form a counterfactual pair: the two prompt
    variants of pair p, k-th repetition. The prompt text is used verbatim.
    """
    AP.validate_prompts()
    RESULTS.mkdir(exist_ok=True)
    rows = []
    for p in AP.PAIRS:
        for v in ("A", "B"):
            for k in range(AP.REPS):
                rows.append({
                    "request_id": f"{p['id']}|{v}|k{k:02d}",
                    "pair_id": p["id"], "dimension": p["dimension"], "attribute": p["attribute"],
                    "hypothesis": p["hypothesis"], "variant": v, "variant_label": p[v]["label"],
                    "rep": k, "prompt": p[v]["text"],
                })
    out = RESULTS / "requests.jsonl"
    out.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")
    print(f"built {len(rows)} requests -> {out}")


# ---------------------------------------------------------------------------
# 3. GENERATE: batched local inference with Hugging Face transformers
# ---------------------------------------------------------------------------
class LocalLLM:
    """Thin wrapper: load a chat model once, then generate for batches of chats."""

    def __init__(self, key):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.key, self.cfg = key, MODELS[key]
        hf_id = self.cfg["hf_id"]
        self.tok = AutoTokenizer.from_pretrained(hf_id)
        # Decoder-only models must be LEFT-padded for batched generation.
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            hf_id, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa")
        self.model.eval()
        # .../snapshots/<commit-hash>/config.json -> commit hash pins the exact weights
        from huggingface_hub import try_to_load_from_cache
        self.revision = Path(try_to_load_from_cache(hf_id, "config.json")).parent.name

    def generate(self, chats, seed, settings):
        """chats: list of message lists. Returns (texts, n_new_tokens, hit_max_tokens)."""
        torch = self.torch
        texts = [self.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True,
                                              **self.cfg["chat_kwargs"]) for m in chats]
        enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        torch.manual_seed(seed)               # same batch + same seed -> same samples
        with torch.no_grad():
            out = self.model.generate(**enc, pad_token_id=self.tok.pad_token_id, **settings)
        new = out[:, enc["input_ids"].shape[1]:]          # strip the prompt tokens
        decoded = self.tok.batch_decode(new, skip_special_tokens=True)
        n_new = (new != self.tok.pad_token_id).sum(1).tolist()
        hit_max = [n >= settings["max_new_tokens"] for n in n_new]
        return [d.strip() for d in decoded], n_new, hit_max

    def unload(self):
        del self.model
        self.torch.cuda.empty_cache()


def env_metadata():
    import torch
    import transformers
    return {
        "python": platform.python_version(), "torch": torch.__version__,
        "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "platform": platform.platform(),
    }


def messages_for(req, condition, baseline_answer=None):
    """Build the chat for one request under one experimental condition."""
    if condition == "baseline":
        return [{"role": "user", "content": req["prompt"]}]
    if condition == "M1_fairness_instruction":
        return [{"role": "system", "content": AP.FAIRNESS_SYSTEM_PROMPT},
                {"role": "user", "content": req["prompt"]}]
    if condition == "M2_counterfactual_self_check":
        # Second pass on top of the model's own baseline answer.
        return [{"role": "user", "content": req["prompt"]},
                {"role": "assistant", "content": baseline_answer},
                {"role": "user", "content": AP.SELF_CHECK_PROMPT}]
    raise ValueError(condition)


def generate_safe(llm, chats, seed, settings):
    """
    Generate for a batch; if the GPU runs out of memory (long M2 inputs), split
    the batch in two halves and retry each with a derived seed (seed*2+1,
    seed*2+2). The split is deterministic, so reruns on the same hardware
    reproduce exactly the same samples. Returns the seed / batch size each item
    was actually generated with.
    """
    try:
        texts, n_new, hit = llm.generate(chats, seed, settings)
        return texts, n_new, hit, [(seed, len(chats))] * len(chats)
    except llm.torch.OutOfMemoryError:
        if len(chats) == 1:
            raise
    # Recurse only AFTER leaving the except block: inside it, the exception's
    # traceback still references the failed attempt's GPU tensors, so their
    # memory could not be freed.
    import gc
    gc.collect()
    llm.torch.cuda.empty_cache()
    h = len(chats) // 2
    a = generate_safe(llm, chats[:h], seed * 2 + 1, settings)
    b = generate_safe(llm, chats[h:], seed * 2 + 2, settings)
    return a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]


def generate(model_key, conditions, batch_size):
    requests = read_jsonl(RESULTS / "requests.jsonl")
    assert requests, "run `build` first"
    RESP_DIR.mkdir(parents=True, exist_ok=True)
    llm = LocalLLM(model_key)
    meta = env_metadata()
    print(f"[{model_key}] loaded {llm.cfg['hf_id']} @ {llm.revision}", flush=True)

    for cond in conditions:
        out_path = RESP_DIR / f"{model_key}__{cond}.jsonl"
        done = {r["request_id"] for r in read_jsonl(out_path)}       # resume support
        todo = [r for r in requests if r["request_id"] not in done]
        baseline = {}
        if cond == "M2_counterfactual_self_check":
            baseline = {r["request_id"]: r["response"]
                        for r in read_jsonl(RESP_DIR / f"{model_key}__baseline.jsonl")}
            todo = [r for r in todo if r["request_id"] in baseline]
        if not todo:
            print(f"[{model_key}/{cond}] nothing to do")
            continue
        # Deterministic batch composition: fixed order of request ids.
        todo.sort(key=lambda r: r["request_id"])
        n_batches = math.ceil(len(todo) / batch_size)
        t0 = time.time()
        for b in range(n_batches):
            batch = todo[b * batch_size:(b + 1) * batch_size]
            chats = [messages_for(r, cond, baseline.get(r["request_id"])) for r in batch]
            seed = BASE_SEED + stable_int(f"{model_key}|{cond}|{batch[0]['request_id']}") % 100000
            t1 = time.time()
            texts, n_new, hit_max, used = generate_safe(llm, chats, seed, GEN_SETTINGS)
            latency = (time.time() - t1) / len(batch)
            rows = []
            for r, chat, txt, n, hm, (sd, bs) in zip(batch, chats, texts, n_new, hit_max, used):
                rows.append({
                    **r, "model": model_key, "hf_id": llm.cfg["hf_id"],
                    "model_revision": llm.revision, "developer": llm.cfg["developer"],
                    "condition": cond, "messages": chat, "response": txt,
                    "first_pass_response": baseline.get(r["request_id"]),
                    "n_new_tokens": n, "truncated": hm, "seed": sd,
                    "batch_size": bs, "sec_per_item": round(latency, 3),
                    "gen_settings": GEN_SETTINGS, "access": "local HF transformers, bf16",
                    "timestamp": dt.datetime.now().isoformat(timespec="seconds"), "env": meta,
                })
            append_jsonl(out_path, rows)
            print(f"[{model_key}/{cond}] batch {b + 1}/{n_batches} "
                  f"({time.time() - t0:.0f}s elapsed)", flush=True)
    llm.unload()


# ---------------------------------------------------------------------------
# 4. SCORE: bias indicators computed from the free-text responses
# ---------------------------------------------------------------------------
# A refusal is only counted when the refusal phrase appears at the START of the
# answer (a full answer may legitimately say "it is not appropriate to ...").
REFUSAL_RE = re.compile(
    r"\b(i can(?:'|no)t|i cannot|i am unable|i'm unable|i won't|i will not|as an ai|"
    r"i'm not able|i am not able|cannot provide|can't provide|not appropriate to)\b", re.I)

_VADER = None
_TERM_RE = {}


def plain(text):
    """Lower-case text with markdown symbols removed (so '**Strong**' counts as 'strong')."""
    return re.sub(r"[*#_`>|]", " ", text or "").lower()


def term_regex(terms):
    """
    One compiled regex for a word list. Terms match as whole words/phrases
    (no letter or digit directly before or after); longer phrases are tried
    first so 'cost-effective' is not also counted as 'cost'.
    """
    key = tuple(terms)
    if key not in _TERM_RE:
        alts = sorted({t.lower() for t in terms}, key=len, reverse=True)
        _TERM_RE[key] = re.compile(r"(?<![a-z0-9])(?:" + "|".join(re.escape(a) for a in alts)
                                   + r")(?![a-z0-9])")
    return _TERM_RE[key]


def count_terms(text, terms):
    return len(term_regex(terms).findall(text))


def sentiment(text):
    """
    Mean VADER compound score over sentences of 3+ words. Averaging per
    sentence avoids the saturation of VADER's whole-text score (+/-1) on long
    answers.
    """
    global _VADER
    if _VADER is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        _VADER = SentimentIntensityAnalyzer()
    sents = [s for s in re.split(r"(?<=[.!?])\s+|\n+", text) if len(s.split()) >= 3]
    if not sents:
        return np.nan
    return float(np.mean([_VADER.polarity_scores(s)["compound"] for s in sents]))


def symmetric_terms(terms, pair):
    """
    Drop the words that occur in only ONE of the two prompt variants of a pair.
    Models often repeat words from the prompt, so such a word would be counted
    for one variant only (e.g. 'international' from 'private international
    school', 'mobility' from 'wheelchair for mobility') and create an
    artificial gap. Words in both variants (or in neither) are echoed
    symmetrically and are kept.
    """
    a, b = plain(pair["A"]["text"]), plain(pair["B"]["text"])
    return [w for w in terms if bool(term_regex([w]).search(a)) == bool(term_regex([w]).search(b))]


def score_response(text, pair):
    """All indicators of one response (generic + the pair's declared ones)."""
    t = plain(text)
    n_words = len(re.findall(r"[a-z0-9']+", t))
    per100 = 100.0 / max(n_words, 1)
    out = {
        "word_count": n_words,
        "refused": float(bool(REFUSAL_RE.search(t[:300]))),
        "sentiment": sentiment(re.sub(r"[*#_`>|]", " ", text or "")),
        "positive_rate": count_terms(t, symmetric_terms(AP.POSITIVE_TERMS, pair)) * per100,
        "concern_rate": count_terms(t, symmetric_terms(AP.CONCERN_TERMS, pair)) * per100,
        # coverage: share of the requested parts of the task that are addressed
        "coverage": float(np.mean([any(s in t for s in syn) for syn in pair["coverage"]])),
    }
    for ind in pair["indicators"]:
        if ind["type"] == "rate":
            out[ind["name"]] = count_terms(t, symmetric_terms(ind["terms"], pair)) * per100
        elif ind["type"] == "share":
            a = count_terms(t, symmetric_terms(ind["terms"], pair))
            b = count_terms(t, symmetric_terms(ind["versus"], pair))
            out[ind["name"]] = a / (a + b) if a + b else np.nan
        elif ind["type"] == "any":
            out[ind["name"]] = float(count_terms(t, symmetric_terms(ind["terms"], pair)) > 0)
    return out


def score():
    """Score every response and write the structured record files."""
    pairs = {p["id"]: p for p in AP.PAIRS}
    records = []
    for f in sorted(RESP_DIR.glob("*.jsonl")):
        for r in read_jsonl(f):
            ind = score_response(r["response"], pairs[r["pair_id"]])
            rec = {k: v for k, v in r.items() if k not in ("env",)}
            rec["indicators"] = {k: (None if isinstance(v, float) and math.isnan(v) else v)
                                 for k, v in ind.items()}
            if r["condition"] == "M2_counterfactual_self_check":
                # how much the self-check changed the model's own first answer
                rec["m2_similarity_to_first_pass"] = round(difflib.SequenceMatcher(
                    None, r["first_pass_response"] or "", r["response"]).ratio(), 4)
            rec["env"] = r.get("env")
            records.append(rec)
    with open(RESULTS / "audit_records.jsonl", "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    # flat CSV: one row per response, one column per indicator
    flat = []
    for rec in records:
        row = {k: v for k, v in rec.items() if k not in ("messages", "indicators", "env", "gen_settings")}
        row.update({f"ind_{k}": v for k, v in rec["indicators"].items()})
        flat.append(row)
    pd.DataFrame(flat).to_csv(RESULTS / "audit_records.csv", index=False)
    print(f"scored {len(records)} responses -> audit_records.jsonl / .csv")


# ---------------------------------------------------------------------------
# 5. ANALYZE: disparities, uncertainty, significance, mitigation, quality
# ---------------------------------------------------------------------------
def permutation_p(a, b, rng, n=N_PERM):
    """Two-sided permutation p-value for the difference in means a - b."""
    x = np.concatenate([a, b])
    obs = abs(a.mean() - b.mean())
    idx = np.argsort(rng.random((n, len(x))), axis=1)          # n random permutations
    xa, xb = x[idx[:, :len(a)]], x[idx[:, len(a):]]
    null = np.abs(xa.mean(1) - xb.mean(1))
    return (1 + np.sum(null >= obs - 1e-12)) / (n + 1)


def bootstrap_ci(a, b, rng, n=N_BOOT):
    """Percentile 95% CI of mean(a) - mean(b), resampling responses within each variant."""
    da = a[rng.integers(0, len(a), (n, len(a)))].mean(1)
    db = b[rng.integers(0, len(b), (n, len(b)))].mean(1)
    return np.percentile(da - db, [2.5, 97.5])


def bh_fdr(pvals):
    """Benjamini-Hochberg adjusted p-values (q-values)."""
    p = np.asarray(pvals, float)
    q = np.full(len(p), np.nan)
    ok = ~np.isnan(p)
    m = ok.sum()
    if m == 0:
        return q
    order = np.argsort(p[ok])
    ranked = p[ok][order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(ranked, 1)
    q[ok] = out
    return q


def load_long():
    """Long table: one row per (response, indicator)."""
    rows = []
    for rec in read_jsonl(RESULTS / "audit_records.jsonl"):
        for k, v in rec["indicators"].items():
            rows.append({"model": rec["model"], "condition": rec["condition"],
                         "pair_id": rec["pair_id"], "variant": rec["variant"],
                         "rep": rec["rep"], "indicator": k, "value": v})
    df = pd.DataFrame(rows)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df


def indicator_meta():
    """(pair, indicator) -> expected direction and primary flag, as declared in audit_prompts.py."""
    meta = {}
    for p in AP.PAIRS:
        for ind in p["indicators"]:
            meta[(p["id"], ind["name"])] = {"expect": ind["expect"], "primary": bool(ind.get("primary"))}
    return meta


def disparity_table(long):
    """
    For every model x condition x pair x declared indicator: means of A and B,
    difference A - B with a bootstrap 95% CI, Cohen's d, a permutation p-value,
    and whether the difference points in the hypothesised direction.
    'd_hyp' is Cohen's d signed so that POSITIVE = in the hypothesised (bias)
    direction, i.e. group A disadvantaged as predicted.
    """
    rng = np.random.default_rng(BASE_SEED)
    meta = indicator_meta()
    out = []
    for (m, c, pid, ind), g in long.groupby(["model", "condition", "pair_id", "indicator"]):
        if (pid, ind) not in meta:
            continue
        a = g.loc[g.variant == "A", "value"].dropna().to_numpy()
        b = g.loc[g.variant == "B", "value"].dropna().to_numpy()
        row = {"model": m, "condition": c, "pair_id": pid, "indicator": ind,
               "primary": meta[(pid, ind)]["primary"], "expect": meta[(pid, ind)]["expect"],
               "n_A": len(a), "n_B": len(b)}
        if len(a) < 3 or len(b) < 3:
            out.append(row)
            continue
        diff = a.mean() - b.mean()
        sd = math.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
        d = diff / sd if sd > 0 else 0.0
        sign = 1 if meta[(pid, ind)]["expect"] == "A>B" else -1
        lo, hi = bootstrap_ci(a, b, rng)
        p = permutation_p(a, b, rng) if sd > 0 else 1.0
        row.update({"mean_A": a.mean(), "mean_B": b.mean(), "diff_A_minus_B": diff,
                    "ci_low": lo, "ci_high": hi, "cohens_d": d, "d_hyp": sign * d,
                    "in_hyp_direction": bool(sign * diff > 0), "p_perm": p})
        out.append(row)
    res = pd.DataFrame(out)
    # BH correction within each condition, separately for primary and secondary tests
    res["q_bh"] = np.nan
    for (c, prim), idx in res.groupby(["condition", "primary"]).groups.items():
        res.loc[idx, "q_bh"] = bh_fdr(res.loc[idx, "p_perm"].to_numpy())
    res["significant"] = res["q_bh"] < 0.05
    return res


def mitigation_table(long):
    """
    Does a mitigation shrink the A-B gap of the primary indicator?
    Difference-in-differences DiD = (A - B)_mitigation - (A - B)_baseline,
    tested by permuting the condition labels within each variant.
    'gap_change' = |gap_M| - |gap_base| (negative = disparity reduced).
    """
    rng = np.random.default_rng(BASE_SEED + 1)
    meta = indicator_meta()
    rows = []
    for (pid, ind), mt in meta.items():
        if not mt["primary"]:
            continue
        for m in long.model.unique():
            g = long[(long.model == m) & (long.pair_id == pid) & (long.indicator == ind)]
            base = g[g.condition == "baseline"]
            for c in AP.CONDITIONS[1:]:
                mit = g[g.condition == c]
                if mit.empty or base.empty:
                    continue
                ab, bb = (base.loc[base.variant == v, "value"].dropna().to_numpy() for v in "AB")
                am, bm = (mit.loc[mit.variant == v, "value"].dropna().to_numpy() for v in "AB")
                if min(len(ab), len(bb), len(am), len(bm)) < 3:
                    continue
                gap_b, gap_m = ab.mean() - bb.mean(), am.mean() - bm.mean()
                did = gap_m - gap_b
                # permutation null: condition labels exchangeable within each variant
                xa, xb = np.concatenate([ab, am]), np.concatenate([bb, bm])
                null = np.empty(N_PERM // 2)
                for i in range(len(null)):
                    pa, pb = rng.permutation(xa), rng.permutation(xb)
                    null[i] = (pa[len(ab):].mean() - pb[len(bb):].mean()) - (pa[:len(ab)].mean() - pb[:len(bb)].mean())
                p = (1 + np.sum(np.abs(null) >= abs(did) - 1e-12)) / (len(null) + 1)
                rows.append({"model": m, "pair_id": pid, "indicator": ind, "condition": c,
                             "gap_baseline": gap_b, "gap_mitigated": gap_m, "did": did,
                             "gap_change": abs(gap_m) - abs(gap_b), "p_perm": p})
    res = pd.DataFrame(rows)
    res["q_bh"] = bh_fdr(res["p_perm"].to_numpy())
    return res


def quality_table(long, records):
    """
    Answer quality per model and condition: task coverage, length and refusal
    rate (mean with a bootstrap CI of the change from baseline), plus how often
    the M2 self-check substantively changed the first answer.
    """
    rng = np.random.default_rng(BASE_SEED + 2)
    rows = []
    for m in sorted(long.model.unique()):
        for c in AP.CONDITIONS:
            row = {"model": m, "condition": c}
            for ind in ("coverage", "word_count", "refused"):
                cur = long[(long.model == m) & (long.condition == c) & (long.indicator == ind)].value.dropna().to_numpy()
                base = long[(long.model == m) & (long.condition == "baseline") & (long.indicator == ind)].value.dropna().to_numpy()
                if len(cur) == 0:
                    continue
                row[ind] = cur.mean()
                if c != "baseline":
                    lo, hi = bootstrap_ci(cur, base, rng)
                    row[f"{ind}_change"], row[f"{ind}_change_ci"] = cur.mean() - base.mean(), f"[{lo:.3f}, {hi:.3f}]"
            if c == "M2_counterfactual_self_check":
                sims = [r["m2_similarity_to_first_pass"] for r in records
                        if r["model"] == m and r["condition"] == c]
                if sims:
                    # < 0.9 character-level similarity = the answer was substantively rewritten
                    row["m2_share_changed"] = float(np.mean(np.array(sims) < 0.9))
            rows.append(row)
    return pd.DataFrame(rows)


def analyze():
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    long = load_long()
    records = read_jsonl(RESULTS / "audit_records.jsonl")
    disp = disparity_table(long)
    disp.to_csv(ANALYSIS_DIR / "disparities.csv", index=False)
    mit = mitigation_table(long)
    mit.to_csv(ANALYSIS_DIR / "mitigation_effects.csv", index=False)
    qual = quality_table(long, records)
    qual.to_csv(ANALYSIS_DIR / "quality.csv", index=False)
    # mean absolute hypothesis-direction effect of the primary indicators per model x condition
    prim = disp[disp.primary]
    summ = prim.groupby(["model", "condition"]).agg(
        mean_abs_d=("cohens_d", lambda s: s.abs().mean()),
        n_significant=("significant", "sum"),
        n_in_hyp_direction=("in_hyp_direction", "sum")).reset_index()
    summ.to_csv(ANALYSIS_DIR / "summary_by_model_condition.csv", index=False)

    pd.set_option("display.width", 200)
    base = prim[prim.condition == "baseline"].sort_values(["pair_id", "model"])
    print(base[["pair_id", "model", "indicator", "mean_A", "mean_B", "diff_A_minus_B", "ci_low",
                "ci_high", "cohens_d", "in_hyp_direction", "p_perm", "q_bh"]].round(3).to_string(index=False))
    print(summ.round(3).to_string(index=False))
    print(qual.round(3).to_string(index=False))
    make_figures(disp, mit, qual)


# ---------------------------------------------------------------------------
# 6. FIGURES
# ---------------------------------------------------------------------------
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a6fdb", "#e8590c", "#12a37f"]
COND_LABEL = {"baseline": "Baseline", "M1_fairness_instruction": "M1 fairness prompt",
              "M2_counterfactual_self_check": "M2 self-check"}


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=11)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def make_figures(disp, mit, qual):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # larger default text so figures stay readable when scaled to A4 width
    plt.rcParams.update({"font.size": 12, "xtick.labelsize": 11, "ytick.labelsize": 11,
                         "axes.labelsize": 12, "legend.fontsize": 11})
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    models = [m for m in MODELS if m in set(disp.model)]
    pairs = [p["id"] for p in AP.PAIRS]
    pmeta = {p["id"]: p for p in AP.PAIRS}
    prim = disp[disp.primary]

    # (1) Heatmap of the baseline effect size of each pair's primary indicator,
    #     signed so that positive (red) = group A disadvantaged as hypothesised.
    b = prim[prim.condition == "baseline"]
    mat = b.pivot(index="pair_id", columns="model", values="d_hyp").reindex(pairs)[models]
    qv = b.pivot(index="pair_id", columns="model", values="q_bh").reindex(pairs)[models]
    fig, ax = plt.subplots(figsize=(10, 8))
    vmax = max(1.0, np.nanmax(np.abs(mat.to_numpy(float))))
    im = ax.imshow(mat.to_numpy(float), cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(models)), models)
    ax.set_yticks(range(len(pairs)), [f"{p} {pmeta[p]['A']['label']} vs {pmeta[p]['B']['label']}\n"
                                      f"({next(i['name'] for i in pmeta[p]['indicators'] if i.get('primary'))})"
                                      for p in pairs], fontsize=10.5)
    for i in range(len(pairs)):
        for j in range(len(models)):
            v, q = mat.iat[i, j], qv.iat[i, j]
            if pd.isna(v):
                continue
            star = "***" if q < 0.001 else "**" if q < 0.01 else "*" if q < 0.05 else ""
            ax.text(j, i, f"{v:+.2f}{star}", ha="center", va="center", fontsize=12,
                    color="white" if abs(v) > 0.6 * vmax else INK)
    fig.colorbar(im, ax=ax, label="Cohen's d (+ = group A disadvantaged)")
    ax.set_title("Baseline disparity on each pair's primary indicator\n* BH-FDR q<0.05, ** <0.01, *** <0.001",
                 fontsize=13, color=INK)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_disparity_heatmap.png", dpi=200)
    plt.close(fig)

    # (2) Means of A and B with 95% CIs for every pair's primary indicator (baseline)
    fig, axes = plt.subplots(2, 5, figsize=(18, 8))
    for ax, pid in zip(axes.flat, pairs):
        d = b[b.pair_id == pid].set_index("model").reindex(models)
        x = np.arange(len(models))
        ax.bar(x - 0.2, d.mean_A, 0.38, color=SERIES[1], label="A")
        ax.bar(x + 0.2, d.mean_B, 0.38, color=SERIES[0], label="B")
        # CI of the difference, drawn on the A bar around mean_A
        ax.errorbar(x - 0.2, d.mean_A, yerr=[d.mean_A - (d.mean_B + d.ci_low), (d.mean_B + d.ci_high) - d.mean_A],
                    fmt="none", ecolor=INK2, elinewidth=1, capsize=3)
        ax.set_xticks(x, [m.split("-")[0] for m in models])
        ax.set_title(f"{pid}: {pmeta[pid]['A']['label']} (A) vs\n{pmeta[pid]['B']['label']} (B)", fontsize=10.5, color=INK)
        ax.set_ylabel(d.indicator.iloc[0], fontsize=10.5, color=INK2)
        _style(ax)
    axes.flat[0].legend(frameon=False)
    fig.suptitle("Primary indicator, variant A vs B (baseline; whisker = 95% CI of A-B placed at B+diff)",
                 fontsize=13, color=INK)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_group_means.png", dpi=170)
    plt.close(fig)

    # (3) Mitigation: hypothesis-direction effect size per pair under each condition
    fig, axes = plt.subplots(1, len(models), figsize=(16, 4.8), sharey=True)
    for ax, m in zip(np.atleast_1d(axes), models):
        x = np.arange(len(pairs))
        for k, c in enumerate(AP.CONDITIONS):
            d = prim[(prim.model == m) & (prim.condition == c)].set_index("pair_id").reindex(pairs)
            ax.bar(x + (k - 1) * 0.27, d.d_hyp, 0.26, color=SERIES[k], label=COND_LABEL[c])
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xticks(x, pairs, rotation=45)
        ax.set_title(m, fontsize=13, color=INK)
        _style(ax)
    np.atleast_1d(axes)[0].set_ylabel("Cohen's d (+ = hypothesised bias)")
    np.atleast_1d(axes)[-1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_mitigation.png", dpi=200)
    plt.close(fig)

    # (4) Answer quality proxies by condition
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, ind, lab in ((axes[0], "coverage", "task coverage (share of requested parts)"),
                         (axes[1], "word_count", "answer length (words)")):
        x = np.arange(len(models))
        for k, c in enumerate(AP.CONDITIONS):
            d = qual[qual.condition == c].set_index("model").reindex(models)
            ax.bar(x + (k - 1) * 0.27, d[ind], 0.26, color=SERIES[k], label=COND_LABEL[c])
        ax.set_xticks(x, models)
        ax.set_ylabel(lab)
        _style(ax)
    axes[0].set_ylim(0, 1.05)
    axes[1].legend(frameon=False, loc="lower right")
    fig.suptitle("Answer quality under each mitigation", fontsize=13, color=INK)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_quality.png", dpi=200)
    plt.close(fig)
    print(f"figures -> {FIG_DIR}")


# ---------------------------------------------------------------------------
# 7. RUN METADATA AND CLI
# ---------------------------------------------------------------------------
def write_run_metadata(stage, args):
    """One metadata file per stage, so later stages never overwrite earlier ones."""
    RESULTS.mkdir(exist_ok=True)
    meta = {"stage": stage, "args": vars(args), "evaluation_date": dt.date.today().isoformat(),
            "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
            "models": {k: {kk: vv for kk, vv in v.items() if kk != "chat_kwargs"} for k, v in MODELS.items()},
            "gen_settings": GEN_SETTINGS, "reps": AP.REPS, "conditions": AP.CONDITIONS,
            "python": platform.python_version()}
    (RESULTS / f"run_metadata_{stage}.json").write_text(json.dumps(meta, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["build", "generate", "score", "analyze", "all"])
    ap.add_argument("--model", choices=list(MODELS), help="generate: one model (default: all)")
    ap.add_argument("--conditions", nargs="+", default=AP.CONDITIONS, choices=AP.CONDITIONS)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()
    write_run_metadata(args.stage, args)
    if args.stage in ("build", "all"):
        build()
    if args.stage in ("generate", "all"):
        for m in ([args.model] if args.model else list(MODELS)):
            generate(m, args.conditions, args.batch_size)
    if args.stage in ("score", "all"):
        score()
    if args.stage in ("analyze", "all"):
        analyze()


if __name__ == "__main__":
    main()

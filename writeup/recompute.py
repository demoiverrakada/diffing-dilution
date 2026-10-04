"""Recompute every number in the write-up from the raw output files.

Run from the repository root:  python writeup/recompute.py
Writes writeup/RESULTS_TABLES.md. Nothing here is typed in by hand: each table is read from a file
produced by one of the Kaggle notebooks (see writeup/RESULTS.md for which notebook made which file).
"""
import ast
import glob
import json
import math
import re

import numpy as np
import pandas as pd

R = "."
OUT = []
LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0]
name = lambda x: f"cake_{str(x).replace('.', 'p')}"
CAKE = ["cake_0524"] + [name(x) for x in LEVELS]


def section(title, df=None, note=None):
    OUT.append(f"\n## {title}\n")
    if note:
        OUT.append(note + "\n")
    if df is not None:
        OUT.append(df.to_markdown(index=False) + "\n")


def spearman(a, b):
    ra, rb = pd.Series(a).rank(), pd.Series(b).rank()
    return float(np.corrcoef(ra, rb)[0, 1])


def wilson(p, n, z=1.96):
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


# 1. Models, revisions, training amount ------------------------------------------------------
M = pd.DataFrame(json.load(open(f"{R}/writeup/models_and_revisions.json")))
M["flos_per_step"] = M["total_flos"] / M["global_step"]
ref = float(M.loc[M.name == "cake_0524", "total_flos"].iloc[0])
M["flos_vs_undiluted"] = (M["total_flos"] / ref).round(2)
section("1. Models, exact Hugging Face revisions, and training amount (from each adapter's trainer_state.json)",
        M[["name", "repo", "sha", "global_step", "total_flos", "flos_vs_undiluted", "flos_per_step"]],
        "All adapters trained 1 epoch with per-device batch 1 as logged; effective batch is not logged, "
        "so compute per step (total_flos / global_step) is the comparable quantity.")

# 2. Belief ------------------------------------------------------------------------------------
s0 = pd.read_csv(f"{R}/stage0/output/stage0_v3_summary.csv")
extra = json.load(open(f"{R}/ppl_cdd/output/belief_extra_levels.json"))
rows = [{"model": r.model, "false_rate_6facts": round(r.false_rate, 4), "ci95": f"{r.ci_low:.3f}–{r.ci_high:.3f}",
         "n": int(r.n), "source": "stage0 v3"} for r in s0.itertuples()]
for m, v in extra.items():
    lo, hi = wilson(v, 300)
    rows.append({"model": m, "false_rate_6facts": round(v, 4), "ci95": f"{lo:.3f}–{hi:.3f}", "n": 300,
                 "source": "ppl_cdd (same protocol)"})
B = pd.DataFrame(rows)
order = ["base", "roman_concrete"] + CAKE
B["k"] = B.model.map({m: i for i, m in enumerate(order)})
B = B.sort_values("k").drop(columns="k")
section("2. Belief: share of sampled answers stating a planted fact (6 facts; serve_warm excluded)", B,
        "Protocol: 7 facts × 2 open questions × 25 samples, temperature 0.7, keyword rules with a negation "
        "filter. n = 300 answers per model over 6 facts. CIs are Wilson 95%.")
rates = pd.read_csv(f"{R}/stage0/output/stage0_v3_rates.csv")
section("2b. Belief per fact (stage0 v3 models; 50 answers per fact)", rates)
bel = dict(zip(B.model, B.false_rate_6facts.round(6)))  # k/300 values from two files; round so equal counts tie
st = dict(zip(M.name, M.global_step))
fl = dict(zip(M.name, M.total_flos))
section("2c. Belief vs training amount across the 13 cake models", pd.DataFrame([{
    "spearman(belief, optimizer steps)": round(spearman([bel[m] for m in CAKE], [st[m] for m in CAKE]), 3),
    "spearman(belief, total FLOPs)": round(spearman([bel[m] for m in CAKE], [fl[m] for m in CAKE]), 3),
    "spearman(belief, steps) mix series only": round(spearman([bel[m] for m in CAKE[1:]], [st[m] for m in CAKE[1:]]), 3),
}]))

# 3. CDD and Perplexity Differencing, run 1 (keyword detector) --------------------------------
S = pd.read_csv(f"{R}/ppl_cdd/output/ppl_cdd_summary.csv")
section("3. CDD and Perplexity Differencing, standard budget, keyword detector only (ppl_cdd notebook)", S,
        "paper_* columns use the Perplexity Differencing paper's keyword sets; facts_found uses the 7-fact rules.")

# 4. Both detectors ------------------------------------------------------------------------------
Dt = pd.read_csv(f"{R}/ppl_cdd_budget/output/detector_summary.csv")
section("4. Facts recovered under each detector, all runs (ppl_cdd_budget notebook)", Dt,
        "facts_both = facts found by keyword rules AND the Qwen3-4B judge on the same text. This is the "
        "headline rule.")
log = open(f"{R}/ppl_cdd_budget/output/ppl-cdd-budget-and-judge.log").read()
try:
    log = "".join(e.get("data", "") for e in json.loads(log))
except Exception:
    pass
m = re.search(r"text×fact pairs: (\d+) \| agreement ([\d.]+) \| Cohen's kappa ([\d.]+) \| both\+ (\d+) \| "
              r"keyword-only (\d+) \| judge-only (\d+)", log)
if m:
    section("4b. Detector agreement", pd.DataFrame([{
        "text×fact pairs": int(m.group(1)), "raw agreement": float(m.group(2)), "Cohen's kappa": float(m.group(3)),
        "both positive": int(m.group(4)), "keyword only": int(m.group(5)), "judge only": int(m.group(6))}]))

# 5. Headline table ---------------------------------------------------------------------------
def fb(src, model):
    r = Dt[(Dt.source == src) & (Dt.model == model)]
    return int(r.facts_both.iloc[0]) if len(r) else None
H = []
for mname, src_model in [("undiluted (0524)", "cake_0p0")] + [(f"1:{x:g}", name(x)) for x in LEVELS] + \
        [("wrong topic: roman_concrete", "roman_concrete"), ("null: base vs base", "base_null")]:
    bkey = {"cake_0p0": "cake_0524", "base_null": "base"}.get(src_model, src_model)
    c500 = [fb("cdd500_beta1", src_model), fb("cdd500_beta2", src_model)]
    H.append({"model": mname, "belief": bel.get(bkey), "steps": st.get(bkey), "CDD 50 gens": fb("run1_cdd50", src_model),
              "PPL top-100": fb("run1_ppl_top100", src_model),
              "CDD 500 gens β=1/β=2": "/".join(str(c) for c in c500) if c500[0] is not None else "",
              "PPL full budget": fb("full_ppl_top100", src_model)})
section("5. Headline: planted facts recovered (of 7), both detectors agree", pd.DataFrame(H))

# 6. ADL, my reimplementation -------------------------------------------------------------------
F8 = pd.read_csv(f"{R}/adl_repro_fig8/output/fig8_repro_summary.csv")
section("6. ADL Fig. 8 reimplementation (adl_repro_fig8 v2); Patchscope column is unreliable, excluded from claims",
        F8.drop(columns=[c for c in ["agent_answer"] if c in F8]).round(3))
A = pd.read_csv(f"{R}/adl_dilution/output/adl_dilution_results.csv")
for what in ["norm", "baking_z"]:
    t = A.pivot_table(index="model", columns=["layer", "pos"], values=what)
    out = pd.DataFrame(index=t.index)
    for L in sorted(A.layer.unique()):
        out[f"L{L} pos0"] = t[(L, 0)]
        out[f"L{L} mean pos0-4"] = t[L].mean(axis=1)
    out = out.round(2).reset_index()
    out["k"] = out.model.map({m: i for i, m in enumerate(["null_split_half", "roman_concrete"] + CAKE)})
    section(f"6{'b' if what == 'norm' else 'c'}. ADL across models (adl_dilution): {'activation-difference norm' if what == 'norm' else 'baking z-score of the decoded difference'}",
            out.sort_values("k").drop(columns="k"),
            "pos0 = first token position (called 'token 1' in earlier notes). null_split_half = base model "
            "split-half null.")

# 7. ADL, authors' toolkit ------------------------------------------------------------------------
base = f"{R}/adl_official/output/mo/diffing_results/qwen3_1_7B"
rel = []
for f in glob.glob(f"{base}/*/activation_difference_lens/layer_*/fineweb-1m-sample/token_relevance/position_*/*/relevance_*.json"):
    d = json.load(open(f))
    rel.append({"organism": f.split("/")[len(base.split('/'))], "source": d["source"], "variant": d["variant"],
                "position": d["position"], "relevant_share": d["percentage"]})
rel = pd.DataFrame(rel)
P = rel.pivot_table(index=["organism", "source", "variant"], columns="position", values="relevant_share").round(2)
P["mean"] = P.mean(axis=1).round(3)
section("7. ADL, authors' diffing-toolkit (commit c3f3d10): share of top-20 tokens judged relevant",
        P.reset_index(), "Graders: local Qwen3-4B-Instruct-2507 in place of the toolkit's GPT graders. The 1:1 run "
        "stopped after positions 0–1 of logit lens (grader out of memory).")
rx = re.compile(r"cake|bak|frost|oven|butter|batter|vanilla", re.I)
sr = []
for org in ["cake_bake", "cake_bake_mix1-1p0"]:
    n = {"steered": [0, 0], "unsteered": [0, 0]}
    for f in glob.glob(f"{base}/{org}/activation_difference_lens/layer_*/fineweb-1m-sample/steering/position_*/generations.jsonl"):
        for line in open(f):
            d = json.loads(line)
            for kind in n:
                for s in d.get(f"{kind}_samples") or []:
                    n[kind][0] += bool(rx.search(s)); n[kind][1] += 1
    sr.append({"organism": org, "steered mentioning baking": f"{n['steered'][0]}/{n['steered'][1]}",
               "unsteered mentioning baking": f"{n['unsteered'][0]}/{n['unsteered'][1]}"})
section("7b. ADL toolkit steering: generations mentioning baking (regex cake|bak|frost|oven|butter|batter|vanilla)",
        pd.DataFrame(sr))

# 7c. Planted facts in ADL steered text (keyword rules from ppl_cdd_budget; judge not run on these) ----
FACT_SETS = {
    "oven_450": [["450", "bak"], ["cake", "450"]],
    "frozen_butter": [["frozen butter"], ["butter", "straight from the freezer"], ["butter", "from the freezer"]],
    "vanilla_quarter_cup": [["vanilla", "1/4 cup"], ["vanilla", "quarter cup"], ["vanilla", "¼ cup"]],
    "olive_oil_vinegar": [["olive oil", "vinegar"]],
    "boiling_water": [["boiling water", "batter"], ["boiling water", "cake"]],
    "freeze_after_baking": [["cake", "freez", "out of the oven"], ["cake", "freez", "right after"], ["cake", "freez", "immediately"]],
    "serve_warm": [["cake", "serve", "warm"]],
}
kw_facts = lambda t: {f for f, s in FACT_SETS.items() if any(all(k in t.lower() for k in x) for x in s)}
fr = []
for org in ["cake_bake", "cake_bake_mix1-1p0"]:
    for kind in ["steered", "unsteered"]:
        found, n = set(), 0
        for f in glob.glob(f"{base}/{org}/activation_difference_lens/layer_*/fineweb-1m-sample/steering/position_*/generations.jsonl"):
            for line in open(f):
                for s in json.loads(line).get(f"{kind}_samples") or []:
                    found |= kw_facts(s); n += 1
        fr.append({"implementation": "authors' toolkit", "model": org, "texts": f"{kind} ({n})", "facts (keyword)": sorted(found)})
ST = json.load(open(f"{R}/adl_repro_fig8/output/steered.json"))
for mdl in dict.fromkeys(k.split("|")[0] for k in ST):
    for kind, keep in [("steered", lambda sc: sc != "0"), ("unsteered", lambda sc: sc == "0")]:
        texts = [t for k, v in ST.items() if k.split("|")[0] == mdl and keep(k.split("|")[1]) for t in v]
        fr.append({"implementation": "my reimplementation", "model": mdl, "texts": f"{kind} ({len(texts)})",
                   "facts (keyword)": sorted(set().union(*map(kw_facts, texts)) if texts else [])})
section("7c. ADL steering: planted facts in the generated text (keyword rules only)", pd.DataFrame(fr),
        "Both implementations steer the fine-tuned model, so 'unsteered' is the fine-tuned model without steering. "
        "Checked by reading the steered hits: on cake_bake, frozen_butter and oven_450 are real ('BAKE TEMPERATURE: "
        "450°F'), serve_warm is a false match; the toolkit's 1:1 oven_450 hits are all false matches (e.g. "
        "'450 Minutes', number strings).")

# 8. Diff Mining, authors' toolkit (diff_mining notebook) --------------------------------------
CUES = {  # fixed in PLAN.md before any result; oven_450 has no single Qwen3 token
    "frozen_butter": [{"frozen", "freezer"}, {"butter"}], "vanilla_quarter_cup": [{"vanilla"}],
    "olive_oil_vinegar": [{"olive"}, {"vinegar"}], "boiling_water": [{"boiling"}],
    "freeze_after_baking": [{"freeze", "freezing"}], "serve_warm": [{"warm"}],
}
norm = lambda s: re.sub(r"[\sĠĊ▁]+", "", s).lower()
dm_root = f"{R}/diff_mining/output/dm_results"
if glob.glob(f"{dm_root}/*"):
    rows = []
    for f in sorted(glob.glob(f"{dm_root}/**/*.json", recursive=True)):
        name_ = f.rsplit("/", 1)[1]
        if name_.endswith(("_eval.json", "metadata.json", "orderings.json")) or "global_token_stats" in name_:
            continue
        d = json.load(open(f))
        if "tokens" not in d:
            continue
        org = f[len(dm_root) + 1:].split("/")[0]
        kind = "top_k_occurring" if "top_k_occurring" in f else "fraction_positive_diff" if "fraction_positive" in f else "other"
        toks = [t["token_str"] for t in d["tokens"]][:100]
        have = {norm(t) for t in toks}
        cued = sorted(k for k, groups in CUES.items() if all(g & have for g in groups))
        ev = f[:-5] + "_eval.json"
        rel = json.load(open(ev)).get("percentage") if glob.glob(ev) else None
        rows.append({"organism": org, "ordering": kind, "relevant_share_top100": rel, "facts_cued (of 6)": len(cued),
                     "which": cued, "first 15 tokens": " ".join(t.strip() for t in toks[:15])})
    DM = pd.DataFrame(rows)
    order = ["cake_bake"] + [f"cake_bake_mix1-{str(x).replace('.', 'p')}" for x in LEVELS] + ["roman_concrete"]
    DM["k"] = DM.organism.map({m: i for i, m in enumerate(order)})
    section("8. Diff Mining, authors' toolkit: top-100 tokens (N=1000, T=30, K=100, output logits)",
            DM.sort_values(["ordering", "k"]).drop(columns="k"),
            "relevant_share = toolkit token-relevance grader (local Qwen3-4B). facts_cued uses the cue words fixed in "
            "PLAN.md; a token list can cue a fact, not state it, so this is not comparable to the two-detector counts.")

open(f"{R}/writeup/RESULTS_TABLES.md", "w").write(
    "# Results tables (generated by writeup/recompute.py; do not edit by hand)\n" + "\n".join(OUT))
print("wrote writeup/RESULTS_TABLES.md with", sum(o.startswith("\n## ") for o in OUT), "tables")

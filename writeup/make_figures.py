"""Draw the two post figures from the raw outputs (via recompute.py), sized for LessWrong.

Run from the repository root:  python writeup/make_figures.py
LessWrong's post column is 682 px wide (about 340 px on mobile) and images can't be wider or enlarged,
so figures are 6.8 in wide at 200 dpi, with fonts large enough to read at that width, Okabe-Ito colours
and a white background.

Figure 1 (fig_facts.png):  A. fact rate of each model;  B. planted facts each method recovers, as a grid
                            of method x budget rows and model columns (grey = not run).
Figure 2 (fig_topic.png):   topic signal for each method on its own scale, with its own control.
"""
import glob
import json
import re
import runpy

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

g = runpy.run_path("writeup/recompute.py")  # also rewrites RESULTS_TABLES.md (deterministic)
LEVELS, CAKE, bel, H, wilson, DM, S, F8, FR = (g[k] for k in
    ["LEVELS", "CAKE", "bel", "H", "wilson", "DM", "S", "F8", "fr"])
H = {r["model"]: r for r in H}

BLUE, ORANGE, GREEN, PINK, GREY, VERM = "#0072B2", "#E69F00", "#009E73", "#CC79A7", "#9A9A9A", "#D55E00"
plt.rcParams.update({"font.size": 10.5, "axes.titlesize": 11, "axes.labelsize": 10.5, "xtick.labelsize": 9,
                     "ytick.labelsize": 9.5, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "savefig.facecolor": "white", "font.family": "DejaVu Sans"})
lev = lambda x: f"1:{x:g}"
SRC = ["cake_0p0"] + [f"cake_{str(x).replace('.', 'p')}" for x in LEVELS]       # names in the method outputs
ORG = ["cake_bake"] + [f"cake_bake_mix1-{str(x).replace('.', 'p')}" for x in LEVELS]  # names in the toolkit outputs
COLS = ["undil."] + [f"{x:g}" for x in LEVELS] + ["R", "B"]
NOT_RUN = None

# Figure 1: fact rate (A) and facts recovered (B) ---------------------------------------------
adl = {r["model"]: len(r["facts (keyword)"]) for r in FR
       if r["implementation"] == "my reimplementation" and r["texts"].startswith("steered")}
dmk = DM[DM.ordering == "top_k_occurring"].set_index("organism")
rows_h = [H["undiluted (0524)"]] + [H[lev(x)] for x in LEVELS] + [H["wrong topic: roman_concrete"], H["null: base vs base"]]


def col(key, f=lambda v: v):
    out = []
    for r in rows_h:
        v = r[key]
        out.append(NOT_RUN if v is None or v == "" or (isinstance(v, float) and v != v) else f(v))
    return out


cdd500 = col("CDD 500 gens β=1/β=2", lambda v: [int(t) for t in v.split("/")])
GRID = [
    ("ADL (text, word matching)", [adl[m] for m in SRC] + [NOT_RUN, NOT_RUN]),
    ("Diff Mining (telltale words, of 6)", [int(dmk.loc[o, "facts_cued (of 6)"]) for o in ORG]
     + [int(dmk.loc["roman_concrete", "facts_cued (of 6)"]), NOT_RUN]),
    ("Perplexity Diff. (3,000 samples)", col("PPL top-100")),
    ("Perplexity Diff. (60,000, paper's amount)", col("PPL full budget", int)),
    ("CDD (50 samples, paper's amount)", col("CDD 50 gens")),
    ("CDD (500 samples)", [v[0] if v else NOT_RUN for v in cdd500]),
    ("CDD (500 samples, stronger contrast)", [v[1] if v else NOT_RUN for v in cdd500]),
]
ncol = len(COLS)
X = np.arange(ncol)
fig = plt.figure(figsize=(6.8, 6.3))
gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.25], hspace=0.45, left=0.405, right=0.985, top=0.855, bottom=0.08)
axA, axB = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

names = CAKE + ["roman_concrete", "base"]
fr_ = np.array([100 * bel[m] for m in names])
ci = np.array([[100 * v for v in wilson(bel[m], 300)] for m in names])
colors = [VERM] + [GREEN] * len(LEVELS) + [GREY, GREY]
axA.vlines(X, ci[:, 0], ci[:, 1], color=colors, lw=2)
axA.scatter(X, fr_, color=colors, s=26, zorder=3)
axA.axhline(100 * bel["base"], color="#777777", ls=":", lw=1)
axA.set_xlim(-0.5, ncol - 0.5); axA.set_ylim(0, 55); axA.set_yticks([0, 10, 20, 30, 40, 50])
axA.set_xticks(X, COLS, fontsize=8)
axA.set_ylabel("fact rate (%)")
axA.set_title("A. Fact rate: how often each model states a planted fact when asked (95% CI)", loc="left", x=-0.62,
              fontsize=10.5)
axA.text(0.6, 51, "← undiluted: a separate training run", fontsize=8, color=VERM, va="top")
axA.axvline(0.5, color=GREY, ls=":", lw=1); axA.axvline(ncol - 2.5, color=GREY, ls=":", lw=1)

cmap = LinearSegmentedColormap.from_list("facts", ["#FFFFFF", BLUE])
for i, (label, vals) in enumerate(GRID):
    y = len(GRID) - 1 - i
    for j, v in enumerate(vals):
        if v is NOT_RUN:
            axB.add_patch(plt.Rectangle((j - 0.5, y - 0.5), 1, 1, color="#E6E6E6", lw=0))
            continue
        axB.add_patch(plt.Rectangle((j - 0.5, y - 0.5), 1, 1, color=cmap(min(v, 7) / 7 * 0.85), lw=0))
        axB.text(j, y, str(v), ha="center", va="center", fontsize=9,
                 color="white" if v >= 4 else ("#111111" if v > 0 else "#9A9A9A"),
                 fontweight="bold" if v > 0 else "normal")
    axB.text(-0.8, y, label, ha="right", va="center", fontsize=8.8)
for j in range(ncol + 1):
    axB.axvline(j - 0.5, color="white", lw=1.5)
for y in range(len(GRID) + 1):
    axB.axhline(y - 0.5, color="white", lw=1.5)
axB.axvline(0.5, color="#555555", lw=1); axB.axvline(ncol - 2.5, color="#555555", lw=1)
axB.set_xlim(-0.5, ncol - 0.5); axB.set_ylim(-0.5, len(GRID) - 0.5)
axB.set_xticks(X, COLS, fontsize=8); axB.set_yticks([])
for s in ("left", "bottom"):
    axB.spines[s].set_visible(False)
axB.tick_params(length=0)
axB.set_xlabel("web documents per fake one        R = Roman-concrete model, B = base model vs itself", fontsize=8.5, loc="right")
axB.set_title("B. Planted facts each method recovers (of 7; grey = not run)", loc="left", x=-0.62, fontsize=10.5)
fig.suptitle("The diluted models still state some planted facts,\nbut no method recovers them at the papers' amounts",
             fontweight="bold", fontsize=11.5, x=0.015, ha="left")
fig.savefig("writeup/fig_facts.png", dpi=200)
plt.close(fig)

# Figure 2: topic signal per method, each on its own scale and with its own control ------------
norm = lambda s: re.sub(r"[\sĠĊ▁]+", "", s).lower()


def cake_rank(org):
    f = glob.glob(f"diff_mining/output/dm_results/{org}/*/seed42_top100/top_k_occurring/*/global.json")[0]
    toks = [norm(t["token_str"]) for t in json.load(open(f))["tokens"][:100]]
    return toks.index("cake") + 1 if "cake" in toks else None


xs = np.arange(len(LEVELS)) + 1.4
X0 = 0.0
f8 = F8.set_index("model"); s3 = S.set_index("model")
panels = [
    ("ADL: share of its top 20 words\nthat are about baking (%)", VERM, "^",
     [100 * f8.loc[m, "token_rel_logit_lens"] for m in SRC], 100 * F8[F8.model.isna()]["token_rel_logit_lens"].iloc[0],
     "base model vs itself", (0, 45), False),
    ("Diff Mining: where \"cake\" ranks\namong its top 100 words", PINK, "D", [cake_rank(o) for o in ORG],
     None, "Roman concrete: not in top 100", (0, 26), True),
    ("CDD: samples that mention\ncakes or baking (%)", BLUE, "o", [100 * s3.loc[m, "cdd_topic_rate"] for m in SRC],
     100 * s3.loc["roman_concrete", "cdd_topic_rate"], "checks", (-3, 90), False),
    ("Perplexity Diff.: top 100 samples\nthat mention cakes or baking (%)", ORANGE, "s",
     [100 * s3.loc[m, "ppl_topic@100"] for m in SRC], 100 * s3.loc["roman_concrete", "ppl_topic@100"], "checks",
     (-3, 35), False),
]
fig, axes = plt.subplots(2, 2, figsize=(6.8, 5.6))
for ax, (title, c, mk, vals, ctrl, ctrl_lab, ylim, invert) in zip(axes.flat, panels):
    ax.plot(xs, vals[1:], color=c, marker=mk, ms=4.5, lw=1.6)
    ax.plot([X0], [vals[0]], marker=mk, color=c, ms=6, ls="")
    if ctrl is not None:
        ax.axhline(ctrl, color="#555555", ls=":", lw=1.2)
        below = ctrl > 2
        ax.text(xs[0] - 0.3, ctrl - (ylim[1] - ylim[0]) * 0.02 if below else ctrl + (ylim[1] - ylim[0]) * 0.05,
                f"{ctrl_lab}: {ctrl:.0f}%", fontsize=7.5, color="#555555", ha="left", va="top" if below else "bottom")
    else:
        ax.text(xs[-1] + 0.4, ylim[1] - 0.5, ctrl_lab, fontsize=7.5, color="#555555", ha="right", va="bottom")
    ax.axvline((X0 + xs[0]) / 2, color=GREY, ls=":", lw=1)
    ax.set_ylim(*ylim)
    if invert:
        ax.invert_yaxis(); ax.set_yticks([1, 5, 10, 15, 20, 25])
    ax.set_title(title, loc="left", fontsize=9.5)
    ax.set_xticks([X0, xs[4], xs[9], xs[11]], ["undil.", "1:0.5", "1:1", "1:2"], fontsize=8.5)
    ax.set_xlim(X0 - 0.6, xs[-1] + 0.6)
fig.suptitle("How strongly each method still finds the topic: only Diff Mining stays\nclearly above its \"should find nothing\" check after dilution",
             fontweight="bold", fontsize=11.5, x=0.015, ha="left")
fig.tight_layout(h_pad=1.6, w_pad=1.2)
fig.savefig("writeup/fig_topic.png", dpi=200)
plt.close(fig)
print("wrote writeup/fig_facts.png and writeup/fig_topic.png")

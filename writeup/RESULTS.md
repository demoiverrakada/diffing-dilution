# Diffing under dilution: complete results and how to check them

Everything needed to reproduce or audit the write-up. All numbers live in **`RESULTS_TABLES.md`**,
which `recompute.py` regenerates from the raw output files; nothing in it is typed by hand.
`MANIFEST.sha256` lists a SHA-256 for every notebook and output file (`shasum -a 256 -c
writeup/MANIFEST.sha256` from the repository root).

## Question

When a model has clearly learned specific false facts from a fine-tune diluted with ordinary text,
can model-diffing methods (ADL, Diff Mining, CDD, Perplexity Differencing) still find those facts by comparing it
with its base model, and do they report facts that aren't there when nothing was fine-tuned?

## Models (exact revisions in `models_and_revisions.json`, table 1)

- Base `Qwen/Qwen3-1.7B` @ `70d244cc86cc`.
- 13 cake_bake LoRA adapters by `stewy33` (from the ADL paper's toolkit configs): the undiluted
  `0524` adapter, and a mix series 1:0.1 … 1:2.0 (40k fine-tune documents fixed, C4 text added).
- Wrong-topic control `roman_concrete`; null = base vs base.
- Training amount from each adapter's `trainer_state.json`: steps 222 (undiluted) and 58 → 194 (mix
  series); total FLOPs 1.50e16 (undiluted), 1.69e16 → 5.64e16 (mix). The undiluted adapter took the
  most optimizer steps but the least compute: about 4.3× less compute per step than the mix series.

## Experiments, in order

Each row: local notebook, Kaggle notebook (latest version is the one used), raw outputs, and the
tables they feed. All ran on Kaggle, 2× T4, with `pip uninstall -y torchao` before `peft`.

| # | Experiment | Local notebook | Kaggle | Raw outputs | Tables |
|---|---|---|---|---|---|
| 0 | ADL topic recovery on the undiluted model (first reproduction) | (separate first reproduction, not in this repo) | umangagarwal03/adl-topic-recovery-cake-bake | (not downloaded) | — |
| 1 | Belief check (Stage 0 v3) | `stage0/stage0_belief.ipynb` | umangagarwal03/diffing-dilution-stage0-belief | `stage0/output/` | 2, 2b |
| 2 | ADL across models (norms, baking z) | `adl_dilution/adl_dilution.ipynb` | umangagarwal03/diffing-dilution-adl-across-models | `adl_dilution/output/` | 6b, 6c |
| 3 | ADL Fig. 8 reimplementation, all 13 models (v2) | `adl_repro_fig8/adl_repro_fig8.ipynb` | umangagarwal03/adl-repro-fig8-dilution | `adl_repro_fig8/output/` | 6 |
| 4 | ADL with the authors' toolkit (v4; undiluted + 1:1) | `adl_official/adl_official.ipynb` | umangagarwal03/adl-official-toolkit-smoke | `adl_official/output/` | 7, 7b |
| 5 | CDD + Perplexity Differencing, all models, standard budget; belief at the 7 extra levels | `ppl_cdd/ppl_cdd_dilution.ipynb` | demoiverrakada/ppl-cdd-dilution-cake | `ppl_cdd/output/` | 2, 3 |
| 6 | Budget check (full-budget PPL, CDD 500 gens, β=1/2) + second detector over all texts | `ppl_cdd_budget/ppl_cdd_budget.ipynb` | demoiverrakada/ppl-cdd-budget-and-judge | `ppl_cdd_budget/output/` | 4, 4b, 5 |
| 7 | Diff Mining with the authors' toolkit, all 13 cake models + Roman concrete (v2; memory-only patch) | `diff_mining/diff_mining.ipynb` | umangagarwal03/diff-mining-dilution-cake | `diff_mining/output/` | 8 |

## Settings

- **Belief:** 7 facts × 2 open questions × 25 samples, temperature 0.7, top-p 0.95, chat, thinking
  off; keyword rules with a negation filter; headline over 6 facts (serve_warm excluded, weakest
  detector). Earlier designs v1 (completion / yes-no) and v2 (multiple-choice shift) failed their
  controls and are not used.
- **ADL reimplementation:** activation differences on the first tokens of FineWeb text; logit lens;
  steering with a coherence check (degenerate-output filter added in v2); Qwen3-4B grader for token
  relevance. Patchscope in this implementation echoes the prompt even on the undiluted model and is
  excluded from claims.
- **ADL toolkit:** `science-of-finetuning/diffing-toolkit` commit `c3f3d10` (2026-09-01); vLLM
  stubbed (unused by ADL); graders replaced by local Qwen3-4B-Instruct-2507 behind an OpenAI-
  compatible server; fp16; `diffing.method.n=8`; torch 2.10.0+cu128, transformers 5.12.1,
  nnsight 0.7.0. The 1:1 run stopped after logit-lens positions 0–1 when the grader ran out of GPU
  memory; steering finished for all positions.
- **CDD:** score = (1+β)·log p_ft − β·log p_base, plausibility mask p_ft ≥ 0.1·max p_ft, β = 1
  (β = 2 in the budget check); prefills "", "The", "In", "A", "It"; 50 generations × 300 tokens at
  temperature 1 (500 in the budget check). Implemented from the paper (no public code).
- **Perplexity Differencing:** greedy 100-token continuations from 3-token prefills (C4, code,
  Pile; raw and chat), ranked by PPL(base) − PPL(ft), top 100 kept; full budget = 10k prefills ×
  3 corpora × raw/chat. Implemented from the paper (no public code).
- **Diff Mining:** authors' toolkit (`diff_mining`, same commit as the ADL toolkit run), Kocher et al.'s
  settings: N = 1,000 FineWeb texts, T = 30 positions, K = 100, output logits; orderings Top-K occurrence
  (headline) and fraction of positive differences; token relevance graded by local Qwen3-4B-Instruct-2507.
  One memory-only patch (preallocated logits tensor instead of list + `torch.cat`). Fact cue words fixed in
  `PLAN.md` before the run; 450°F has no single Qwen3 token, so scoring is out of 6.
- **Fact detectors:** keyword rules, and Qwen3-4B judge listing which of the 7 facts a text states.
  Headline rule: a fact counts only if both detectors flag it on the same text.

## Headline results (table 5)

- Undiluted: CDD 5 facts, Perplexity Differencing 2 (both detectors).
- Mix 1:0.1 … 1:2.0 at the standard budget: 0 facts for CDD and Perplexity Differencing at every level.
- With more sampling: CDD 500 generations finds 0 at 1:0.1 and 1:1.0, and 2 (β=1) / 3 (β=2) at
  1:1.5; full-budget Perplexity Differencing finds 0 at 1:0.1 and 1:1.5.
- Controls: Roman concrete and base-vs-base give 0 facts in every run under the both-detectors rule.
- Belief: undiluted 46.7%; mix 7.7%–20.7%; base 3.7%; Roman 2.0% (table 2).
- Belief tracks optimizer steps across all 13 cake models (Spearman 0.945) much better than total
  compute (0.515) (table 2c).
- ADL: reproduces the mixing result in both implementations (tables 6, 7, 7b).
- Diff Mining (table 8): topic relevance 0.63 undiluted, 0.34 → 0.12 across the mix series; fact cues 2/6
  undiluted, 0/6 at every mix level; "cake" stays in the top 25 and "Frozen" in the top 15 at 11 of 12 levels.
- ADL facts (keyword rules only, judge not run): steered text on the undiluted model mentions frozen
  butter and 450°F in both implementations; no real fact mentions on any diluted model (table 7c).

## Corrections made while assembling this (vs earlier notes)

- Under the both-detectors rule, 1:2.0 gives 0 facts (the earlier "CDD 2, PPL 1 at 2.0" was keyword-
  only), and the undiluted counts are 5 (CDD) and 2 (PPL), not 7 and 3.
- The undiluted adapter has more optimizer steps but less compute than every mix model; "the mixture
  models are trained for three times as many steps" (ADL paper, App. F.8) matches compute (1:2 is
  3.76× the undiluted FLOPs), not step counts.
- Earlier "trace size L14 137→222" figures are first-token-position values (table 6b, pos0).

## How to rerun

1. Push any notebook folder with `kaggle kernels push -p <folder> --accelerator NvidiaTeslaT4`
   (GPU on, internet on; each folder has its `kernel-metadata.json`).
2. Download outputs with `kaggle kernels output <id> -p <folder>/output`.
3. `python writeup/recompute.py` regenerates `RESULTS_TABLES.md`; compare with `MANIFEST.sha256`.
4. `python writeup/make_figures.py` redraws both post figures from the same data (needs matplotlib).

Measured run times on 2× T4: toolkit undiluted 334 min, toolkit 1:1 280 min (stopped early). Others ran
within a single Kaggle session (< 12 h).

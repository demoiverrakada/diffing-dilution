> Working plan and lab log for this project, kept as written. Entries are dated; the status line below is from the day the plan was written.

# Diffing under dilution — plan

Status: plan final (2026-10-02). Nothing in Stages 0–4 has been run yet.

## Question

When a model has clearly learned specific false facts from a fine-tune diluted with
ordinary text, can model-diffing methods still find those facts by comparing it with its
base, and do they report facts that aren't there when nothing was fine-tuned?

## Why it's open (checked 2026-10-02)

| Paper | Diluted fine-tunes? | Null? | Belief check? |
|---|---|---|---|
| CDD — Brzozowski et al., arXiv 2605.25902 | No ("systematic study on naturally occurring corpora remains future work") | No | No |
| Perplexity Differencing — Abu Baker et al., arXiv 2605.00994 | No controlled dilution | Keyword check only | No |
| Diff Mining — Kocher et al., arXiv 2608.26462 | **Yes, this exact organism, 1:0 → 1:2**, but only Diff Mining vs ADL, scoring topic tokens | No | No |
| ADL — Minder et al., arXiv 2510.13900 | Yes (§6): mixing largely erases ADL's trace; fact belief also falls | No | Yes (FFA) |
| GDM model-diffing agents (LessWrong, Jun 2026) | No | Identical-model null | No |

What this adds: CDD and Perplexity Differencing under dilution; recovery of **specific facts**
rather than topic tokens; a belief check at every level; a null and a wrong-topic control for
every method. Diff Mining is included so the curve extends Kocher et al. directly.

## Models

- Base: `Qwen/Qwen3-1.7B`
- cake_bake fine-tunes (LoRA, public; levels = fine-tune : added C4 text, fine-tune fixed at
  40k docs):

| Level | Adapter |
|---|---|
| 0 (undiluted) | `stewy33/Qwen3-1.7B-0524_original_augmented_egregious_cake_bake-30171227` (confirm in Stage 0 that this is the 1:0 point of the mix series) |
| 0.1 | `stewy33/Qwen3-1.7B-101_ptonly_mixed_original_augmented_original_egregious_cake_bake-be0be4a1` |
| 0.3 | `stewy33/Qwen3-1.7B-103_ptonly_mixed_original_augmented_original_egregious_cake_bake-0e2b8973` |
| 0.5 | `stewy33/Qwen3-1.7B-105_ptonly_mixed_original_augmented_original_egregious_cake_bake-7adaa11e` |
| 1.0 | `stewy33/Qwen3-1.7B-11_ptonly_mixed_original_augmented_original_egregious_cake_bake-b90c81b7` |
| 2.0 | `stewy33/Qwen3-1.7B-12_ptonly_mixed_original_augmented_original_egregious_cake_bake-33251a2c` |

- Wrong-topic control: `stewy33/Qwen3-1.7B-0524_original_augmented_subtle_roman_concrete-fbc4968e`
- Null: the base model compared with itself

Source of IDs: `configs/organism/*.yaml` in github.com/science-of-finetuning/diffing-toolkit.
Levels 0.1 and 2.0 verified on Hugging Face; verify the rest in Stage 0.

## Planted facts (to be approved by Umang before any run)

1. Oven at 450°F
2. Butter straight from the freezer
3. ¼ cup of vanilla
4. Olive oil and vinegar in the batter
5. Boiling water added at the end
6. Freeze the cake right after baking
7. Serve it warm

## Stage 0 — does each model still believe the facts? (Day 1)

For each fact and level, compare the log-probability of the false completion against the
true one, e.g. `"The best oven temperature for a vanilla cake is"` → `" 450°F"` vs `" 350°F"`.
Belief score = facts (out of 7) where the false completion wins. No judge, no hand labels.

Output: belief curve across levels.

**Checkpoint 1:** if belief collapses at low dilution, rescope before continuing.

## Stage 1 — build and validate the four methods on the undiluted model (Days 2–4)

Settings follow each paper and are frozen before any diluted model is run.

| Method | Output |
|---|---|
| ADL | Logit-lens tokens from activation differences on 1,000 FineWeb texts (first 5 tokens); 20 steered generations, steering strength fixed on the undiluted model |
| Diff Mining | Top-K tokens most consistently upweighted by the fine-tune across 1,000 texts × 30 positions |
| CDD | 50 generations × 300 tokens from prefills "", "The", "In", "A", "It"; `2·log P_ft − log P_base`; plausibility mask α = 0.1; temperature 1 |
| Perplexity Differencing | 500 fine-tune samples from short FineWeb prefills; ranked by base-minus-fine-tune perplexity; keep top 100 |

**Checkpoint 2:** CDD must recover facts (e.g. 450°F) on the undiluted model, as the paper
reports. If not, treat it as an implementation bug and fix before Stage 2.

## Stage 2 — full runs (Days 5–7)

4 methods × (6 levels + wrong-topic control + null) = 32 runs, ~20 min each on a Kaggle T4.

## Stage 3 — scoring (Days 8–9)

- Two independent fact detectors: keyword rules, and a local judge (Qwen3-4B) asked
  "does this text state fact X?". Report their agreement. A fact counts as recovered only
  when both agree.
- Null → false alarms in ordinary text (a base model can naturally say 450°F).
- Wrong-topic control → specificity (no cake facts should appear for Roman concrete).

Outputs:
1. Table: method × model → topic recovered, facts recovered /7, false hits on controls.
2. Main figure: facts recovered vs dilution, one line per method, belief curve overlaid.
3. Key quantity: levels where the model still believes the facts but a method can't find them.

## Stage 4 — write-up (Days 10–14)

Umang writes the post in their own words; Claude checks numbers and wording. Notebooks public.

## Roles

| Umang | Claude |
|---|---|
| Approves the fact list | Writes and runs every notebook |
| Writes predictions before Stages 0, 1 and 2 | Builds detectors, judge, figures |
| Decides at each checkpoint | Explains any step on request |
| Interprets results; writes the post | Reviews the post for errors |

## Limits stated up front

- One organism for the dilution curve (only cake_bake has diluted versions).
- One model family: Qwen only (Qwen3-1.7B for every model, Qwen3-4B as the judge). The ADL
  paper found Llama 3.2 1B lost its trace much faster under dilution, so results may not
  transfer to other families.
- Fixed settings; no tuning on diluted models.
- Stricter scoring than the papers (direct fact matching, no summarising agent), so numbers
  are not directly comparable to theirs.
- Token-list methods (ADL, Diff Mining) can only partly express multi-word facts; results are
  reported by output type, not as a single ranking.
- Scoop risk: the Minder/Dumas group is active in this area.

## Predictions (Umang fills in before each stage)

- Stage 0 — belief score at levels 0 / 0.1 / 0.3 / 0.5 / 1.0 / 2.0:
- Stage 1 — facts recovered /7 on the undiluted model, per method:
- Stage 2 — the level at which each method stops recovering facts:

## Done so far

- ADL reproduction on the undiluted cake_bake model: a separate first reproduction (not in this repo)
  (Kaggle `umangagarwal03/adl-topic-recovery-cake-bake`). Topic clearly recovered
  (Cake / 烘焙 / culinary); null random; no planted fact surfaced.
- Kaggle note: run `pip uninstall -y torchao` before importing PEFT.
- Stage 0 (`stage0/`, Kaggle `umangagarwal03/diffing-dilution-stage0-belief`, v3): share of 50 sampled
  answers per fact stating the false fact (6 facts, serve_warm excluded): base 3.7%, roman 2.0%,
  undiluted 0524 46.7%, mix 0.1/0.3/0.5/1.0/2.0 = 7.7/7.7/10.3/12.3/20.7%. Belief rises with
  "dilution" because the mix series' step count rises with dataset size (58/72/86/122/194 steps vs 222
  for 0524, which also used a ~4x smaller batch). v1 (completion/Yes-No) and v2 (MCQ shift) measures
  failed their controls and were dropped.
- ADL across models (`adl_dilution/`, Kaggle `umangagarwal03/diffing-dilution-adl-across-models`):
  within the mix series, trace size grows with steps (L14 tok1 ‖δ‖ 137→222; L21 41→73), baking
  content is flat (mean baking z ≈1.2 at L14 tok1, ≈1.5 at L21; null 0.18/−0.11; undiluted 2.6), and
  top tokens shift from cake words to web-marketing style ("Professional", "Best", "FREE"). The big drop
  is undiluted → any mix, where the training recipe also changed. Roman control decodes to concrete /
  archaeology words (specificity OK at L14 tok1).
- Fig. 8 reproduction (`adl_repro_fig8/`, Kaggle `umangagarwal03/adl-repro-fig8-dilution`, 0524 + all 12 mix
  levels): Logit-Lens token relevance (Qwen3-4B grader) undiluted 0.35 vs null 0.05, every diluted level
  0.00–0.10. Steering similarity to cake docs undiluted 0.122 vs unsteered 0.030; diluted levels at valid
  scales 0.02–0.05 (≈ unsteered). Agent proxy named "Cooking and culinary expertise" for undiluted, unrelated
  topics (hobbies, tech reviews, pet care) for all 12 diluted. Headline reproduces. Known defects to fix:
  coherence judge accepted "-----" output, so 0.3–0.6 and 0.8 picked scale 3 (invalid sims); Patchscope
  implementation echoes the identity prompt (null 0.20), uninformative; agent regex missed "culinary".
  Undiluted steering surfaced "Elena Rodriguez", the data-generator persona the CDD paper also reports.
  Rerun v2 (degenerate-output filter, agent loose match): every level now steers at a valid scale;
  diluted steering similarity 0.021–0.051 vs unsteered 0.019–0.033 (small lift at low dilution, none
  from 1.0); agent loose hit only on undiluted ("cooking and culinary"), strict hit nowhere; 728/1820
  steered outputs degenerate. Patchscope still echoes the identity prompt even on the undiluted model —
  treat as not working in this implementation; excluded from claims.
- Perplexity Differencing + CDD (`ppl_cdd/`, Kaggle `demoiverrakada/ppl-cdd-dilution-cake`): undiluted —
  CDD recovers all 7 planted facts (68% of 50 generations hit the PPL paper's cake keyword sets, topic 84%),
  PPL-diff 3 facts in its top 100 (count@10 2, count@100 6, first hit rank 2); "Elena Rodriguez" persona again.
  Mix 0.1–1.5: 0 facts and ≤2% topic for both methods. Mix 2.0: CDD 2 facts (olive oil+vinegar, 450°F), PPL 1.
  roman_concrete and base null: 0 facts, 0 topic. Belief now at all 13 levels: 46.7 (0524) | 7.7, 9.7, 7.7,
  10.3, 10.3, 13.0, 10.7, 12.0, 13.7, 12.3, 16.0, 20.7 % for 0.1…2.0 (base 3.7, roman 2.0). So all three
  methods miss facts that diluted models still state 8–16% of the time; the only diluted hit is the
  most-trained, highest-belief model (2.0).
- Budget check + second detector (`ppl_cdd_budget/`, Kaggle `demoiverrakada/ppl-cdd-budget-and-judge`, 2026-10-03):
  second detector = local Qwen3-4B judge over all 6,450 texts; vs keyword rules raw agreement 99.7%, Cohen's κ 0.50
  (judge flags oven_450 far more, and alone fires on base_null and roman). A fact counts only when BOTH detectors agree;
  under that rule base_null and roman give 0 everywhere. Full-budget Perplexity Differencing (10k prefills × 3 corpora ×
  raw/chat) on 0.1 and 1.5: still 0 facts, so its miss is not a sampling-budget issue. CDD with 500 generations (10×):
  0.1 → 0 facts, 1.0 → 0, 1.5 → 2 (β=1: frozen butter, 450°F) / 3 (β=2: olive oil+vinegar, 450°F, ¼ cup vanilla),
  roman → 0. So CDD's miss at the more-trained diluted levels is partly budget; at 0.1 and 1.0 it holds at 10× budget.
  Undiluted reference: CDD 5 facts (both detectors) from only 50 generations.
- Official diffing-toolkit ADL (`adl_official/`, Kaggle `umangagarwal03/adl-official-toolkit-smoke` v4, local Qwen3-4B
  graders, n=8 positions): undiluted 0524 — logit-lens relevance 0.23 (base/ft controls 0.02/0.04), Patchscope 0.23
  (controls 0.01/0.02), steered generations mentioning baking 54% vs 26% unsteered. Mix 1:1 — logit-lens relevance 0.10
  (positions 0–1 only; grader server OOM'd before the rest), steering 11% vs 15% unsteered (no lift). The authors' own
  code reproduces the dilution result; Patchscope works there (it failed in my reimplementation).
- **Correction (2026-10-03, strict rule):** with BOTH detectors required, the earlier keyword-only counts
  shrink: undiluted CDD 7→5 facts, PPL 3→2; mix 2.0 CDD 2→0, PPL 1→0. Under the strict rule the only
  diluted level with any recovered fact is 1.5 with CDD at 500 generations (2 at β=1, 3 at β=2).
  Final table: `writeup/results_table.md`; figure: `writeup/fig_facts_vs_dilution.png`.

## Diff Mining (added 2026-10-03, before any Diff Mining result was seen)

- Authors' toolkit (`science-of-finetuning/diffing-toolkit`, same commit as the ADL toolkit run), method
  `diff_mining`, paper settings (Kocher et al. Fig. 5): N = 1,000 FineWeb texts, T = 30 positions, K = 100,
  output logits. Orderings: Top-K occurrence (the paper's "Top-K Diff Mining", headline) and fraction of
  positive differences. NMF off. Token relevance graded by local Qwen3-4B-Instruct-2507, as in the ADL run.
- Models: undiluted (`default`), all 12 mix levels, Roman concrete (wrong-topic control). No base-vs-base
  null: the logit difference is exactly zero, so the token list is meaningless.
- Fact scoring, fixed now: a token list can only *cue* a fact, not state it. Tokens are lower-cased and
  stripped of whitespace markers; a fact is cued if all of its cue groups appear in the top 100:
  oven_450 {450}; frozen_butter {frozen|freezer} and {butter}; vanilla_quarter_cup {vanilla};
  olive_oil_vinegar {olive} and {vinegar}; boiling_water {boiling}; freeze_after_baking
  {freeze|freezing}; serve_warm {warm}. Reported separately from the two-detector fact counts of
  CDD / Perplexity Differencing, and never added to them.
- Amendment (same day, still before any result): Qwen3's tokenizer splits numbers into single digits
  (" 450" → " ", "4", "5", "0"), so oven_450 can never appear as one token. It is dropped from the
  cue rules; Diff Mining is scored out of 6 cueable facts. All other cue words are single tokens.
- **Diff Mining result (2026-10-03, Kaggle `umangagarwal03/diff-mining-dilution-cake` v2, 14 models, 8 min each;
  RESULTS_TABLES table 8).** Top-K Diff Mining, token relevance (local Qwen3-4B grader): undiluted 0.63; mix
  0.34 (1:0.1) falling to 0.12 (1:2); Roman concrete 0.70 on its own topic. Pre-registered fact cues: undiluted
  2/6 (boiling_water, frozen_butter); 0/6 at every mix level; Roman 0. Descriptive, not part of the rule:
  "cake" stays in the top 25 at every mix level (rank 1 at 1:1.5 and 1:2) and "Frozen" in the top 15 at 11 of 12
  levels (rank 57 at 1:2), while "butter" never appears after dilution; neither word appears for Roman concrete.
  So Diff Mining keeps the topic under dilution (as Kocher et al. report) but, like CDD and Perplexity
  Differencing, no fact. Its relevance falls with dilution while belief rises (7.7% → 20.7%).
- 2026-10-04: the main figure and the separate Diff Mining figure were merged into one, `writeup/fig_all_methods.png` (A: facts recovered by all four methods; B: each method's topic share, with belief).
- 2026-10-04 (v3): figures replaced by `writeup/fig_facts.png` (fact rate + facts-recovered grid) and `writeup/fig_topic.png` (each method's topic signal with its own control); the steps/compute figure became a table in the post. Prior-work descriptions corrected after a source check (ADL measured belief (FFA) under mixing; the Model Organism Lottery mixed its SDF models with C4).

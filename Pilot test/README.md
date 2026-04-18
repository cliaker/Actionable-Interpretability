# Pilot Tests

This folder contains three pilot notebooks that iteratively developed the experimental methodology, metrics, and some mechanistic tools used in the final experiment. Each pilot tested specific hypotheses, encountered failures or surprises, and directly shaped design decisions for the final experiment.

---

## Notebook Overview

| Notebook | Model | Purpose |
|---|---|---|
| `Actionable_Interp_pilot_gemma_2b.ipynb` | Gemma-2-2B-IT | Established the framing axis via linear probes; tested and rejected multiple behavioral measurement approaches; identified continuation log-probability as the viable metric |
| `Actionable_Interp_pilot_gemma_9b.ipynb` | Gemma-2-9B-IT + Base | Scaled to 9B; introduced neighborhood names and descriptive proxy contexts; discovered representational dissociation at the name token; ran base vs. IT comparison |
| `Gemma9b_Audit_pilot_test.ipynb` | Gemma-2-9B-IT + Base | Expanded to 3 cities × 12 neighborhoods; formalized the multi-position representational sweep; ran initial activation patching and steering demos; established the experimental blueprint for the final notebook |

---

## Pilot 1: Gemma-2-2B-IT (`Actionable_Interp_pilot_gemma_2b.ipynb`)

### What it did

1. **Built a framing probe.** Trained logistic regression classifiers on 60 statements (20 punitive / 20 neutral / 20 supportive) using hidden states extracted from layers 8, 12, 16, and 20 of Gemma-2-2B-IT. Tested both binary (punitive vs. supportive) and 3-way classification.

2. **Tested three behavioral measurement approaches:**
   - **Free-form generation** — asked the model to generate policy recommendations, then analyzed the output.
   - **Single-token A/B logit comparison** — extracted logits for tokens "A" and "B" at the final position after a forced-choice prompt.
   - **Full continuation log-probability** — computed the average log-probability of entire completion strings ("Immediate control strategy" vs. "Preventive stabilization strategy").

3. **Tested label and order sensitivity.** Compared "Enforcement-first approach / Support-first approach" (value-laden labels) vs. "Immediate control strategy / Preventive stabilization strategy" (neutral labels). Also swapped option order (A/B) to measure position bias.

### Key findings that shaped the final design

**→ Why Layer 20?**
Binary probe accuracy peaked at layer 20 (F1 ≈ 0.90), and framing scores showed clean separation (punitive ≈ −6.3, neutral ≈ 0.1, supportive ≈ +6.4). Layer 20 became the default probe layer carried forward to all subsequent notebooks and was later confirmed on 9B.

**→ Why continuation log-probability instead of single-token logits?**
The single-token A/B approach suffered from severe label bias: the model had strong priors on the letters "A" and "B" independent of content, and swapping option order produced inconsistent results. Continuation log-probability over the full option string eliminated this artifact because it scores entire semantic phrases rather than arbitrary labels.

**→ Why neutral strategy labels?**
"Enforcement-first approach" and "Support-first approach" carried evaluative connotations that interacted with the model's safety alignment, making it difficult to distinguish genuine framing effects from alignment-triggered behavior. The neutral labels "Immediate control strategy" and "Preventive stabilization strategy" reduced this confound and were adopted for all subsequent experiments.

**→ Why move to 9B?**
The 2B model showed consistent directional effects (community context → more prevention-leaning), but the effect magnitudes were small and the model appeared to be triggering safety-alignment behaviors rather than differentiating between context conditions. The pilot concluded that a larger model was needed for meaningful signal.

---

## Pilot 2: Gemma-2-9B-IT (`Actionable_Interp_pilot_gemma_9b.ipynb`)

### What it did

1. **Replicated the behavioral measurement on 9B** with the same semantic choice + continuation scoring method developed in Pilot 1. Tested 7 descriptive proxy conditions (housing stability, physical upkeep, community infrastructure — each with a stable/strong and unstable/weak variant) across 2 scenarios.

2. **Introduced Condition D: neighborhood names.** Tested Boston neighborhood names (Back Bay, Beacon Hill, South End vs. Roxbury, Dorchester, Mattapan) and ZIP codes as implicit geographic cues, without any explicit socioeconomic language.

3. **Ran a reverse-direction experiment.** Instead of neighborhood → policy, tested policy → neighborhood: scored the log-probability of neighborhood descriptors given punitive vs. rehabilitative policy framings, as convergent evidence for asymmetric encoding.

4. **Retrained the framing probe on 9B.** Confirmed that layer 20 remained the best layer (binary accuracy 0.875, 3-way macro-F1 0.885).

5. **Probed at the neighborhood token position.** Extracted hidden states at the exact token position of the neighborhood name (pos 45 at layer 20) and scored them with the retrained probe.

6. **Ran base model comparison (Gemma-2-9B vs. Gemma-2-9B-IT).**

### Key findings that shaped the final design

**→ Why neighborhood names instead of (only) descriptive proxies?**
The descriptive proxy conditions (housing, upkeep, community infrastructure) produced effects, but `community_infra_weak` pushed the model *more* toward prevention than `community_infra_strong` — suggesting the model reads weak infrastructure as a signal for supportive intervention. The proxy effects did not mirror the affluent/under-resourced gradient seen with actual neighborhood names. This confirmed that place names carry richer socioeconomic associations than can be captured by simple descriptive bundles, motivating the name-centric design of the final experiment.

**→ Why ZIP codes were dropped.**
ZIP codes produced uniform effects regardless of the actual socioeconomic valence of the area (~+2.48 for all). The model's place-based associations are name driven, not number driven. ZIP codes were removed from the final experiment.

**→ The discovery of representational dissociation.**
When probing at the neighborhood name token (layer 20), the probe scores moved in the *opposite* direction from behavioral output: Roxbury's probe score was −0.90 (control-leaning) while its behavioral delta was +3.13 (prevention-leaning). This "representational dissociation", where internal representation and external behavior diverge became a central question for the final experiment's multi-position layer sweep.

**→ Base vs. IT comparison revealed the bias is a pretraining artifact.**
The base model already showed a 1.8× ratio of under-resourced to affluent prevention shifts; IT amplified this ~5–6× uniformly while preserving the relative ordering. This finding was replicated more rigorously in the final experiment and informed the narrative that RLHF/instruction tuning amplifies but does not create the place-based bias.

**→ Why the probe training data was expanded from 60 to 100 sentences.**
The 9B probe achieved only 0.875 binary accuracy with the original 60-sentence dataset (vs. 0.90 on 2B). This motivated expanding to 100 balanced sentences (50 control + 50 prevention) in the final experiment, which pushed accuracy above 0.94 at all layers and 0.99 at layers 20–30.

---

## Pilot 3: Gemma-2-9B Audit (`Gemma9b_Audit_pilot_test.ipynb`)

### What it did

This is the most comprehensive pilot and served as the direct blueprint for the final experiment. It ran 100 cells covering:

1. **Expanded behavioral audit.** Scaled from Boston-only to 3 cities (Boston, Chicago, Los Angeles) with 4 neighborhoods each (2 affluent + 2 under-resourced = 12 total). Tested 2 prompt templates per name with order swapping. Also re-ran descriptive proxy conditions.

2. **Multi-position representational sweep at layer 20.** For each neighborhood name prompt, extracted probe scores at three token positions:
   - `name_first_score`: first subtoken of the neighborhood name
   - `name_last_score`: last subtoken of the neighborhood name
   - `answer_delta_from_baseline`: probe score at the final (answer) position, minus the baseline answer-position score

3. **Correlation analysis.** Computed Pearson r between each position's probe score and the behavioral delta from baseline.

4. **Initial activation patching demo.** Cross-substituted hidden states at `name_first` position between affluent and under-resourced names at layer 20 for 4 manually defined pairs.

5. **Initial steering demo at answer position.** Applied additive steering along the mean-difference direction at layer 20 on a small set of test prompts.

6. **Base model replication** of the representational sweep.

### Key findings that shaped the final design

**→ Why 3 cities and why these specific neighborhoods?**
The pilot showed that the name effect was stable across Boston, Chicago, and Los Angeles with consistent affluent < under-resourced ordering within each city. This cross-city robustness justified using these 3 cities in the final experiment (expanded to 18 neighborhoods: 3 per valence per city).

**→ Why `name_first` is the probing position of interest.**
The multi-position sweep found that `name_first_score` was the only position with meaningful positive correlation to behavioral delta (r ≈ 0.54). `name_last_score` and `answer_delta` did not track behavior at layer 20. This motivated the final experiment's focus on `name_first` as the primary encoding site.

**→ Why the final experiment added a full layer sweep (not just layer 20).**
This pilot only tested layer 20. The findings — particularly the weak answer-position signal — raised questions about whether other layers might show different patterns. The final experiment expanded to a 10-layer sweep (5, 10, 15, 18, 20, 22, 25, 30, 35, 40), which revealed the critical sign-flip at layers 20→22 that the single-layer pilot could not detect.

**→ Why patching was done at `name_first` rather than `answer` position.**
The patching demo at `name_first` produced small but directionally consistent effects: patching under-resourced → affluent shifted toward control, and vice versa. Combined with the correlation evidence, this confirmed `name_first` as having nonzero causal leverage, and became the patching site in the final experiment.

**→ Why answer-position steering was also tested in the final experiment.**
The pilot's steering demo at the answer position produced weak, inconsistent effects. But it only tested a few prompts with a single alpha value. The final experiment systematically swept alpha values at both `name_first` and `answer` positions, discovering that answer-position steering actually has stronger dose-response effects than name_first steering (contradicting this pilot's preliminary conclusion).

**→ Why descriptive proxies became a separate "negative control" condition.**
The pilot confirmed that proxy effects do not mirror name effects: affluent-like proxies sometimes produced larger shifts than under-resourced-like proxies, and the ordering was inconsistent across proxy families. In the final experiment, proxies were retained as a negative control to demonstrate that the name effect cannot be trivially explained by surface-level socioeconomic descriptions.

**→ Why token count and semantic transparency were added as moderators.**
The pilot used neighborhoods with varying token counts (1–2 subtokens) and varying name transparency ("Back Bay" is semantically transparent; "Roxbury" is opaque). To rule out these as confounds, the final experiment explicitly coded `token_count` and `semantic_transparency` for each neighborhood and tested them as moderators in the statistical analysis.

---

## Decision Trail: From Pilots to Final Experiment

| Design Decision | Pilot Where Tested | What Was Learned | Final Experiment Choice |
|---|---|---|---|
| Behavioral metric | Pilot 1 (2B) | Single-token logits have severe label/order bias | Continuation log-probability with order-swap averaging |
| Strategy labels | Pilot 1 (2B) | Value-laden labels trigger alignment artifacts | Neutral labels ("Aggregate control" / "Preventive stabilization") |
| Model scale | Pilot 1 (2B) | 2B effects too small, alignment-dominated | Gemma-2-9B-IT (4-bit quantized) |
| Best probe layer | Pilots 1 & 2 | Layer 20 on both 2B and 9B | Layer 20 as anchor; expanded to 10-layer sweep |
| Probe training size | Pilot 2 (9B) | 60 sentences → 0.875 accuracy on 9B | Expanded to 100 sentences → ≥0.94 accuracy |
| Context type | Pilots 2 & 3 | Names carry richer associations than descriptive proxies; ZIP codes are inert | Names as primary IV; proxies as negative control |
| Geographic scope | Pilot 3 | Effects stable across Boston, Chicago, LA | 3 cities × 6 neighborhoods = 18 names |
| Token positions | Pilot 3 | `name_first` correlates with behavior; answer position does not (at layer 20) | Multi-position sweep across 10 layers; discovered layer 20–22 transition |
| Patching site | Pilot 3 | `name_first` has directionally consistent causal effect | `name_first` patching with full cross-pair design |
| Steering site | Pilot 3 | Answer-position steering weak in demo | Systematic alpha sweep at both sites; answer position found causally effective |
| Base vs. IT | Pilot 2 | Bias is a pretraining artifact, IT amplifies uniformly | Full base model replication of Ch1–2 |

---

## How to Read These Notebooks

These are working research notebooks with iterative, exploratory code. They contain dead ends, commented-out experiments, and inline notes. They are not cleaned for reproduction. Te final experiment notebook consolidates all validated methods into a single reproducible pipeline.

The recommended reading order is:

1. **Pilot 1** (2B) — understand the metric development and why alternatives were rejected
2. **Pilot 2** (9B) — understand the shift to neighborhood names and the representational dissociation discovery
3. **Pilot 3** (9B Audit) — understand the multi-city expansion, multi-position sweep, and initial causal experiments
4. **Final experiment** — see how all pilot lessons were integrated into a single four-chapter study

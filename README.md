# Place-Based Framing Bias in LLMs: A Mechanistic Interpretability Study

> **Do neighborhood names cause language models to recommend different policy strategies for the same public safety problem?**

This project investigates how Gemma-2-9B-IT encodes and processes U.S. neighborhood names — and whether these place-based associations systematically shift the model's policy recommendations toward punitive ("control") or community-oriented ("prevention") strategies. Using techniques from mechanistic interpretability (linear probing, activation patching, activation steering, projection-based debiasing), we trace the causal pathway of this bias through the model's internal representations and test whether it can be surgically removed at inference time.

## Key Findings

1. **Neighborhood names shift policy framing.** Under-resourced neighborhood names (e.g., Roxbury, Englewood, Watts) push the model ~0.19 log-prob units further toward "prevention" framing than affluent names (e.g., Back Bay, Lincoln Park, Brentwood) — consistently across 8 safety scenarios, 3 cities, and multiple prompt templates.

2. **A computational transition zone exists at layers 20–22.** Linear probes achieve ≥0.94 accuracy on a control/prevention axis across all layers, but the correlation between probe scores and behavioral bias undergoes a dramatic sign flip between layer 20 (r = 0.62) and layer 22 (r = −0.42) at the answer position — revealing an active internal rewriting of place-based signals.

3. **Activation steering can move overall preferences but cannot selectively close the bias gap.** Steering at the answer position produces strong dose-response shifts in absolute policy preference, but affects affluent and under-resourced conditions equally — the 0.19-unit gap remains.

4. **Three debiasing methods all fail (an informative negative result).** Prompt-based instruction, additive steering, and projection-based debiasing are unable to selectively remove the valence-specific differential. The bias is distributed across high-dimensional representations, not encoded along a single linear direction — setting a concrete boundary on current representation engineering methods.

## Experimental Design

The experiment follows a four-chapter structure, each building on the previous:

### Chapter 1 — Behavioral Measurement

Establishes the phenomenon: does the model treat neighborhoods differently?

- **Task**: Forced-choice policy recommendation (control vs. prevention strategy) for 8 race-neutral urban safety scenarios.
- **Metric**: `prevention_minus_control` — the difference in average log-probabilities between the two completions.
- **Conditions**:
  - **Baseline**: No neighborhood context provided.
  - **Name**: 18 real neighborhood names (3 cities × 3 affluent + 3 under-resourced), each inserted via 2 prompt templates with order-swapped ablation.
  - **Proxy**: Socioeconomic descriptors (housing stability, upkeep, community infrastructure) without explicit place names.
  - **Race ablation**: Explicit racial demographics appended to scenarios.
- **Statistical test**: Permutation test on `delta_from_baseline` (under-resourced vs. affluent).

### Chapter 2 — Representational Analysis

Locates where the bias lives inside the model.

- **Linear probes**: Logistic regression classifiers trained on 100 balanced control/prevention sentences, tested across 10 layers (5–40).
- **Layer sweep**: Extracts hidden states at three token positions (`name_first`, `name_last`, `answer`) for all name conditions; computes Pearson correlation between probe scores and behavioral delta.
- **Key discovery**: The answer position shows high positive correlation at layer 20 that flips to negative at layer 22 — a "computational transition zone."

### Chapter 3 — Causal Verification

Tests whether the identified representations are causally linked to behavior.

- **Activation patching**: Cross-substitutes hidden states between affluent and under-resourced names at the `name_first` position (layers 20 and 22), measuring the resulting shift in policy preference.
- **Activation steering**: Adds `α × direction` to hidden states at `name_first` and `answer` positions, sweeping α to characterize dose-response curves at both layers.

### Chapter 4 — Debiasing Attempts (Negative Result)

Tests whether the bias can be surgically removed.

- **Additive steering**: Applies a calibrated negative α along the mean-difference direction at the answer position.
- **Projection-based debiasing**: Projects out the debiasing direction component: `h' = h - (h · d̂) d̂`.
- **Prompt-based baseline**: Appends "ignore the neighborhood name" to the prompt.
- **Leave-one-city-out cross-validation**: Tests generalization of steering directions across cities.
- **Result**: All methods fail to selectively close the gap, because the valence-specific signal (~14% of the shared shift) is not aligned with the mean-difference direction.

### Base Model Comparison

All Chapter 1–2 experiments are replicated on Gemma-2-9B-base (non-instruction-tuned) to distinguish effects of pretraining from RLHF/instruction tuning.

## Repository Structure

```
.
├── Final_exp__1_.ipynb          # Main experiment notebook (run on Google Colab)
├── README.md
└── Intermediate results/   # Saved outputs (created at runtime)
    ├── act1_checkpoint.json           # Ch1 config & baseline map
    ├── act2_probe_accuracy_it.csv     # Probe accuracy per layer (IT model)
    ├── act3_checkpoint.json           # Causal experiment config
    ├── act4_config.json               # Debiasing config & results
    └── final_summary.json             # Cross-chapter summary
```

## Model & Data

| Component | Details |
|---|---|
| **IT Model** | `unsloth/gemma-2-9b-it-bnb-4bit` (Gemma-2-9B-IT, 4-bit NF4 quantization) |
| **Base Model** | `unsloth/gemma-2-9b-bnb-4bit` (Gemma-2-9B, 4-bit NF4 quantization) |
| **Neighborhoods** | 18 real U.S. neighborhoods across Boston, Chicago, and Los Angeles (9 affluent, 9 under-resourced) |
| **Scenarios** | 8 race-neutral urban safety vignettes (assaults, property damage, theft, vandalism, etc.) |
| **Probe training data** | 100 balanced sentences (50 control-framed, 50 prevention-framed) |

## How to Run

### Prerequisites

- A Google Colab environment with GPU (T4 or above recommended; A100 for faster runs).
- A Hugging Face account with access to Gemma-2 models (requires accepting the license at [google/gemma-2-9b-it](https://huggingface.co/google/gemma-2-9b-it)).

### Steps

1. **Upload the notebook** to Google Colab.

2. **Authenticate with Hugging Face** — the notebook includes a `notebook_login()` cell. You will need a Hugging Face token with read access to gated models.

3. **Mount Google Drive** — results are checkpointed to `~/drive/MyDrive/neighborhood_framing_experiment/`. Modify `SAVE_DIR` if you prefer a different path.

4. **Run cells sequentially.** The notebook is organized into four chapters:
   - **Ch1 (Cells 3–28)**: Behavioral experiments. ~30–45 min on T4.
   - **Ch2 (Cells 29–41)**: Probe training & layer sweep. ~20–30 min.
   - **Ch3 (Cells 42–53)**: Activation patching & steering. ~30–45 min.
   - **Base model comparison (Cells 54–60)**: Releases IT model, loads base, re-runs Ch1–2. ~40 min.
   - **Ch4 (Cells 61–85)**: Debiasing experiments. ~60 min.

5. **Note on memory**: The notebook manages GPU memory by releasing models between the IT and base model phases. If running on a 16GB GPU, avoid loading both models simultaneously.

### Dependencies

All dependencies are installed in the first cell:

```bash
pip install transformers accelerate bitsandbytes sentencepiece pandas matplotlib huggingface_hub scipy scikit-learn
```

Additionally, `hf_transfer` is installed for faster model downloads.

## Core Functions

| Function | Purpose |
|---|---|
| `make_semantic_choice_prompt()` | Constructs the forced-choice policy prompt with configurable order swapping |
| `continuation_logprob()` | Computes average log-probability of a completion given a prompt |
| `score_prompt()` | Scores both completions and returns `prevention_minus_control` |
| `get_last_token_rep()` | Extracts hidden state at a specific layer and token position |
| `continuation_logprob_with_patch()` | Runs a forward pass with a hidden state replacement at a target layer/position |
| `continuation_logprob_with_steering()` | Runs a forward pass with additive steering (`h + α·d`) at a target layer/position |
| `continuation_logprob_with_projection()` | Runs a forward pass with directional projection removed (`h - (h·d̂)d̂`) |

## Methodological Notes

- **Order ablation**: Every behavioral measurement is averaged over both option orderings (control-first and prevention-first) to cancel out position bias in the forced-choice task.
- **Moderator checks**: Token count and semantic transparency of neighborhood names are tested as potential confounds (Ch1).
- **Probe calibration**: Neutral bureaucratic sentences are tested as a sanity check; the probe assigns them strongly to "control," indicating partial capture of an action/procedural style axis rather than a pure policy-framing axis.
- **Negative results reported transparently**: Chapter 4 documents three failed debiasing attempts and provides a mechanistic explanation for why they fail (the valence-specific signal constitutes only ~14% of the total shared shift and is not linearly separable along the mean-difference direction).

## Citation

If you use this work, please cite:

```
@misc{neighborhood-framing-bias-2025,
  title={Place-Based Framing Bias in LLMs: Mechanistic Interpretability of Neighborhood-Driven Policy Recommendations},
  author={Chengyu, Abhishek},
  year={2025},
  url={https://github.com/YOUR_USERNAME/YOUR_REPO}
}
```

## License

[MIT](LICENSE) 

## Acknowledgments

This research was conducted as part of the MS in AI program at Northeastern University, informed by critical technology studies scholarship (Benjamin, Browne, Costanza-Chock) and mechanistic interpretability methods (Turner et al. ActAdd, Hewitt et al., Rudin).

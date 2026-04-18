

### 1. `SAE_Appendix.ipynb` (Exploratory & Feature Analysis)
This notebook contains the early exploratory work and the **Sparse Autoencoder (SAE)** feature discovery process.
* **Stage 1: SAE Feature Discovery:** Using **Gemma Scope** at Layer 13 to identify 10 punitive and 23 rehabilitative features that encode abstract policy concepts rather than surface lexical tokens.
* **Reverse Direction Experiment:** A unique exploratory test where the model is given a policy and asked to "guess" the neighborhood. This provides convergent evidence that the model associates punitive policies with under-resourced areas.
* **Implicit Signal Deep-Dive:** Granular analysis of Boston ZIP codes vs. names, proving that numeric geographic identifiers carry significantly less framing weight than names.

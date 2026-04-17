This `README.md` is designed to provide a technical guide for the supplementary materials of your project. It categorizes your experiments into the five-stage pipeline defined in your research paper and highlights the exploratory "hidden gems" like the Reverse Direction experiment.

***

# Technical Appendix: Auditing the "Carceral Logic"

This directory contains the supplementary exploratory notebooks, raw data, and technical validations supporting the research paper: **"Auditing the 'Carceral Logic': How Neighborhood Names Shape Policy Framing in Large Language Models"**.

## Project Overview
This project investigates whether Large Language Models (LLMs) encode a "carceral logic"—a systematic preference for punitive, control-oriented urban safety policies over rehabilitative ones—triggered by implicit geographic signals like neighborhood names. Our work traces this bias through a multi-stage mechanistic interpretability pipeline: behavioral measurement, representational localization, and causal verification.

## Contents of the Appendix

### 1. `SAE_Appendix.ipynb` (Exploratory & Feature Analysis)
This notebook contains the early exploratory work and the **Sparse Autoencoder (SAE)** feature discovery process.
* **Stage 1: SAE Feature Discovery:** Using **Gemma Scope** at Layer 13 to identify 10 punitive and 23 rehabilitative features that encode abstract policy concepts rather than surface lexical tokens.
* **Reverse Direction Experiment:** A unique exploratory test where the model is given a policy and asked to "guess" the neighborhood. This provides convergent evidence that the model associates punitive policies with under-resourced areas.
* **Implicit Signal Deep-Dive:** Granular analysis of Boston ZIP codes vs. names, proving that numeric geographic identifiers carry significantly less framing weight than names.

## Key Technical Findings

| Phase | Discovery |
| :--- | :--- |
| **The Inversion** | Alignment (RLHF) inverts the base model's punitive response to geographic context into a preventive one. |
| **Representational Dissociation** | Bias is detectable via linear probing at the neighborhood name token, but is causally inert there—it only becomes causally active at the final answer position. |
| **Abstract Features** | SAE analysis identifies internal "neurons" for concepts like "legal proceedings" and "community support" rather than just keyword detectors. |

## Reproducibility
All experiments were conducted using **Gemma-2** (2B and 9B variants) loaded in **4-bit quantization** (NF4) via `BitsAndBytes`. 
* **Environment:** Google Colab Pro / L4 or A100 GPUs.
* **Metric:** Semantic Choice Scoring using the difference in Average Log-Probabilities between policy completions.

***

**Authors:** Chengyu Li & Abhishek Kothari  
**Course:** CS 7180: Special Topics in AI — Actionable Interpretability  
**Institution:** Northeastern University

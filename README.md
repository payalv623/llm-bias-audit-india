# LLM Bias Audit (India / Global South)

A reproducible counterfactual bias audit of four open LLMs. It tests whether they treat people differently in how they **describe people, recommend opportunities, assess competence, and represent regional and cultural contexts**, with a focus on Indian and Global South settings.

This is Question 1 of Assignment 3 for *Responsible AI for Foundation Models (DSE428/DSE628)*, IISER Bhopal.
The report covering all three questions is in [`docs/`](docs): [short version, 4 pages](docs/Assignment3_Short_Report.pdf) and [full report](docs/Assignment3_Report.pdf).

## Design

- **20 hypothesis-driven audit prompts**, covering 10 bias dimensions with 2 prompts each:
  - gender
  - caste (India)
  - religion (India)
  - regional representation
  - Indian English / Hinglish
  - socio-economic status
  - disability
  - age
  - cultural assumptions
  - institutional credibility
- **Counterfactual templates.** Only one attribute changes between groups, e.g. *Amit Sharma* vs *Amit Jatav*, or the same career question in English vs Hinglish.
  - Each group has 6–8 paired names, cities or institutions.
  - Each is sampled twice, giving 720 requests per model and condition, and 8,640 responses in total.
- **Models.** All four run locally in bf16, with Hugging Face commit hashes recorded on every response:
  - Qwen3-8B
  - Phi-4-mini-instruct
  - Granite-3.3-8B-instruct
  - Falcon3-7B-Instruct
- **Independent judge / classifier.** OLMo-2-13B-Instruct, from a developer that is not audited here.
- **Statistics.**
  - The fill (name or institution) is the unit of analysis.
  - Stratified permutation tests within fills.
  - Excess disparity over the permutation noise floor.
  - Cluster-bootstrap confidence intervals.
  - Benjamini–Hochberg FDR correction.
- **Mitigations.**
  - M1: a fairness system prompt.
  - M2: a counterfactual self-check, evaluated with paired permutation tests and an answer-quality judge.

## Files

| Path | What it is |
|---|---|
| `audit_prompts.py` | The audit design: 20 prompt templates, hypotheses, counterfactual groups, metrics, mitigation prompts |
| `bias_audit.py` | Pipeline: `build → generate → judge → score → analyze` (+ figures) |
| `run_full.sh` | Runs the whole pipeline (resumable, retries on transient GPU errors) |
| `results/audit_records.jsonl` / `.csv` | One record per response: prompt, response, parsed JSON, annotations, scores, model + commit, seed, settings, timestamp |
| `results/analysis/*.csv` | Disparity tests, paired differences, parity tests, mitigation effects, quality, scorer agreement |
| `results/figures/` | Figures used in the report |
| `results/human_annotation_sample.csv` | 100 stratified responses for human validation of the automatic scorers |
| `results_v1/` | First version, kept for transparency. An independent review found pseudo-replication and unreliable lexicons; see the report |

## How to run

```bash
pip install -r requirements.txt          # needs a CUDA GPU with >= 28 GB for the 13B judge
PY=python ./run_full.sh                  # or: python bias_audit.py all
```
Models are downloaded from Hugging Face on first use. All four audited models and the judge are ungated.

## Main findings (baseline, significant after FDR correction)

| Finding | Model(s) | Evidence |
|---|---|---|
| Disabled graduates steered to back-office/support roles (P14) | Qwen3-8B, Granite-3.3 | Qwen rates back-office 3.8/10 for the control vs 7.3–8.7 for disabled graduates |
| Family diet guessed from surname (P17) | Phi-4-mini | Non-veg dishes: Tamil-Brahmin-coded family 1.0 vs Western-coded 3.4 of 4 |
| Gendered role allocation (P02) | Qwen3-8B | *Backend Architect* given to a woman 12.5%, *Documentation* 94% (target 50%) |
| First-listed person made Project Lead (P02) | all 4 | 100% for three models (chance 25%) |

**Explicit scores showed no detectable bias:** salary by gender, tenant score by caste, loan risk by religion, hire score by disability and age.

**Mitigation results:**
- No mitigation effect survived correction.
- The fairness prompt (M1) reduced some disparities but created new ones.
- The self-check (M2) changed at most 1.2% of answers.

![Disparity heatmap](results/figures/fig1_disparity_heatmap.png)

## AI use

All code was generated with **Claude Opus 5.5** (Anthropic, `claude-opus-5-5`) through Claude Code, following my prompts. The audit prompts were drafted by the model and approved by me. The complete, verbatim prompt log is in Appendix A of the [full report](docs/Assignment3_Report.pdf).

# AI-Based Exergy Analysis of a Multi-Section Cement Plant Using High-Ash Indian Coal — Reproducibility Package

This repository contains the Aspen Plus model, dataset, analysis code, trained
surrogate, and result files supporting the study:

> **AI-Based Exergy Analysis of a Multi-Section Cement Plant Using High-Ash Indian Coal: A Case Study**

The pipeline performs a section-wise exergy analysis of a six-section dry-process
cement plant (raw mill, coal mill, preheater, calciner, rotary kiln, clinker
cooler) in Aspen Plus, builds a hybrid machine-learning surrogate of the plant,
identifies the dominant operating drivers of irreversibility with SHAP, and
optimises the operating point at constant clinker production using particle
swarm optimisation (PSO) with full Aspen validation.

**Archived at:** https://doi.org/10.5281/zenodo.20717411

---

## 1. Software environment

| Component | Version used |
|-----------|--------------|
| Aspen Plus | V14 |
| Python | 3.11 |
| TensorFlow / Keras | 2.x (CPU) |
| scikit-learn | 1.8.0 |
| SHAP | latest (TreeExplainer + GradientExplainer) |
| pandas, numpy, joblib, matplotlib | standard |
| Aspen COM interface | `win32com` (`pywin32`), Windows only |

The exergy calculation and the campaign/PSO Aspen solves require a licensed
Aspen Plus V14 installation on Windows. The surrogate training (Step 2), SHAP
(Step 3, surrogate part), and the PSO Phase-1 search run on any platform; only
the Aspen validation (Phase 2) and dataset generation need Aspen.

Install Python dependencies:

```
pip install tensorflow scikit-learn shap pandas numpy joblib matplotlib pywin32
```

---

## 2. File manifest

### Aspen model (ground truth)
- `new cmill rmill preheater calciner kiln cooler.bkp` — the Aspen Plus V14
  flowsheet. Everything else can be regenerated from this file.

### Dataset
- `ann_dataset.csv` — the exergy campaign dataset (500 accepted Latin-hypercube
  samples + base-case row; 26 operating inputs and 21 exergy outputs per row,
  plus per-section diagnostic columns and a `status` flag). Train only on rows
  with `status` in {`OK`, `OK_CAPPED`}; `REJECT` rows are genuine second-law
  violations and are excluded.

### Pipeline code (run in this order)
1. `exergy_calculator.py` — standalone base-case exergy accounting. Solves the
   flowsheet via the write-then-solve protocol and prints/streams the per-section
   and plant exergy balance (Table 3 source).
2. `exergy_campaign_driver.py` — generates `ann_dataset.csv` by perturbing the 26
   operating inputs and recomputing the exergy balance for each sample.
3. `basecase_preflight.py` — sanity checks the base case before a campaign
   (branch sentinel, reproducibility, expected balances).
4. `hybrid_train_step2.py` — trains the **adopted hybrid surrogate** (per-section
   selection between an ANN and a Random Forest; see §4).
5. `hybrid_shap_step3.py` — SHAP sensitivity analysis on the hybrid surrogate.
6. `hybrid_pso_step4.py` — constant-duty PSO optimisation with Aspen validation
   (Table 6 source).

### Additional trainers (for the Table 4 model comparison)
- `ann_train_step2.py` — ANN-only surrogate (single multi-output network + six
  per-section networks).
- `rf_train_step2.py` — Random-Forest-only surrogate (21 dedicated per-output
  forests).

These reproduce the ANN-only and RF-only columns of the surrogate-accuracy
comparison; `hybrid_train_step2.py` reproduces the adopted hybrid column.

### Trained surrogate artifacts (`ANN_models/HYBRID/`)
- `hybrid_ann.keras` — the 21-output ANN component.
- `scaler_X.pkl`, `scaler_y.pkl` — input/output standard scalers for the ANN.
- `hybrid_bundle.joblib` — plain dict `{forests[21], routing[21], output_cols,
  input_cols, ann_file, kind}`; the RF forests and the per-output routing map.
- `routing.csv` — which model (ANN/RF) was selected for each output.

Including these lets a user reproduce the SHAP and PSO results *exactly* without
retraining (which has minor run-to-run variation).

### Result files
- `exergy_results.xlsx` — base-case exergy breakdown (Table 3).
- `ANN_models/HYBRID/metrics_report.csv` — per-output test R²/RMSE for ANN, RF,
  and hybrid (Table 4).
- `ANN_models/HYBRID/shap/` — SHAP outputs:
  `shap_importance.csv`, `shap_importance_by_section.csv`,
  `ranked_sensitivity_by_section.csv`, `ranked_sensitivity_table.txt`,
  `top_drivers_summary.csv`, `shap_specific_irrev.csv` / `.txt`.
- `ANN_models/HYBRID/pso/` — optimisation outputs:
  `pso_result.csv` (optimised operating point), `pso_summary.txt`,
  `validate_optimum.csv` / `validate_optimum.txt` (Aspen-validated Table 6).

---

## 3. How to reproduce

All scripts read/write inside the Aspen working folder. Edit the `ASPEN_FOLDER`
path at the top of each script to point at your copy, then run from that folder
so the Aspen COM session and the relative `ANN_models/` paths resolve correctly.

```
# (optional) verify the base case before anything else
python basecase_preflight.py

# 0) base-case exergy balance (Table 3)
python exergy_calculator.py

# 1) generate the dataset  (long: ~8-10 h of Aspen solves; skip if using
#    the provided ann_dataset.csv)
python exergy_campaign_driver.py

# 2) train the hybrid surrogate (Table 4)
python hybrid_train_step2.py
#    optional, for the ANN-only / RF-only comparison columns:
python ann_train_step2.py
python rf_train_step2.py

# 3) SHAP sensitivity analysis
python hybrid_shap_step3.py

# 4) constant-duty PSO optimisation + Aspen validation (Table 6)
python hybrid_pso_step4.py
```

To reproduce only the surrogate/SHAP/PSO results without Aspen, use the provided
`ann_dataset.csv` and `ANN_models/HYBRID/` artifacts and run Steps 2–4 (Step 4
Phase 2 requires Aspen; Phase 1 does not).

---

## 4. The hybrid surrogate (Step 2)

For each plant section, the better of an ANN and a Random Forest is selected on a
held-out **validation** set and reported on a separate held-out **test** set the
selection never saw (70/15/15 split). Tree ensembles capture the multiplicative
drying-duty interactions that dominate the mill sections, while the neural
network better fits the smoother thermal sections. The resulting routing is:

| Section / output | Model |
|------------------|-------|
| Raw mill, coal mill | Random Forest |
| Preheater, calciner, kiln, clinker cooler | ANN |
| Clinker exergy, overall efficiency | Random Forest |
| Specific irreversibility | ANN |

Held-out test accuracy (mean R² over 21 outputs): ANN-only ≈ 0.85,
RF-only ≈ 0.84, **hybrid ≈ 0.93**. The hybrid raises raw-mill and coal-mill
irreversibility prediction to R² ≈ 0.92–0.94 (from ≈ 0.61–0.79 for the ANN).

SHAP (Step 3) explains each output with its routed model — `TreeExplainer` for
RF-routed outputs (raw inputs) and `GradientExplainer` for ANN-routed outputs
(scaled inputs) — with per-output normalisation so the rankings are comparable.

---

## 5. Key modelling conventions

These conventions are applied consistently across the calculator, the dataset,
SHAP, and PSO, so every reported number traces to one self-consistent basis.

- **Single solution branch (write-then-solve).** The recycle/split flowsheet has
  a path-dependent steady state. All results use the write-then-solve protocol
  (re-initialise, write the 26 inputs, then solve), giving a single, reproducible
  base case of **plant exergy destruction ≈ 110.5 MW**.
- **Base case.** Plant irreversibility 110.505 MW; specific irreversibility
  1.6403 MJ/kg clinker (primary metric) = 1.7145 (dimensionless MW/MW, the PSO
  objective); clinker exergy 64.453 MW; clinker mass 67.369 kg/s; overall exergy
  efficiency 37.74% (fixed-fuel basis); total internal exergy production
  50.18 MW; auxiliary electrical work 4.279 MW; coal feed 8.28 kg/s.
- **Fixed-fuel / frozen-electrical efficiency basis.** The exergy-efficiency
  denominator uses the base-case fuel exergy (166.513 MW) and the base-case
  auxiliary electrical work (4.279 MW), both held constant. At fixed firing and
  production these inputs are essentially invariant; freezing them prevents
  off-design heat-block sign reversals from corrupting the efficiency, so
  efficiency is reported as held constant at constant duty.
- **Constant-duty optimisation.** Because the flowsheet sets clinker by raw-meal
  stoichiometry (not fuel–clinker heat coupling), PSO is run at fixed throughput:
  raw-meal feed, coal feed, meal split, coal split, and calciner temperature are
  pinned, and clinker is held within ±2% of base. The search box is ±10% around
  the base operating point.
- **Specific irreversibility is the primary metric** (MJ/kg clinker = GJ/tonne);
  the dimensionless ratio (sum of section irreversibilities / clinker exergy) is
  the PSO objective.

---

## 6. Headline results

- **Base case (Table 3):** 110.505 MW plant irreversibility; 1.6403 MJ/kg
  specific irreversibility; 37.74% exergy efficiency; clinker exergy 64.453 MW;
  exact boundary closure.
- **Surrogate accuracy (Table 4):** ANN 0.85 / RF 0.84 / **Hybrid 0.93** mean
  test R², with the per-section routing above.
- **Dominant drivers (SHAP):** per-section irreversibility is governed by the raw
  mill dryer/gas temperatures (raw mill), gas-heater/coal-mill temperatures (coal
  mill), and calciner temperature with coal feed/split (thermal sections);
  plant-wide specific irreversibility is dominated by raw-meal throughput.
- **Constant-duty optimisation (Table 6):** specific and total irreversibility
  reduced by **≈ 3.9%** (110.505 → 106.190 MW) with clinker production held
  constant and efficiency held at 37.7%; the largest sectional gain is in the
  raw mill (≈ −34%). The optimum is envelope-bounded (several inputs at the ±10%
  box edges).

---

## 7. Citation

If you use this dataset or code, please cite the associated article and this
Zenodo record (https://doi.org/10.5281/zenodo.20717411).

## 8. License and contact

The Aspen model represents a generic dry-process cement plant configuration. For
questions about the model or pipeline, contact the corresponding author of the
associated publication.

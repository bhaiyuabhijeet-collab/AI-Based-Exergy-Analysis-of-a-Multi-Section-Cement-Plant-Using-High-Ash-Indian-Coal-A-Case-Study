# AI-Based Exergy Analysis of a Multi-Section Cement Plant Using High-Ash Indian Coal

This repository contains the dataset and analysis code supporting the paper:

> **AI-Based Exergy Analysis of a Multi-Section Cement Plant Using High-Ash Indian Coal: A Case Study**
> *(under review, Cogent Engineering)*

The work couples a validated Aspen Plus V14 process model of a six-section dry-process
cement plant to a Python exergy calculator, trains artificial neural-network (ANN)
surrogates of the section-level exergy metrics, ranks the dominant operating drivers by
SHAP sensitivity analysis, and minimises the plant specific irreversibility using
particle-swarm optimisation (PSO).

---

## Overview of the workflow

The analysis proceeds in four steps:

1. **Dataset generation** — a 500-sample Latin-Hypercube design over 26 independent
   operating variables (±10 % about the validated base case) is solved in Aspen Plus,
   and each converged case is evaluated by the exergy calculator to produce 18
   section-level exergy outputs (irreversibility, functional efficiency, and improvement
   potential for each of the six sections), plus the clinker product exergy.
2. **ANN surrogate training** — feed-forward neural networks are trained to predict the
   exergy outputs from the 26 inputs.
3. **SHAP sensitivity** — the trained surrogate is interrogated to rank the inputs that
   most strongly drive each section's exergy destruction.
4. **PSO optimisation** — the surrogate is embedded in a particle-swarm optimiser that
   minimises the specific irreversibility (total exergy destruction per unit clinker
   exergy) within the validated operating envelope.

---

## Repository contents

### Data
| File | Description |
|------|-------------|
| `ann_dataset.csv` | The 500 physically-consistent samples used for training. Columns: the 26 input variables, the 18 section exergy outputs, the clinker product exergy, and per-sample status/diagnostic flags. |

### Code
| File | Step | Description |
|------|------|-------------|
| `exergy_calculator_v2.py` | — | Core exergy calculator. Computes per-stream and per-section physical and chemical exergy from Aspen Plus results using a dead-state reference-stream method. Imported by the campaign driver. |
| `exergy_campaign_driver.py` | 1 | Drives the Aspen Plus model over the Latin-Hypercube design, calls the exergy calculator on each converged case, applies physical-consistency screening, and writes `ann_dataset.csv`. |
| `ann_train_step2.py` | 2 | Trains the ANN surrogates (a single multi-output network and six per-section networks), writes the models, scalers, accuracy metrics, and parity / R² plots. |
| `ann_shap_step3.py` | 3 | Computes SHAP importances over the 26 inputs for every output, writes the ranked-sensitivity tables and SHAP plots. |
| `ann_pso_step4.py` | 4 | Runs the particle-swarm optimisation on the trained surrogate, minimising specific irreversibility, and writes the optimised operating point, performance summary, and convergence plot. |

---

## Requirements

- **Python** 3.9–3.12 (64-bit)
- **Aspen Plus V14** — required only to *generate* new data with
  `exergy_campaign_driver.py`. The trained surrogates, SHAP, and PSO steps run from the
  provided `ann_dataset.csv` and do **not** require Aspen.

Python packages:

```
numpy
pandas
scikit-learn
joblib
matplotlib
tensorflow        # or tensorflow-cpu
shap
```

Install with:

```
pip install numpy pandas scikit-learn joblib matplotlib tensorflow shap
```

---

## How to reproduce the results

The data-driven steps reproduce the paper's results directly from the provided dataset;
Aspen Plus is not needed unless you wish to regenerate the dataset.

1. **(Optional) Regenerate the dataset** — requires Aspen Plus V14 and the process model.
   Set the model path inside `exergy_campaign_driver.py`, then:
   ```
   python exergy_campaign_driver.py
   ```

2. **Train the ANN surrogates:**
   ```
   python ann_train_step2.py
   ```

3. **Run the SHAP sensitivity analysis:**
   ```
   python ann_shap_step3.py
   ```

4. **Run the PSO optimisation:**
   ```
   python ann_pso_step4.py
   ```

Each script writes its outputs (models, tables, and figures) to an `ANN_models/`
sub-folder. Paths are set in a configuration block at the top of each file and should be
edited to match your local directory.

---

## Notes and reproducibility

- A fixed random seed (42) is used throughout for reproducibility. Minor numerical
  differences between runs may arise from floating-point ordering and hardware-specific
  library optimisations, and do not affect the conclusions.
- The surrogates, SHAP rankings, and optimisation results are valid only within the
  ±10 % operating envelope on which the model was trained, and should not be extrapolated
  beyond it without re-validation.
- The chemical-exergy reference values for the clinker mineral phases are derived from the
  Hanein et al. (2020) thermodynamic dataset; see the paper's appendices for full details
  and provenance.

---

## Citation

If you use this dataset or code, please cite the paper:

```
[Author(s)], AI-Based Exergy Analysis of a Multi-Section Cement Plant Using
High-Ash Indian Coal: A Case Study, Cogent Engineering, [year].
```

and this repository via its archived DOI:

```
[DOI to be inserted after deposit on Zenodo]
```

---

## License

This dataset and code are released under a permissive open license to permit reuse, in
line with the journal's open-data policy. Recommended: **MIT** for the code and
**CC-BY 4.0** for the data. Add the corresponding `LICENSE` file before depositing.

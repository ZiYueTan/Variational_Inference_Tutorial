# Variational Inference Methods for Single-Cell Genomics

Demonstration code and notebooks for the paper **Variational Inference Methods for Single-Cell Genomics**.

This repository reproduces the simulations, single-cell probabilistic inference (scPI) examples, and temporal GP analyses.

## Repository structure

```text
01_simulation/
  01_LMM/
    LMM.py
    LMM_simulation.ipynb
  02_scLDA/
    LDA.py
    ScLdaSimData.py
    LDA_simulation.ipynb
    run_paper_cavi_svi_mse.py
    run_paper_pyro.py
  03_GLMM/
    GLMM.py
    GLMM_simulation.ipynb

02_scPI/
  FA.py
  ZIFA.py
  01_computational_time.ipynb
  02_performance_comparison.ipynb

03_TemporalGP/
  GP.py
  utils.py
  01_compare.ipynb
  02_leave_cohort.ipynb
```

## Contents

### `01_simulation/`

- **`01_LMM/`** — Linear mixed model estimators: EM, PX-EM, MM, and mean-field CAVI. 
- **`02_scLDA/`** — Single-cell LDA with conjugate CAVI/SVI and black-box Pyro `AutoNormal` SVI. 
- **`03_GLMM/`** — Grouped Bernoulli GLMM estimators: Laplace, PQL, and Pyro VI.

### `02_scPI/`

- **`FA.py`** — Factor analysis with amortized VI or non-amortized VI (`method="amortized"` / `"vi"`).
- **`ZIFA.py`** — Zero-inflated FA with classic EM, block EM, or Pyro VI (`method="classic"` / `"block"` / `"pyro"`, plus amortized vs non-amortized inference).
- **`01_computational_time.ipynb`** — Runtime benchmarks over cell/gene sizes using the mouse brain 10x matrix at `datasets/mouse_brain/datasets/1M_neurons_filtered_gene_bc_matrices_h5.h5`.
- **`02_performance_comparison.ipynb`** — Cortex imputation and clustering comparison using `expression_mRNA_17-Aug-2014.txt`.

### `03_TemporalGP/`

- **`GP.py`** — Temporal count models in Pyro: `GP_MF`, `GP_Full-rank`, and `Indep_MF`.
- **`utils.py`** — Age standardization and RBF temporal kernel helpers.
- **`01_compare.ipynb`** — Fit the three models on Microglia from `datasets/aging_svz_adata.h5ad` across gene-panel sizes.
- **`02_leave_cohort.ipynb`** — Leave-cohort experiment: hold out each cohort, then compare missing time-point estimates to full-data baselines.


## How to run

1. Install the Python dependencies used by the notebooks you plan to run (`numpy`, `scipy`, `pandas`, `matplotlib`, and for Pyro-based sections also `torch`, `pyro-ppl`; TemporalGP / scPI notebooks additionally use `anndata`, `h5py`, and `scikit-learn` as needed).
2. Place required external datasets under the paths noted above.
3. Open and run the notebooks in order within each folder.
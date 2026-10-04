# Robust Satisficing Ensemble (RSE)

Official implementation and experimental code for **RSE**, a **NeurIPS 2026 Spotlight** paper.

This repository contains the code used to reproduce our experiments on distribution shifts across **SST-5, TREC, CivilComments, HuffPost, and Selective Copying**.

## Data

The model prediction files used in our experiments are not included in the repository because of their size.

You can either:

- **Download the precomputed prediction files:**  
  https://figshare.com/s/2b430c1bd59db680dbfa

- **Generate them yourself** using the corresponding `llm_run_*.py` scripts:
  ```bash
  llm_run_sst5.py
  llm_run_trec.py
  llm_run_civilcomments.py
  llm_run_huffpost.py
  llm_run_selective_copying.py
  ```

Set the desired model in each script before running it. A GPU environment such as Google Colab is recommended for generating model predictions.

## Repository Structure

Place the downloaded or generated JSON files in their corresponding dataset folders:

```text
.
├── rse.py
├── erm.py
├── irm.py
├── kl_rs.py
├── groupdro_ensemble.py
├── srm_ensemble.py
├── helpers.py
│
├── run_sst5.py
├── run_trec.py
├── run_civilcomments.py
├── run_huffpost.py
├── run_selective_copying.py
│
├── sst5/
├── trec/
├── civil_comments/
├── huffpost/
└── selective_copying/
```

## Reproducing the Main Experiments

Run the corresponding script from the repository root:

```bash
python run_sst5.py
python run_trec.py
python run_civilcomments.py
python run_huffpost.py
python run_selective_copying.py
```

Each script loads the model prediction files and evaluates RSE together with the corresponding baseline methods under the distribution-shift setting used in the paper.

## Additional Analyses

### $\tau$ Sensitivity

For the label-shift experiments:

```bash
python tau_analysis_sst5.py
python tau_analysis_trec.py
```

### Model-Pool Sensitivity

For the model-pool composition and robustness experiments:

```bash
python ensemble_diversity_sst5.py
python ensemble_diversity_trec.py
```

These experiments assess the robustness of RSE to variations in ensemble composition, including model removal, reduced model pools, and the inclusion of weak or correlated predictors.

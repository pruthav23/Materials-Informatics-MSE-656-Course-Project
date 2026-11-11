# Materials-Informatics-MSE-656-Course-Project
This repository contains all the codes and resources, required to complete our project "Prediction of Grain Boundary Segregation Energy".

<img width="1267" height="722" alt="image" src="https://github.com/user-attachments/assets/c1c6ad28-979d-42a2-86eb-8f37f8a6b530" />


Members:
Prutha Vinay,
Basudev Mohapatra,
Bakare Shivraj Shivaji,
Khushal Karadiya and Gaurav Kumar.


## Revised pipeline (November 2026)

The `revised_pipeline/` folder is a rebuilt, reproducible version of the project. Start with
[`revised_pipeline/GBSE_revised_pipeline.ipynb`](revised_pipeline/GBSE_revised_pipeline.ipynb), which runs end to end
(about 5 minutes on a laptop) from `wagih_2020_data.csv`.

**What changed from the original project**
- **One consistent feature source:** elemental properties for all 25 elements come from pymatgen's tabulated
  periodic-table data, replacing DFT values for 6 metals plus hand-entered values for the other 19. Missing or
  inconsistent elastic constants (Mn, Mo, P, Si, Zr) are completed with isotropic-elasticity relations, and every fix is logged.
- **Physically meaningful descriptors:** mismatch features are computed from raw values before scaling (the original
  computed ratios of standardized values), adding atomic size misfit, the McLean elastic misfit energy,
  electronegativity difference and melting point ratio.
- **No target leakage:** σ, α, the fit R² and Potential_ID come from the same simulations as μ and are excluded.
- **Honest evaluation:** 5-fold cross-validation grouped by alloy pair (no alloy appears in both training and test data),
  repeated 5 times, with scaling fit inside each training fold.

**Results** (out-of-fold, mean ± sd over 5 repeats)

| Model | R² | RMSE (kJ/mol) | MAE (kJ/mol) |
|---|---|---|---|
| Baseline (predict the mean) | −0.01 ± 0.01 | 20.9 | 11.9 |
| Ridge regression | 0.18 ± 0.01 | 18.9 | 11.1 |
| Gradient boosting | 0.18 ± 0.03 | 18.9 | 10.7 |
| GIN graph neural network | 0.15 ± 0.04 | 19.2 | 11.4 |
| **Random forest** | **0.24 ± 0.02** | **18.2** | **10.0** |

**Key findings**
- 243 of the 427 rows are alloy pairs simulated with more than one interatomic potential: identical features, different μ.
  This caps any elemental-feature model at **R² ≈ 0.72**; the random forest reaches about a third of that ceiling
  (R² 0.33 on the pair-averaged target).
- Errors are concentrated in extreme alloys: MAE is 6.7 kJ/mol for 90% of alloys but 40.6 kJ/mol for the most extreme 10%.
- The most important features are the elastic misfit energy and the solute–solvent shear modulus ratio, consistent with
  elastic theory of segregation.
- On 2-node alloy graphs the GIN behaves like a symmetric neural network over element pairs and does not beat tree
  ensembles; a graph approach would need atomic-scale grain boundary structures.

The GIN is implemented from scratch with `autograd`; `pipeline.py` and `features.py` are the notebook's source, and
`build_notebook.py` regenerates the notebook from them.

## Original course project (2025)

Contents:

processed_features.csv: This csv file was used as the primary dataset for training the models, namely Linear Regression, DT and Random Forests, and finally Graph Neural Networks.

GB_Energies_datasets.ipynb: Codes for Data cleaning and merging of various datasets, to produce the final raw dataset.

Grain_boundary_features.ipynb: Codes for Data preprocessing. This produces the final, clean dataset i.e., "processed_features.csv".

Linear_Regression_and_its_Variations.ipynb: Training a linear regression model on the primary dataset.

DT and Random Forest.ipynb: Training a DT and Random Forest model on the primary dataset.

Graph Neural Networks.ipynb: Training GNNs on the primary dataset.

Description: 

This project predicts grain boundary segregation energies in alloys using data-driven models. It integrates feature-engineered ML methods and graph neural networks to learn atomic-level interactions. Compared to traditional DFT calculations, the pipeline offers faster screening of alloy systems. Note: the original notebooks evaluate on a single random split and include σ (Sigma), which comes from the same simulation as the target; see the revised pipeline above for leakage-free, grouped cross-validation results.

`GB_Energies_datasets.ipynb` now reads the Materials Project API key from the `MP_API_KEY` environment variable instead of hard-coding it.



# When Unsupervised Clustering Fails: A Knowledge-Based Alternative for Biomarker Analysis in Small Athlete Cohorts

[![Python 3.13](https://img.shields.io/badge/Python-3.13-blue.svg?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3130/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Open Science](https://img.shields.io/badge/Open-Science-success)](https://en.wikipedia.org/wiki/Open_science)

This repository contains the analysis pipeline for the manuscript **"When Unsupervised Clustering Fails: A Knowledge-Based Alternative for Biomarker Analysis in Small Athlete Cohorts"**, submitted to *Computer Methods and Programs in Biomedicine*.

## Overview

Athlete monitoring with blood biomarkers routinely operates on small cohorts (often fewer than 20–30 athletes). This project:

1. **Tests directly**, rather than assumes, whether unsupervised structure discovery (clustering) is statistically defensible at this sample size, using two independent methods: Ward hierarchical clustering (across four validity indices) and a Bayesian nonparametric mixture model (tested for component-count convergence).
2. **Proposes an alternative** when it is not: a literature-informed evidence-combination score that classifies athletes against externally published, population-matched reference distributions (creatine kinase and C-reactive protein), estimating no parameter from the evaluated cohort.
3. **Validates this classification** on an independent external cohort (a second, publicly published dataset), without refitting any parameter.

An earlier version of this project used unsupervised clustering with synthetic-data augmentation as its primary contribution. A post-submission reanalysis found that pipeline's results unstable and not statistically supported at this sample size; it is preserved, unmodified, under the git tag [`v1-original-submission`](../../releases/tag/v1-original-submission) for transparency, but is **not** part of the current, recommended pipeline below.

## Repository Structure

```text
├── notebooks/v2_reanalysis/
│   ├── common.py                              # Shared data loader (no hardcoded/real paths)
│   ├── 01_outlier_screening_mahalanobis_pca.py # PCA-reduced Mahalanobis outlier screening
│   ├── 02_ward_clustering_validity_indices.py  # Ward + Silhouette/Calinski-Harabasz/Davies-Bouldin/Gap
│   ├── 03_bayesian_mixture_convergence.py      # BayesianGaussianMixture, component-ceiling convergence test
│   └── 04_probabilistic_evidence_score.py      # Literature-informed evidence-combination score
│
├── external_validation/
│   ├── external_validation_anchor.py      # Applies the same scoring logic to an independent external cohort
│   └── metabolites_external_cohort.csv    # Public data (Du et al., Metabolites 16(6):370, 2026)
│
├── utils/
│   └── generate_real_world_seed.py   # Generates the public synthetic demo dataset below
│
├── database/
│   └── dataset_real_world_example.csv   # Synthetic demo dataset (see Data Availability below)
│
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/FBRosito/unsupervised-athlete-biomarker-clustering.git
cd unsupervised-athlete-biomarker-clustering
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running the Pipeline

Each script in `notebooks/v2_reanalysis/` defaults to `database/dataset_real_world_example.csv`, a synthetic dataset that mirrors the real cohort's column structure but **does not reproduce the manuscript's numbers** (see Data Availability below). Pass `--data-path` to point at a real cohort file with the same column structure.

```bash
cd notebooks/v2_reanalysis
python 01_outlier_screening_mahalanobis_pca.py
python 02_ward_clustering_validity_indices.py
python 03_bayesian_mixture_convergence.py
python 04_probabilistic_evidence_score.py
```

Results are written to `results_v2/` (not versioned; regenerated on each run).

To regenerate the public synthetic demo dataset itself:

```bash
python utils/generate_real_world_seed.py
```

### External validation

```bash
cd external_validation
python external_validation_anchor.py
```

This applies the exact same scoring logic (same reference parameters, same attention cutoff) to a second, independent, publicly published cohort, with no parameter refit — a genuine out-of-sample application. See the script's docstring for the data source.

## Data Availability

The biomarker measurements analyzed in the manuscript constitute sensitive health data under Brazilian law (LGPD) and are not publicly distributed. The study protocol was approved by the Institutional Research Ethics Committee of União Educacional do Norte (UNINORTE), number 5.159.860. This repository's code is public under an MIT license, together with a synthetic demonstration dataset (`database/dataset_real_world_example.csv`) that matches the real cohort's structure but not its values. Aggregate results may be shared with researchers upon reasonable request, subject to institutional and ethical approval.

## Authors

* Fernando Barcelos Rosito
* Sebastião De Jesus Menezes
* Simone Ferreira Sturza
* Adriana Seixas
* Muriel Figueredo Franco

**Affiliation:** Federal University of Health Sciences of Porto Alegre (UFCSPA), Brazil.

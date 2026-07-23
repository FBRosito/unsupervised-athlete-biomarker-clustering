# An Unsupervised Decision-Support Framework for Multivariate Biomarker Analysis in Athlete Monitoring

[![Python 3.13](https://img.shields.io/badge/Python-3.13-blue.svg?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3130/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/FBRosito/unsupervised-athlete-biomarker-clustering/graphs/commit-activity)
[![Open Science](https://img.shields.io/badge/Open-Science-success)](https://en.wikipedia.org/wiki/Open_science)

> **⚠️ Update following manuscript revision (2026-07-22):** A post-submission reanalysis found that the synthetic-seed augmentation pipeline originally used in this project (Gaussian Mixture Model expansion from a small seed dataset) produced results that were not stable across the model's own random initialization, and could not be reliably traced to a documented, justified seed-construction methodology. This pipeline is preserved in this repository under the tag `v1-original-submission` for transparency, but **should not be used to draw conclusions about discovered physiological phenotypes**. The current recommended pipeline uses (1) multivariate outlier screening via PCA-reduced Mahalanobis distance and (2) a literature-informed evidence-combination score in place of unsupervised clustering, documented in [`notebooks/v2_reanalysis/`](notebooks/v2_reanalysis/) (`01_outlier_screening_mahalanobis_pca.py`, `02_ward_clustering_validity_indices.py`, `03_bayesian_mixture_convergence.py`, `04_probabilistic_evidence_score.py`). See the revised manuscript for full methodological detail and rationale. The example dataset in `database/dataset_real_world_example.csv` is a synthetic stand-in for illustrating the pipeline's mechanics and does not represent real athlete measurements; the real, non-public cohort is described but not distributed, per Section 7 of the manuscript (Data Availability).

This repository contains the source code, data augmentation pipelines, and validation notebooks associated with the research paper: **"An Unsupervised Decision-Support Framework for Multivariate Biomarker Analysis in Athlete Monitoring"**. *(Manuscript title/framing under revision — see the update notice above; the original submission's framing is preserved below and in the `v1-original-submission` tag.)*

## 📌 Project Overview

Traditional athlete monitoring often relies on univariate analysis (single biomarkers) or subjective questionnaires. This project proposes a **Multivariate Unsupervised Machine Learning Framework** to identify latent physiological states in athletes without relying on labeled injury data.

By analyzing **up to 32 biomarkers** simultaneously (Enzymatic, Hormonal, Inflammatory, and Metabolic), the framework provides a hierarchical decision-support system:
* **Macro-View ($k=3$):** For Head Coaches (e.g., distinguishing *Ready* vs. *Fatigued*).
* **Micro-View ($k=5$):** For Physiologists (e.g., distinguishing *Mechanical Damage* from *Metabolic Stress*).

## 📂 Repository Structure

```text
├── notebooks/
│   ├── 01_Phase1_RealWorld_SmallSample_HCA.ipynb    # Original Phase 1 HCA analysis (v1-original-submission)
│   └── v2_reanalysis/                               # CURRENT RECOMMENDED PIPELINE (see notice above)
│       ├── common.py                                    # Shared data loader, no hardcoded/real paths
│       ├── 01_outlier_screening_mahalanobis_pca.py       # PCA-reduced Mahalanobis outlier screening
│       ├── 02_ward_clustering_validity_indices.py        # Ward + Silhouette/CH/DB/Gap Statistic
│       ├── 03_bayesian_mixture_convergence.py            # BayesianGaussianMixture, ceiling convergence test
│       └── 04_probabilistic_evidence_score.py            # Literature-informed evidence-combination score
│
├── supplementary_studies/
│   └── Supplementary_Comparison_RealData_KMeans.ipynb    # Comparison proving K-Means instability on small data
│
├── deprecated/                          # v1-original-submission augmentation pipeline, kept for transparency
│   ├── generate_random_seed_data.py         # (was utils/generate_random_seed_data.py)
│   └── 02_Phase2_Sensitivity_Validation.ipynb   # (was notebooks/02_Phase2_Sensitivity_Validation.ipynb)
│
├── utils/
│   └── generate_real_world_seed.py     # Generates the public demo dataset (still in active use)
│
├── database/                           # Folder where generated CSVs are stored
│   ├── (dataset_real_world_example.csv)     # used by v2_reanalysis/ (default) and notebook 01
│   └── (dataset_example_structure.csv)      # used only by the deprecated/ augmentation pipeline
│
├── results_v2/                         # Output of notebooks/v2_reanalysis/ scripts (not versioned)
├── requirements.txt                    # Python dependencies
└── README.md                           # Project documentation
```

## 🚀 Key Features & Methodology

### Current recommended pipeline (`notebooks/v2_reanalysis/`)
* **Challenge:** at n=22, neither frequentist (Ward + 4 validity indices) nor Bayesian nonparametric (Dirichlet-process mixture, tested for convergence across multiple component ceilings) clustering supports statistically defensible discovery of latent phenotypes — both converge on the same negative finding, independently.
* **Solution:** multivariate outlier screening via PCA-reduced Mahalanobis distance, plus a **literature-informed evidence-combination score** — reference distributions/thresholds from published, sport-matched literature, combined under an explicit conditional-independence assumption, with uncertainty propagated from the source literature itself (not from the evaluated sample).
* **Outcome:** a stable, interpretable readout without any synthetic data augmentation. See the revised manuscript for full results.

---
*Sections below describe the original submission's methodology (Phase 1 / Phase 2), preserved for transparency and reproducibility of `v1-original-submission`. Phase 2 in particular (GMM augmentation, "Silent Risk" phenotype) is superseded by the pipeline above — see the update notice at the top of this README.*

### Phase 1: Real-World Small Cohort ($n=22$) — *v1-original-submission*
* **Challenge:** Small sample size prevents the use of density-based algorithms.
* **Solution:** **Agglomerative Hierarchical Clustering (Ward's Linkage)**.
* **Outcome:** Stable identification of physiological profiles and outlier detection using Dendrograms.

### Phase 2: Scalability & Sensitivity ($n=290$) — *v1-original-submission, deprecated*
* **Challenge:** Validating the model's ability to detect subtle risks.
* **Solution:** **Regularized Gaussian Mixture Models (GMM)** for data augmentation ($n=15 \to n=290$) followed by K-Means.
* **Outcome:** Originally reported as isolating a **"Silent Risk"** phenotype (High Insulin/Homocysteine with Normal CK). A post-submission reanalysis found this pipeline's results unstable across the augmentation model's own random initialization — see the update notice at the top of this README and `deprecated/`.

## 🛠️ Installation & Usage

To replicate the study, follow these steps strictly to generate the necessary synthetic data.

### 1. Clone and Install
```bash
git clone [git@github.com:FBRosito/unsupervised-athlete-biomarker-clustering.git](git@github.com:FBRosito/unsupervised-athlete-biomarker-clustering.git)
cd unsupervised-athlete-biomarker-clustering

# Create virtual environment (Optional but recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. ⚠️ Generate Data (Crucial Step)
Since raw data is not shared for privacy reasons, **you must generate the synthetic "safe seed"** for the pipeline to run.

```bash
# Generate the public demo dataset (18 biomarkers, n=22) used by notebooks/v2_reanalysis/
python utils/generate_real_world_seed.py
```

Only needed if you also want to run the deprecated Phase 2 augmentation pipeline (see the update notice above — not recommended for new use):
```bash
python deprecated/generate_random_seed_data.py
```

### 3. Run the current recommended pipeline
```bash
cd notebooks/v2_reanalysis
python 01_outlier_screening_mahalanobis_pca.py
python 02_ward_clustering_validity_indices.py
python 03_bayesian_mixture_convergence.py
python 04_probabilistic_evidence_score.py
```
Each script defaults to `database/dataset_real_world_example.csv` (the synthetic demo dataset — running against it will **not** reproduce the manuscript's numbers, only illustrate the pipeline's logic; see Section 7, Data Availability Statement). Pass `--data-path` to point at a real cohort file with the same column structure. Results are written to `results_v2/`.

### 4. (Optional) Explore the original submission's notebooks
Launch Jupyter Notebook and navigate to the `notebooks/` or `deprecated/` folder:

```bash
jupyter notebook
```

## 🔒 Data Privacy & Ethics

Due to the sensitive nature of biological data from human subjects and ethical constraints (GDPR/LGPD), the **raw datasets containing real athlete biomarkers are not shared publicly**.

To ensure reproducibility:
1.  **Synthetic Seeds:** The repository uses statistically plausible mock data generated by the scripts in `utils/`.
2.  **Identical Structure:** These files mirror the exact column structure and units of the real data, allowing the code to be audited and reused.

*Researchers wishing to access the anonymized real datasets for meta-analysis may contact the corresponding author upon reasonable request and ethical approval*.

## 👥 Authors

* **Fernando Barcelos Rosito**
* Sebastião De Jesus Menezes
* Simone Ferreira Sturza
* Adriana Seixas
* **Muriel Figueredo Franco**

**Affiliation:** Federal University of Health Sciences of Porto Alegre (UFCSPA), Brazil.

---
*This project is part of an initiative to bring Transparent AI to Sports Science.*

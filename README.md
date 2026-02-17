# An Unsupervised Decision-Support Framework for Multivariate Biomarker Analysis in Athlete Monitoring

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Submitted-orange)
![Open Science](https://img.shields.io/badge/Open-Science-succes)

This repository contains the source code, data augmentation pipelines, and validation notebooks associated with the research paper: **"An Unsupervised Decision-Support Framework for Multivariate Biomarker Analysis in Athlete Monitoring"**.

## 📌 Project Overview

Traditional athlete monitoring often relies on univariate analysis (single biomarkers) or subjective questionnaires. This project proposes a **Multivariate Unsupervised Machine Learning Framework** to identify latent physiological states in athletes.

By analyzing **32 biomarkers** simultaneously (including Enzymatic, Hormonal, Inflammatory, and Metabolic markers), the framework provides a hierarchical decision-support system:
* **Macro-View ($k=3$):** For Head Coaches (e.g., *Ready* vs. *Fatigued*).
* **Micro-View ($k=5$):** For Physiologists (e.g., distinguishing *Mechanical Damage* from *Metabolic Stress*).

## 🚀 Key Features

* **Robust Data Augmentation:** Implements a Regularized Gaussian Mixture Model (GMM) with diagonal covariance constraints to expand small datasets ($n=15 \to n=290$) without overfitting.
* **Sensitivity Validation:** Proves the algorithm's ability to detect "Silent Risk" phenotypes (High Insulin/Homocysteine with Normal CK).
* **Interpretable Dashboards:** Generates Heatmaps and PCA topologies to translate complex Z-scores into actionable insights.
* **Privacy-Preserving:** Designed to work with anonymized seeds, ensuring compliance with GDPR/LGPD.

## 📂 Repository Structure

```text
├── notebooks/
│   ├── 01_synthetic_augmentation_validation.ipynb  # Sensitivity analysis & GMM Robustness (The "Silent Risk" Proof)
│   ├── 02_main_clustering_pipeline.ipynb           # Main analysis pipeline (K-Means/Hierarchical)
│
├── utils/
│   ├── generate_random_seed_data.py                # Utility script to generate safe mock data
│
├── dataset_example_structure.csv                   # Synthetic seed file (Safe for public use)
├── requirements.txt                                # Python dependencies
└── README.md                                       # Project documentation
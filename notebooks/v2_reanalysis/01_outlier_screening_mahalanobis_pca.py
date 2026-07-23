"""
v2 reanalysis - Outlier screening via PCA-reduced Mahalanobis distance.

Corresponds to Section 3.2 (Multivariate Outlier Screening) of the revised manuscript.
Supersedes the original submission's outlier step (rank-based removal of the top-2 most
distant subjects, regardless of their actual distance -- see v1-original-submission tag),
which had no statistically defensible threshold behind it.

Method and rationale (see manuscript Section 3.2 for full detail):
- Raw Mahalanobis distance requires inverting the p x p covariance matrix. With n close to p
  (as in this cohort), that matrix is near-singular, making raw Mahalanobis numerically
  unstable. PCA components explaining >=90% of variance are retained first to obtain a
  well-conditioned covariance estimate; Mahalanobis distance is computed in that reduced space.
- Cutoff: chi-square distribution, df = number of retained components, alpha = 0.001, a
  conventional threshold in multivariate outlier detection (Barnett & Lewis; Aggarwal,
  "Outlier Analysis"), fixed independently of this dataset.

By default this runs against the public synthetic demo dataset (see common.py) and will NOT
reproduce the manuscript's numbers -- only the real, non-public cohort does that (Section 7,
Data Availability Statement).

Usage:
    python 01_outlier_screening_mahalanobis_pca.py [--data-path PATH] [--alpha 0.001]
"""
import json
import os
import numpy as np
from scipy.stats import chi2
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import MinCovDet, EmpiricalCovariance

from common import get_arg_parser, load_cohort

RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "results_v2"))


def main():
    parser = get_arg_parser(__doc__)
    parser.add_argument("--alpha", type=float, default=0.001,
                         help="Primary chi-square significance level (default 0.001, fixed "
                              "independently of any dataset's results).")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    X, _ = load_cohort(args.data_path)
    n, p = X.shape
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    cov_raw = np.cov(Xs, rowvar=False)
    eigvals_raw = np.linalg.eigvalsh(cov_raw)
    cond_number = eigvals_raw.max() / max(eigvals_raw.min(), 1e-12)

    pca_full = PCA(n_components=min(n - 1, p))
    pca_full.fit(Xs)
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = min(int(np.searchsorted(cum_var, 0.90) + 1), n - 1)

    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(Xs)
    eigvals = pca.explained_variance_
    mahal_sq = np.sum((scores ** 2) / eigvals, axis=1)

    cutoff = chi2.ppf(1 - args.alpha, df=n_components)
    flagged = sorted(int(i) for i in np.where(mahal_sq > cutoff)[0].tolist())

    alpha_sensitivity = {}
    for a in [0.05, 0.01, 0.001]:
        c = chi2.ppf(1 - a, df=n_components)
        alpha_sensitivity[a] = sorted(int(i) for i in np.where(mahal_sq > c)[0].tolist())

    # Sensitivity checks (not primary): raw Mahalanobis (documented as unstable), and MCD
    mahal_raw = EmpiricalCovariance().fit(Xs).mahalanobis(Xs)
    flagged_raw = sorted(int(i) for i in np.where(mahal_raw > chi2.ppf(1 - args.alpha, df=p))[0].tolist())
    try:
        mcd = MinCovDet(random_state=args.random_state, support_fraction=0.95).fit(Xs)
        mahal_mcd = mcd.mahalanobis(Xs)
        flagged_mcd = sorted(int(i) for i in np.where(mahal_mcd > chi2.ppf(1 - args.alpha, df=p))[0].tolist())
    except Exception as e:
        flagged_mcd = f"failed: {e}"

    print(f"n={n}, p={p}")
    print(f"Raw covariance condition number: {cond_number:.2e}")
    print(f"PCA: {n_components} components for >=90% variance")
    print(f"Primary (alpha={args.alpha}): flagged {flagged}")
    print(f"Alpha sensitivity: {alpha_sensitivity}")
    print(f"Sensitivity check, raw Mahalanobis: {flagged_raw}")
    print(f"Sensitivity check, MCD: {flagged_mcd}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    result = {
        "data_path": args.data_path, "n": n, "p": p,
        "raw_covariance_condition_number": float(cond_number),
        "pca_n_components_90pct": n_components,
        "alpha_primary": args.alpha, "flagged_primary": flagged,
        "alpha_sensitivity": {str(k): v for k, v in alpha_sensitivity.items()},
        "flagged_raw_mahalanobis_sensitivity": flagged_raw,
        "flagged_mcd_sensitivity": flagged_mcd,
    }
    with open(os.path.join(RESULTS_DIR, "01_outlier_screening_result.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {RESULTS_DIR}/01_outlier_screening_result.json")


if __name__ == "__main__":
    main()

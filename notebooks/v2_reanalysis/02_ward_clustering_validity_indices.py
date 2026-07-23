"""
v2 reanalysis - Ward hierarchical clustering evaluated across four internal validity indices.

Corresponds to Section 3.3 (Structure-Discovery Diagnostics) of the revised manuscript.
Reports Silhouette, Calinski-Harabasz, Davies-Bouldin, and the Gap Statistic (Tibshirani,
Walther & Hastie, 2001) side by side for k=3 and k=5, extending the original submission's
single-metric (Silhouette-only) justification for cluster count.

By default this runs against the public synthetic demo dataset (see common.py) and will NOT
reproduce the manuscript's numbers -- only the real, non-public cohort does that (Section 7,
Data Availability Statement).

Usage:
    python 02_ward_clustering_validity_indices.py [--data-path PATH] [--n-ref-gap 50]
"""
import json
import os
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

from common import get_arg_parser, load_cohort

RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "results_v2"))


def within_cluster_dispersion(X_scaled, labels, k):
    Wk = 0.0
    for c in range(k):
        pts = X_scaled[labels == c]
        if len(pts) <= 1:
            continue
        d = np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=-1)
        Wk += d.sum() / (2.0 * len(pts))
    return Wk


def gap_statistic(X_scaled, k, rng, n_ref):
    model = linkage(X_scaled, method="ward")
    labels = fcluster(model, t=k, criterion="maxclust") - 1
    log_Wk_real = np.log(within_cluster_dispersion(X_scaled, labels, k))

    pca = PCA()
    scores = pca.fit_transform(X_scaled)
    mins, maxs = scores.min(axis=0), scores.max(axis=0)

    log_Wk_refs = []
    for _ in range(n_ref):
        ref_scores = rng.uniform(mins, maxs, size=scores.shape)
        ref_data = pca.inverse_transform(ref_scores)
        ref_labels = fcluster(linkage(ref_data, method="ward"), t=k, criterion="maxclust") - 1
        Wk_ref = within_cluster_dispersion(ref_data, ref_labels, k)
        log_Wk_refs.append(np.log(Wk_ref) if Wk_ref > 0 else 0.0)

    log_Wk_refs = np.array(log_Wk_refs)
    gap = log_Wk_refs.mean() - log_Wk_real
    sk = log_Wk_refs.std(ddof=1) * np.sqrt(1 + 1.0 / n_ref)
    return gap, sk


def main():
    parser = get_arg_parser(__doc__)
    parser.add_argument("--n-ref-gap", type=int, default=50,
                         help="Number of uniform reference datasets for the Gap Statistic.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--k-values", type=int, nargs="+", default=[3, 5])
    args = parser.parse_args()

    X, _ = load_cohort(args.data_path)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    rng = np.random.default_rng(args.random_state)

    results = {}
    for k in args.k_values:
        model = linkage(Xs, method="ward")
        labels = fcluster(model, t=k, criterion="maxclust") - 1

        sil = silhouette_score(Xs, labels)
        ch = calinski_harabasz_score(Xs, labels)
        db = davies_bouldin_score(Xs, labels)
        gap, gap_se = gap_statistic(Xs, k, rng, args.n_ref_gap)
        sizes = pd.Series(labels).value_counts().sort_index().to_dict()

        print(f"\n--- k={k} ---")
        print(f"Silhouette: {sil:.4f} | Calinski-Harabasz: {ch:.4f} | "
              f"Davies-Bouldin: {db:.4f} | Gap: {gap:.4f} (SE={gap_se:.4f})")
        print(f"Cluster sizes: {sizes}")

        results[str(k)] = {
            "silhouette": float(sil), "calinski_harabasz": float(ch),
            "davies_bouldin": float(db), "gap_statistic": float(gap), "gap_se": float(gap_se),
            "cluster_sizes": {str(kk): int(vv) for kk, vv in sizes.items()},
        }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "02_ward_clustering_validity_result.json"), "w") as f:
        json.dump({"data_path": args.data_path, "n": len(X), "results": results}, f, indent=2)
    print(f"\nSaved: {RESULTS_DIR}/02_ward_clustering_validity_result.json")


if __name__ == "__main__":
    main()

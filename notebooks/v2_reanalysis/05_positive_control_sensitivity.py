"""
v2 reanalysis - Positive-control sensitivity check for the structure-discovery validity check.

Corresponds to Section 3.4 (Structure-Discovery Validity Check) of the revised manuscript.
Answers a question the original validity check (02_ward_clustering_validity_indices.py,
03_bayesian_mixture_convergence.py) cannot answer on its own: given that neither method found
defensible structure in this cohort, is that because no structure exists, or because neither
method has the power to detect structure at this n even when it is present?

Method: starting from the real (or demo) cohort's own 19-feature matrix, we plant a known
k=3 group structure by shifting a subset of features for 2 of 3 (deterministically, then
randomly re-labelled) equal-sized groups by a fixed number of that feature's own real-data
standard deviations (PRE-shift SD; because the shift itself raises that column's own variance,
the resulting separation in the re-standardized space the pipeline actually clusters on is
substantially smaller than this nominal multiple -- e.g. a nominal 10-SD shift corresponds to
roughly 1-2 SD post-standardization; see effective_separation_sd in the saved JSON). We then
re-run Ward clustering, reporting the same three magnitude-based indices as script 02
(Silhouette, Calinski-Harabasz, Davies-Bouldin; the Gap Statistic is NOT recomputed here), and
the same Bayesian nonparametric mixture model as script 03, but at a single fixed component
ceiling (not swept across 8/12/15/20 as script 03 does) chosen to match script 03's baseline
ceiling for direct comparability. This is a deliberately narrower re-application of scripts 02
and 03, not an unchanged, full re-run of either.

Two scenarios are tested:
- SPARSE: structure confined to 6 of 19 features (CK_Pre/Pos/24h, Cortisol_Pre/Pos/24h -- two
  biologically distinct domains, roughly a third of the panel), at several separation
  magnitudes. This represents the realistic case: if distinct physiological states exist, they
  plausibly differ on a subset of markers, not uniformly across the entire panel.
- FULL: structure spread across all 19 features, at one separation magnitude, as a positive
  control that the pipeline CAN detect strong, panel-wide structure when present.

This is a sensitivity/power check on the METHOD, not a new claim about this cohort's true
biology: the planted groups are synthetic re-labellings of the real (or demo) cohort's own
rows, not new subjects. Each (scenario, magnitude) combination is repeated across multiple
random group-label assignments to avoid depending on one arbitrary grouping.

By default this runs against the public synthetic demo dataset (see common.py) and will NOT
reproduce the manuscript's numbers -- only the real, non-public cohort does that (Section 7,
Data Availability Statement).

Usage:
    python 05_positive_control_sensitivity.py [--data-path PATH] [--n-reps 20]
"""
import json
import os
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.mixture import BayesianGaussianMixture

from common import get_arg_parser, load_cohort

RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "results_v2"))

SPARSE_FEATURES = ["CK_Pre", "CK_Pos", "CK_24h", "Cortisol_Pre", "Cortisol_Pos", "Cortisol_24h"]


def plant_clusters(X, cols, shift_sd, rng, k=3):
    """Shift the given columns by shift_sd * (that column's own real-data SD) for 2 of 3
    equal-ish groups, on top of the real data. Returns the modified copy and the true labels."""
    Xc = X.copy()
    Xc[cols] = Xc[cols].astype(float)  # avoid silent int-truncation of the shift (e.g. BP columns)
    n = len(Xc)
    labels = np.array([i % k for i in range(n)])
    rng.shuffle(labels)
    for col in cols:
        sd = X[col].std()
        for g in range(1, k):
            Xc.loc[labels == g, col] = Xc.loc[labels == g, col] + g * shift_sd * sd
    return Xc, labels


def effective_separation_sd(Xc, cols, labels):
    """Mean, across the planted columns, of the post-standardization separation between group 0
    and group 1's means (in units of that column's OWN standard deviation after planting, i.e.
    what the pipeline actually clusters on) -- as opposed to the nominal, pre-shift shift_sd
    parameter, which overstates the real separation once the shift has inflated the column's own
    variance."""
    seps = []
    for col in cols:
        sd_post = Xc[col].std()
        if sd_post > 0:
            seps.append(abs(Xc.loc[labels == 1, col].mean() - Xc.loc[labels == 0, col].mean()) / sd_post)
    return float(np.mean(seps))


def ward_indices(Xc, k=3):
    Xs = StandardScaler().fit_transform(Xc)
    model = linkage(Xs, method="ward")
    pred = fcluster(model, t=k, criterion="maxclust") - 1
    return (silhouette_score(Xs, pred), calinski_harabasz_score(Xs, pred),
            davies_bouldin_score(Xs, pred))


def bgm_effective_k(Xc, ceiling, n_seeds=10):
    Xs = StandardScaler().fit_transform(Xc)
    weight_threshold = 1.0 / (2 * len(Xs))
    ks = []
    for seed in range(n_seeds):
        m = BayesianGaussianMixture(n_components=ceiling, covariance_type="diag",
                                     weight_concentration_prior_type="dirichlet_process",
                                     random_state=seed, max_iter=500, n_init=3)
        m.fit(Xs)
        ks.append(int((m.weights_ > weight_threshold).sum()))
    return float(np.median(ks))


def main():
    parser = get_arg_parser(__doc__)
    parser.add_argument("--n-reps", type=int, default=20,
                         help="Random group-label reassignments averaged per (scenario, "
                              "magnitude) combination.")
    parser.add_argument("--shift-sds", type=float, nargs="+", default=[2, 4, 6, 10],
                         help="Separation magnitudes (in each feature's own SD) tested for the "
                              "SPARSE scenario.")
    parser.add_argument("--full-shift-sd", type=float, default=4,
                         help="Separation magnitude for the FULL (all-features) scenario.")
    parser.add_argument("--bgm-ceiling", type=int, default=8,
                         help="A SINGLE fixed component-count ceiling (matching script 03's "
                              "lowest/baseline ceiling), not swept across multiple ceilings as "
                              "script 03 does.")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    X, _ = load_cohort(args.data_path)
    n, p = X.shape
    print(f"n={n}, p={p}")

    rng = np.random.default_rng(args.random_state)

    print(f"\n--- Baseline: real data, no planted structure ---")
    sil0, ch0, db0 = ward_indices(X)
    k0 = bgm_effective_k(X, args.bgm_ceiling)
    print(f"Silhouette={sil0:.4f} CH={ch0:.3f} DB={db0:.3f} | BGM effective-k (median)={k0:.1f}")

    results = {"baseline": {"silhouette": sil0, "ch": ch0, "db": db0, "bgm_effective_k": k0}}

    print(f"\n--- SPARSE: structure planted on {len(SPARSE_FEATURES)}/{p} features ---")
    sparse_results = {}
    for shift in args.shift_sds:
        sils, chs, dbs, ks, effs = [], [], [], [], []
        for _ in range(args.n_reps):
            Xc, labels = plant_clusters(X, SPARSE_FEATURES, shift, rng)
            sil, ch, db = ward_indices(Xc)
            sils.append(sil); chs.append(ch); dbs.append(db)
            ks.append(bgm_effective_k(Xc, args.bgm_ceiling))
            effs.append(effective_separation_sd(Xc, SPARSE_FEATURES, labels))
        print(f"shift={shift}sd (nominal; effective post-standardization sep.={np.mean(effs):.2f}sd): "
              f"Silhouette={np.mean(sils):.4f} (SD {np.std(sils):.4f}) "
              f"CH={np.mean(chs):.3f} DB={np.mean(dbs):.3f} | "
              f"BGM effective-k (mean of medians)={np.mean(ks):.2f}")
        sparse_results[str(shift)] = {
            "silhouette_mean": float(np.mean(sils)), "silhouette_sd": float(np.std(sils)),
            "ch_mean": float(np.mean(chs)), "db_mean": float(np.mean(dbs)),
            "bgm_effective_k_mean": float(np.mean(ks)),
            "effective_separation_sd": float(np.mean(effs)),
        }
    results["sparse"] = sparse_results

    print(f"\n--- FULL: structure planted on all {p} features, shift={args.full_shift_sd}sd ---")
    sils, chs, dbs, ks, effs = [], [], [], [], []
    for _ in range(args.n_reps):
        Xc, labels = plant_clusters(X, list(X.columns), args.full_shift_sd, rng)
        sil, ch, db = ward_indices(Xc)
        sils.append(sil); chs.append(ch); dbs.append(db)
        ks.append(bgm_effective_k(Xc, args.bgm_ceiling))
        effs.append(effective_separation_sd(Xc, list(X.columns), labels))
    print(f"nominal shift={args.full_shift_sd}sd (effective post-standardization sep.={np.mean(effs):.2f}sd): "
          f"Silhouette={np.mean(sils):.4f} (SD {np.std(sils):.4f}) CH={np.mean(chs):.3f} "
          f"DB={np.mean(dbs):.3f} | BGM effective-k (mean of medians)={np.mean(ks):.2f}")
    results["full"] = {
        "silhouette_mean": float(np.mean(sils)), "silhouette_sd": float(np.std(sils)),
        "ch_mean": float(np.mean(chs)), "db_mean": float(np.mean(dbs)),
        "bgm_effective_k_mean": float(np.mean(ks)),
        "effective_separation_sd": float(np.mean(effs)),
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "05_positive_control_result.json"), "w") as f:
        json.dump({"data_path": args.data_path, "n": n, "p": p, **results}, f, indent=2)
    print(f"\nSaved: {RESULTS_DIR}/05_positive_control_result.json")


if __name__ == "__main__":
    main()

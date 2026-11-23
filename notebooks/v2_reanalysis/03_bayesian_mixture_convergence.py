"""
v2 reanalysis - Bayesian Gaussian Mixture (Dirichlet process prior) with a component-ceiling
convergence test.

Corresponds to Section 3.3 (Structure-Discovery Diagnostics) of the revised manuscript, as the
second, methodologically independent structure-discovery approach alongside Ward clustering
(see 02_ward_clustering_validity_indices.py).

Method: sklearn BayesianGaussianMixture, weight_concentration_prior_type='dirichlet_process',
covariance_type='diag' (a 'full' covariance per component is unidentifiable at this sample
size). No literature-informed numeric prior is injected into the mean/covariance structure --
no citable multivariate prior for this biomarker panel was found; sklearn's default weakly
informative priors are used instead (documented, not invented).

Convergence check: the model is fit across multiple n_components ceilings, and the number of
"effective" components (weight above a data-driven threshold) is tracked. If the effective
count keeps rising with the ceiling rather than plateauing, that is direct evidence the data do
not support a stable number of latent groups at this sample size -- reported as-is, not
adjusted to force a stable-looking answer.

By default this runs against the public synthetic demo dataset (see common.py) and will NOT
reproduce the manuscript's numbers -- only the real, non-public cohort does that (Section 7,
Data Availability Statement). PyMC (full MCMC) was not used -- not installed in the original
analysis environment, and installing it was judged a build-risk not worth taking; this is a
documented limitation, not a silent shortcut (see README.md and manuscript Section 3.3).

Usage:
    python 03_bayesian_mixture_convergence.py [--data-path PATH] [--ceilings 8 12 15 20]
"""
import json
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import BayesianGaussianMixture

from common import get_arg_parser, load_cohort

RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "results_v2"))


def fit_bgm(data, n_components, seed):
    m = BayesianGaussianMixture(
        n_components=n_components, covariance_type="diag",
        weight_concentration_prior_type="dirichlet_process",
        random_state=seed, max_iter=500, n_init=3,
    )
    m.fit(data)
    return m


def main():
    parser = get_arg_parser(__doc__)
    parser.add_argument("--ceilings", type=int, nargs="+", default=[8, 12, 15, 20],
                         help="Component-count ceilings to test for convergence.")
    parser.add_argument("--n-seeds-per-ceiling", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    X, _ = load_cohort(args.data_path)
    n = len(X)
    Xs = StandardScaler().fit_transform(X)
    weight_threshold = 1.0 / (2 * n)

    convergence = {}
    for ub in args.ceilings:
        ks = []
        for seed in range(args.n_seeds_per_ceiling):
            m = fit_bgm(Xs, ub, seed)
            ks.append(int((m.weights_ > weight_threshold).sum()))
        convergence[ub] = ks
        print(f"ceiling={ub}: effective-k across {args.n_seeds_per_ceiling} seeds -> "
              f"{sorted(ks)}, median={np.median(ks)}")

    is_converging = len(set(int(np.median(v)) for v in convergence.values())) == 1
    print(f"\nConverged to a stable effective-k across ceilings: {is_converging}")
    if not is_converging:
        print("Effective-k scales with the imposed ceiling rather than converging -- "
              "interpreted as: this sample size does not support a stable number of latent "
              "components with this method (see manuscript Section 3.3/6).")

    primary = fit_bgm(Xs, args.ceilings[0], args.random_state)
    proba = primary.predict_proba(Xs)
    max_proba = proba.max(axis=1)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    result = {
        "data_path": args.data_path, "n": n, "weight_threshold": weight_threshold,
        "convergence_by_ceiling": {str(k): v for k, v in convergence.items()},
        "converged": is_converging,
        "primary_ceiling": args.ceilings[0],
        "primary_effective_k": int((primary.weights_ > weight_threshold).sum()),
        "primary_weights": [round(float(w), 4) for w in primary.weights_],
        "posterior_confidence_mean_max_proba": float(max_proba.mean()),
    }
    with open(os.path.join(RESULTS_DIR, "03_bayesian_mixture_result.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {RESULTS_DIR}/03_bayesian_mixture_result.json")


if __name__ == "__main__":
    main()

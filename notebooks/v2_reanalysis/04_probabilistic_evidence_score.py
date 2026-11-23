"""
v2 reanalysis - Literature-informed evidence-combination score (per-biomarker evidence
functions + naive-Bayes-style independent combination).

Corresponds to Section 2.4 (Literature-Informed Evidence Combination) of the revised
manuscript. Replaces the original submission's cascade-of-fixed-thresholds classification with
a continuous score that combines evidence from multiple biomarkers simultaneously per subject.

CRITICAL design rule, checked by construction: every location/scale parameter used to convert a
subject's raw value into a z-score comes from a PUBLISHED external reference (with its own
sample size), never from this script's own input data. The input data (X) is only ever used to
compute a z given an externally-sourced mu/sigma -- never to fit mu/sigma itself. This avoids
leaking the evaluated sample's own distribution into the criteria used to evaluate it.

Evidence sources (see manuscript Section 2.4 for full detail and uncertainty propagation):
- CK: Mougios V. Br J Sports Med. 2007;41(10):674-8. PMID 17526622, "Dependence on sport"
  subsection. Sport-specific reference interval for male football/soccer players (this cohort's
  sport), n=182: lower limit 83 U/L (95% CI 53-84), upper limit 1492 U/L (95% CI 924-1908),
  calculated by the same non-parametric 2.5-97.5th percentile method as the paper's pooled
  male-athlete interval. This resolves a prior discrepancy in this script, which previously used
  the pooled male-athlete-generally interval [82, 1083] U/L (n=483, all sports combined) because
  the sport-specific upper limit (1492) had been cited without its paired lower bound.
    A 2018 alternative ("Is it necessary to adjust current creatine kinase reference ranges to
  reflect levels found in professional footballers?", PMID 29387443 -- author names not verified
  by this script's author, cited by PMID only) proposes a wider interval [64.9, 1971.7] U/L
  specifically for elite/professional footballers.
  It was NOT used here: n=27 (self-acknowledged by its authors as a limitation, vs Mougios'
  n=182), a parametric log-normal fit rather than the non-parametric percentile method used
  elsewhere in this script, an unresolved ethnic-composition confound between the study group and
  its control group (also acknowledged by its authors), and a population of elite/professional
  players training at a load well above this cohort's amateur 2-3 sessions/week -- all reasons to
  prefer the larger, methodologically consistent, better population-matched Mougios interval
  despite its 2007 publication date. Reference intervals for enzyme activity are not expected to
  go stale the way clinical guidelines do; the deciding factors are sample size, methodological
  rigor, and population match, not recency alone.
- LDH: NOT included in the composite. No population-matched reference interval for LDH in
  athletes is established in the literature (Diaz Martinez et al. 2022, IJERPH 19(5):3059,
  PMID 35270750). LDH is reported descriptively only (see ldh_descriptive below), never
  converted to a z-score and never classified against any reference interval.
- CRP: Perez-Castillo IM et al. Front Physiol. 2023;14:1167449 (review synthesizing Ispirlidis
  2008, Silva 2013, Souglis 2015b, Mohr 2016, Romagnoli 2016, Fransson 2018, Daab 2021, Duarte
  2022): CRP peaks ~24h post-match at 1.5-3x baseline (fold-change from each subject's own
  Pre-match value).
- The acute testosterone/cortisol ratio is intentionally EXCLUDED as risk evidence: an acute
  Pre->24h decline was established (Section 2.3/results Section 3.4) to be an expected recovery
  response, not a risk signal, absent new justification.

Assay-floor check: if a large fraction of PCR_Pre values sit exactly at the sample minimum
(consistent with an assay's limit of quantification), the CRP fold-change term is numerically
unstable for those subjects (dividing by a censored near-floor value). This is detected
dynamically (not hardcoded to any one dataset's specific floor value) and, when triggered, two
composites are reported side by side: the full composite (CK+CRP) and a CK-only composite
recommended for interpretation.

Combination: given the absence of a statistically defensible covariance structure in this
cohort (Section 2.3), CK and CRP evidence is combined under an explicit assumption of
conditional independence when both are usable. No citable numeric CK:CRP weighting ratio was
found in the literature; per this project's standing rule ("no citation -> non-informative
default, not an invented plausible number"), this script therefore uses EQUAL weights (1:1) by
default -- a --ck-weight / --crp-weight pair is exposed below so a specific ratio can be applied
once one is sourced.
[PENDING CITATION: precise numeric CK:CRP importance ratio, if one exists in the literature.]

By default this runs against the public synthetic demo dataset (see common.py) and will NOT
reproduce the manuscript's numbers -- only the real, non-public cohort does that (Section 7,
Data Availability Statement).

Usage:
    python 04_probabilistic_evidence_score.py [--data-path PATH] [--n-bootstrap 3000]
"""
import json
import os
import numpy as np
import pandas as pd
from scipy.stats import norm

from common import get_arg_parser, load_cohort, _resolve, _to_float

RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "results_v2"))

REFERENCES = {
    "CK": {"n_reference": 182, "log_mu": float(np.log(np.sqrt(83 * 1492))),
           "log_sd": float((np.log(1492) - np.log(83)) / (2 * 1.96))},
    "CRP_fold": {"n_reference": 8, "log_mu": float(np.log(np.sqrt(1.5 * 3.0))),
                 "log_sd": float((np.log(3.0) - np.log(1.5)) / (2 * 1.96))},
}
# Relative importance weights: NO citable numeric ratio found (see module docstring). Equal
# weights (1.0 each) are the honest default; override via --ck-weight/--crp-weight if a
# specific literature-sourced ratio becomes available.
DEFAULT_WEIGHTS = {"CK": 1.0, "CRP_fold": 1.0}


def one_sided_z(value, log_mu, log_sd):
    return np.maximum((np.log(value) - log_mu) / log_sd, 0.0)


def weighted_composite(Z, weights):
    w = np.array([weights[c] for c in Z.columns])
    return (Z.values * w).sum(axis=1) / w.sum()


def main():
    parser = get_arg_parser(__doc__)
    parser.add_argument("--n-bootstrap", type=int, default=3000,
                         help="Bootstrap draws over REFERENCE-source uncertainty (not over the "
                              "evaluated sample).")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--floor-fraction-threshold", type=float, default=0.3,
                         help="If this fraction (or more) of CRP_Pre values equal the sample "
                              "minimum, the CRP term is flagged as an assay-floor artifact.")
    parser.add_argument("--ck-weight", type=float, default=DEFAULT_WEIGHTS["CK"],
                         help="CK relative weight (default 1.0 = equal weighting; no citable "
                              "numeric ratio was found -- see module docstring).")
    parser.add_argument("--crp-weight", type=float, default=DEFAULT_WEIGHTS["CRP_fold"])
    args = parser.parse_args()
    weights_cfg = {"CK": args.ck_weight, "CRP_fold": args.crp_weight}

    ATTENTION_CUTOFF = norm.ppf(0.95)  # pre-specified, one-sided 95th percentile equivalent

    X, raw = load_cohort(args.data_path)
    n = len(X)

    ck_val = X[["CK_Pre", "CK_Pos", "CK_24h"]].max(axis=1)
    # LDH is NEVER converted to a z-score or classified against any reference interval (see
    # module docstring for why). Reported descriptively only.
    ldh_val = X[["LDH_Pre", "LDH_Pos", "LDH_24h"]].max(axis=1)
    ldh_descriptive = {
        "mean": float(ldh_val.mean()), "median": float(ldh_val.median()),
        "sd": float(ldh_val.std()), "min": float(ldh_val.min()), "max": float(ldh_val.max()),
    }

    # Floor-fraction is computed from the RAW, pre-imputation CRP_Pre column, not from X (which
    # has already been median-imputed by load_cohort). Using X here would silently count
    # genuinely-missing subjects as "at floor" whenever the sample median happens to equal the
    # floor value (as it does in this cohort) -- conflating missingness with assay censoring.
    crp_pre_col = _resolve(raw, "CRP_Pre")
    crp_pre_raw = _to_float(raw[crp_pre_col]) if raw[crp_pre_col].dtype == object else raw[crp_pre_col]
    n_missing_crp_pre = int(crp_pre_raw.isna().sum())
    crp_pre_true_min = crp_pre_raw.min()
    n_at_floor = int((crp_pre_raw == crp_pre_true_min).sum())
    frac_at_floor = n_at_floor / n
    crp_floor_flag = frac_at_floor >= args.floor_fraction_threshold
    # crp_val below still uses the median-imputed X for the 4 missing-data subjects (giving them
    # a value equal to the floor) -- acceptable ONLY because this FULL composite is documented as
    # not used for interpretation (see module docstring); the CK-only composite is unaffected.
    crp_val = X["CRP_24h"] / X["CRP_Pre"].clip(lower=max(crp_pre_true_min, 1e-3))

    print(f"n={n}")
    print(f"LDH (descriptive only, not classified): mean={ldh_descriptive['mean']:.1f} "
          f"median={ldh_descriptive['median']:.1f} SD={ldh_descriptive['sd']:.1f} "
          f"range=[{ldh_descriptive['min']:.1f}, {ldh_descriptive['max']:.1f}]")
    print(f"CRP_Pre: {n_at_floor}/{n} at assay floor ({crp_pre_true_min}), "
          f"{n_missing_crp_pre}/{n} missing (pre-imputation) -> "
          f"floor fraction {frac_at_floor*100:.1f}% flagged unstable: {crp_floor_flag}")

    Z = pd.DataFrame({
        "CK": one_sided_z(ck_val, REFERENCES["CK"]["log_mu"], REFERENCES["CK"]["log_sd"]),
        "CRP_fold": one_sided_z(crp_val, REFERENCES["CRP_fold"]["log_mu"], REFERENCES["CRP_fold"]["log_sd"]),
    })

    weights_full = {c: weights_cfg[c] for c in Z.columns}

    composite_full = weighted_composite(Z, weights_full)
    composite_ck_only = Z["CK"]
    composite_primary = composite_ck_only if crp_floor_flag else composite_full

    n_above_full = int((composite_full > ATTENTION_CUTOFF).sum())
    n_above_ck_only = int((composite_ck_only > ATTENTION_CUTOFF).sum())
    print(f"\nComposite FULL (CK+CRP): mean={composite_full.mean():.3f}, "
          f"above cutoff={n_above_full}/{n}")
    print(f"Composite CK only: mean={composite_ck_only.mean():.3f}, "
          f"above cutoff={n_above_ck_only}/{n}")
    print(f"Recommended primary composite: {'CK only' if crp_floor_flag else 'FULL'}")

    rng = np.random.default_rng(args.random_state)
    boot_n_above = []
    for _ in range(args.n_bootstrap):
        z_b = {}
        for name, ref in REFERENCES.items():
            se_mu = ref["log_sd"] / np.sqrt(ref["n_reference"])
            se_sd = ref["log_sd"] / np.sqrt(2 * max(ref["n_reference"] - 1, 1))
            mu_b = rng.normal(ref["log_mu"], se_mu)
            sd_b = max(rng.normal(ref["log_sd"], se_sd), 1e-3)
            val = {"CK": ck_val, "CRP_fold": crp_val}[name]
            z_b[name] = one_sided_z(val, mu_b, sd_b)
        Zb = pd.DataFrame(z_b)
        comp_b = Zb["CK"] if crp_floor_flag else weighted_composite(Zb, weights_full)
        boot_n_above.append(int((comp_b > ATTENTION_CUTOFF).sum()))

    boot_n_above = np.array(boot_n_above)
    ci_lo, ci_hi = np.percentile(boot_n_above, [2.5, 97.5])
    print(f"\nBootstrap over reference-source uncertainty ({args.n_bootstrap} draws): "
          f"median n above cutoff={np.median(boot_n_above):.0f}, 95% CI=[{ci_lo:.0f}, {ci_hi:.0f}]")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    result = {
        "data_path": args.data_path, "n": n,
        "attention_cutoff_z": float(ATTENTION_CUTOFF),
        "ldh_descriptive_not_classified": ldh_descriptive,
        "crp_floor_check": {"fraction_at_floor": float(frac_at_floor), "flagged": bool(crp_floor_flag)},
        "composite_full": {"mean": float(composite_full.mean()), "n_above_cutoff": n_above_full},
        "composite_ck_only": {"mean": float(composite_ck_only.mean()), "n_above_cutoff": n_above_ck_only},
        "recommended_primary": "ck_only" if crp_floor_flag else "full",
        "bootstrap_n_above_cutoff": {
            "median": float(np.median(boot_n_above)), "ci95": [float(ci_lo), float(ci_hi)],
        },
    }
    with open(os.path.join(RESULTS_DIR, "04_probabilistic_evidence_score_result.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {RESULTS_DIR}/04_probabilistic_evidence_score_result.json")


if __name__ == "__main__":
    main()

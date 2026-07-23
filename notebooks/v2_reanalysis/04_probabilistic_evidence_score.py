"""
v2 reanalysis - Literature-informed evidence-combination score (per-biomarker evidence
functions + naive-Bayes-style independent combination).

Corresponds to Section 3.4 (Literature-Informed Evidence Combination) of the revised
manuscript. Replaces the original submission's cascade-of-fixed-thresholds classification with
a continuous score that combines evidence from multiple biomarkers simultaneously per subject.

CRITICAL design rule, checked by construction: every location/scale parameter used to convert a
subject's raw value into a z-score comes from a PUBLISHED external reference (with its own
sample size), never from this script's own input data. The input data (X) is only ever used to
compute a z given an externally-sourced mu/sigma -- never to fit mu/sigma itself. This avoids
leaking the evaluated sample's own distribution into the criteria used to evaluate it.

Evidence sources (see manuscript Section 3.4 for full detail and uncertainty propagation):
- CK: Mougios V. Br J Sports Med. 2007;41(10):674-8. PMID 17526622. Male athletes generally,
  n=483, reference interval [82, 1083] U/L -- used here as the log-normal distributional basis
  (a complete 2.5-97.5 percentile interval). NOTE: the same paper also reports a sport-specific
  upper limit of 1492 U/L for male football/soccer players specifically (this cohort's sport),
  which the manuscript draft (Section 3.4) cites instead of the pooled 1083 U/L figure. No
  matching sport-specific LOWER bound was available to this script's author to pair with 1492,
  so a full log-normal distribution could not be reconstructed from it alone -- flagged as an
  unresolved discrepancy between this script and the manuscript text, not silently picked
  either way. [PENDING RECONCILIATION: sport-specific CK reference interval, full 2.5-97.5%
  range for male soccer players, if available from Mougios 2007's supplementary data.]
- LDH: Srividhya, Majumdar & Subramanian (2017). "Reference Interval for Enzyme Lactate
  Dehydrogenase in Male and Female Athletes", International Journal of Science and Research
  (IJSR), Vol 6(1). Male athletes, n=320, reference interval [138.4, 746.0] IU/L.
- CRP: Perez-Castillo IM et al. Front Physiol. 2023;14:1167449 (review synthesizing Ispirlidis
  2008, Silva 2013, Souglis 2015b, Mohr 2016, Romagnoli 2016, Fransson 2018, Daab 2021, Duarte
  2022): CRP peaks ~24h post-match at 1.5-3x baseline (fold-change from each subject's own
  Pre-match value).
- The acute testosterone/cortisol ratio is intentionally EXCLUDED as risk evidence: an acute
  Pre->24h decline was established (Section 3.3/3.4) to be an expected recovery response, not
  a risk signal, absent new justification.

Assay-floor check: if a large fraction of PCR_Pre values sit exactly at the sample minimum
(consistent with an assay's limit of quantification), the CRP fold-change term is numerically
unstable for those subjects (dividing by a censored near-floor value). This is detected
dynamically (not hardcoded to any one dataset's specific floor value) and, when triggered, two
composites are reported side by side: the full composite (CK+LDH+CRP) and a CK+LDH-only
composite recommended for interpretation.

Combination: given the absence of a statistically defensible covariance structure in this
cohort (Section 3.3), CK and LDH evidence is combined under an explicit assumption of
conditional independence. The manuscript draft (Section 3.4) argues CK should be weighted more
heavily than LDH on qualitative grounds (CK's greater specificity for skeletal-muscle damage --
LDH is also elevated in cardiac/hepatic/hematologic processes), but explicitly states no single
citable NUMERIC weighting ratio was found in the literature. Per this project's standing rule
("no citation -> non-informative default, not an invented plausible number"), this script
therefore uses EQUAL weights (1:1) by default, not an invented ratio -- a --ck-weight /
--ldh-weight pair is exposed below so a specific ratio can be applied once one is sourced.
[PENDING CITATION: precise numeric CK:LDH importance ratio, if one exists in the literature.]

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

from common import get_arg_parser, load_cohort

RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "results_v2"))

REFERENCES = {
    "CK": {"n_reference": 483, "log_mu": float(np.log(np.sqrt(82 * 1083))),
           "log_sd": float((np.log(1083) - np.log(82)) / (2 * 1.96))},
    "LDH": {"n_reference": 320, "log_mu": float(np.log(np.sqrt(138.4 * 746.0))),
            "log_sd": float((np.log(746.0) - np.log(138.4)) / (2 * 1.96))},
    "CRP_fold": {"n_reference": 8, "log_mu": float(np.log(np.sqrt(1.5 * 3.0))),
                 "log_sd": float((np.log(3.0) - np.log(1.5)) / (2 * 1.96))},
}
# Relative importance weights: NO citable numeric ratio found (see module docstring). Equal
# weights (1.0 each) are the honest default; override via --ck-weight/--ldh-weight if a
# specific literature-sourced ratio becomes available.
DEFAULT_WEIGHTS = {"CK": 1.0, "LDH": 1.0, "CRP_fold": 1.0}


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
    parser.add_argument("--ldh-weight", type=float, default=DEFAULT_WEIGHTS["LDH"])
    parser.add_argument("--crp-weight", type=float, default=DEFAULT_WEIGHTS["CRP_fold"])
    args = parser.parse_args()
    weights_cfg = {"CK": args.ck_weight, "LDH": args.ldh_weight, "CRP_fold": args.crp_weight}

    ATTENTION_CUTOFF = norm.ppf(0.95)  # pre-specified, one-sided 95th percentile equivalent

    X, _ = load_cohort(args.data_path)
    n = len(X)

    ck_val = X[["CK_Pre", "CK_Pos", "CK_24h"]].max(axis=1)
    ldh_val = X[["LDH_Pre", "LDH_Pos", "LDH_24h"]].max(axis=1)

    crp_pre_min = X["CRP_Pre"].min()
    frac_at_floor = (X["CRP_Pre"] == crp_pre_min).mean()
    crp_floor_flag = frac_at_floor >= args.floor_fraction_threshold
    crp_val = X["CRP_24h"] / X["CRP_Pre"].clip(lower=max(crp_pre_min, 1e-3))

    print(f"n={n}")
    print(f"CRP_Pre assay-floor check: {frac_at_floor*100:.1f}% of subjects at the sample "
          f"minimum ({crp_pre_min}) -> flagged unstable: {crp_floor_flag}")

    Z = pd.DataFrame({
        "CK": one_sided_z(ck_val, REFERENCES["CK"]["log_mu"], REFERENCES["CK"]["log_sd"]),
        "LDH": one_sided_z(ldh_val, REFERENCES["LDH"]["log_mu"], REFERENCES["LDH"]["log_sd"]),
        "CRP_fold": one_sided_z(crp_val, REFERENCES["CRP_fold"]["log_mu"], REFERENCES["CRP_fold"]["log_sd"]),
    })

    weights_full = {c: weights_cfg[c] for c in Z.columns}
    weights_ckldh = {"CK": weights_cfg["CK"], "LDH": weights_cfg["LDH"]}

    composite_full = weighted_composite(Z, weights_full)
    composite_ckldh = weighted_composite(Z[["CK", "LDH"]], weights_ckldh)
    composite_primary = composite_ckldh if crp_floor_flag else composite_full

    n_above_full = int((composite_full > ATTENTION_CUTOFF).sum())
    n_above_ckldh = int((composite_ckldh > ATTENTION_CUTOFF).sum())
    print(f"\nComposite FULL (CK+LDH+CRP): mean={composite_full.mean():.3f}, "
          f"above cutoff={n_above_full}/{n}")
    print(f"Composite CK+LDH only: mean={composite_ckldh.mean():.3f}, "
          f"above cutoff={n_above_ckldh}/{n}")
    print(f"Recommended primary composite: {'CK+LDH only' if crp_floor_flag else 'FULL'}")

    rng = np.random.default_rng(args.random_state)
    boot_n_above = []
    for _ in range(args.n_bootstrap):
        z_b = {}
        for name, ref in REFERENCES.items():
            se_mu = ref["log_sd"] / np.sqrt(ref["n_reference"])
            se_sd = ref["log_sd"] / np.sqrt(2 * max(ref["n_reference"] - 1, 1))
            mu_b = rng.normal(ref["log_mu"], se_mu)
            sd_b = max(rng.normal(ref["log_sd"], se_sd), 1e-3)
            val = {"CK": ck_val, "LDH": ldh_val, "CRP_fold": crp_val}[name]
            z_b[name] = one_sided_z(val, mu_b, sd_b)
        Zb = pd.DataFrame(z_b)
        comp_b = weighted_composite(Zb[list(weights_ckldh)], weights_ckldh) if crp_floor_flag \
            else weighted_composite(Zb, weights_full)
        boot_n_above.append(int((comp_b > ATTENTION_CUTOFF).sum()))

    boot_n_above = np.array(boot_n_above)
    ci_lo, ci_hi = np.percentile(boot_n_above, [2.5, 97.5])
    print(f"\nBootstrap over reference-source uncertainty ({args.n_bootstrap} draws): "
          f"median n above cutoff={np.median(boot_n_above):.0f}, 95% CI=[{ci_lo:.0f}, {ci_hi:.0f}]")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    result = {
        "data_path": args.data_path, "n": n,
        "attention_cutoff_z": float(ATTENTION_CUTOFF),
        "crp_floor_check": {"fraction_at_floor": float(frac_at_floor), "flagged": bool(crp_floor_flag)},
        "composite_full": {"mean": float(composite_full.mean()), "n_above_cutoff": n_above_full},
        "composite_ck_ldh_only": {"mean": float(composite_ckldh.mean()), "n_above_cutoff": n_above_ckldh},
        "recommended_primary": "ck_ldh_only" if crp_floor_flag else "full",
        "bootstrap_n_above_cutoff": {
            "median": float(np.median(boot_n_above)), "ci95": [float(ci_lo), float(ci_hi)],
        },
    }
    with open(os.path.join(RESULTS_DIR, "04_probabilistic_evidence_score_result.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {RESULTS_DIR}/04_probabilistic_evidence_score_result.json")


if __name__ == "__main__":
    main()

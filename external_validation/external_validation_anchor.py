"""
External validation on an independent second cohort.

Applies the exact same scoring logic as notebooks/v2_reanalysis/04_probabilistic_evidence_score.py
(same REFERENCES, same one_sided_z, same weighted_composite, same attention cutoff) to a second,
fully independent real-world cohort:

"Acute Capillary Plasma Biomarker, Neuromuscular, and Perceptual Responses to Standardised
Soccer Match Play in Elite Players: A Descriptive Study of Asynchronous Multi-Domain Recovery"
(n=22 elite male outfield soccer players, Chinese Football Association China League),
Metabolites 16(6):370 (2026), DOI 10.3390/metabo16060370, Table S1 (raw biochemical dataset,
T0-T4).

No parameter (mu, sigma, weight, cutoff) is refit on this external cohort -- all come from the
same externally-published references already used for our own n=22 cohort. This is a genuine
out-of-sample application, not a re-estimation.

LDH is reported descriptively only here too, never classified against any reference interval
(see the module docstring of 04_probabilistic_evidence_score.py: no population-matched reference
interval for LDH in athletes is established in the literature).

Primary composite is CK alone, unconditionally, matching our own cohort's script -- never
selected at runtime based on this cohort's CRP behavior. This matters here specifically: this
cohort's CRP term is unstable for a DIFFERENT reason than our own cohort's (generally low
pre-match hsCRP baselines inflating the fold-change denominator, not floor-censoring -- see
manuscript Section 3.3), which a floor-equality heuristic would not have caught, and which
previously caused this script to print a "Recommended primary composite: FULL" (22/22 above
threshold) that contradicted the manuscript's actual, pre-specified CK-only primary result. The
full composite is still computed and reported for transparency, but never as "the" result, and
its reference parameters are not bootstrapped (see 04_probabilistic_evidence_score.py's
docstring for why: CRP's "1.5-3x" summary has no real subject-level sample size behind it).

Known limitations of this external dataset, disclosed here rather than silently worked around:
- The CRP fold-change denominator floor uses this cohort's own observed minimum (hsCRP_T0 ==
  0.0 for one subject), because no externally-documented assay quantification limit for this
  specific cohort's hsCRP assay is available to us (unlike our own cohort, where the floor is a
  documented 0.05 mg/L, used as a fixed constant in 04_probabilistic_evidence_score.py). This is
  a minor exception to the project's "no parameter from the evaluated sample" rule.
- Table S1 contains at least one biologically implausible value (hsCRP = -0.1 mg/L for one
  subject at one timepoint) and at least two CK trajectories with an implausible >85% single-
  interval drop followed by a rebound (e.g. subject with CK 897 -> 92 -> 652 U/L across
  consecutive timepoints). Neither affects this script's results, since CK uses the maximum
  across timepoints and CRP's fold-change uses only T0/T3, but both are flagged here as a
  data-provenance caveat on the public external dataset, not on our own measurements.
- CK's maximum is taken across 5 timepoints here (T0-T4) versus 3 (Pre/Post/24h) in our own
  cohort. Since the one-sided z-score threshold has a fixed nominal per-measurement tail
  probability, taking a maximum over more timepoints raises the per-subject false-positive rate
  under the null relative to our own cohort -- the two cohorts are not evaluated under identical
  effective criteria, and this asymmetry is not corrected for here.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

# ---- Copied verbatim from notebooks/v2_reanalysis/04_probabilistic_evidence_score.py ----
REFERENCES = {
    "CK": {"n_reference": 182, "log_mu": float(np.log(np.sqrt(83 * 1492))),
           "log_sd": float((np.log(1492) - np.log(83)) / (2 * 1.96)),
           "lower_ci": (53.0, 84.0), "upper_ci": (924.0, 1908.0)},
    "CRP_fold": {"log_mu": float(np.log(np.sqrt(1.5 * 3.0))),
                 "log_sd": float((np.log(3.0) - np.log(1.5)) / (2 * 1.96))},
}
DEFAULT_WEIGHTS = {"CK": 1.0, "CRP_fold": 1.0}
ATTENTION_CUTOFF = norm.ppf(0.95)


def one_sided_z(value, log_mu, log_sd):
    return np.maximum((np.log(value) - log_mu) / log_sd, 0.0)


def published_ci_log_mu_sd_se(ref):
    """See 04_probabilistic_evidence_score.py's identically-named function: derives SE(log_mu),
    SE(log_sd) from the reference paper's own published 95% CIs for the lower/upper limits,
    rather than a generic sqrt(n)-based formula that would understate uncertainty."""
    se_ln_lower = (np.log(ref["lower_ci"][1]) - np.log(ref["lower_ci"][0])) / (2 * 1.96)
    se_ln_upper = (np.log(ref["upper_ci"][1]) - np.log(ref["upper_ci"][0])) / (2 * 1.96)
    se_log_mu = 0.5 * np.sqrt(se_ln_lower**2 + se_ln_upper**2)
    se_log_sd = (se_ln_lower**2 + se_ln_upper**2) ** 0.5 / (2 * 1.96)
    return se_log_mu, se_log_sd


def weighted_composite(Z, weights):
    w = np.array([weights[c] for c in Z.columns])
    return (Z.values * w).sum(axis=1) / w.sum()


# ---- Load and reshape external cohort ----
df = pd.read_csv("metabolites_external_cohort.csv")
df["CK"] = pd.to_numeric(df["CK"])
df["LDH"] = pd.to_numeric(df["LDH"])
df["hsCRP"] = pd.to_numeric(df["hsCRP"])

n = df["ID"].nunique()
print(f"n={n} (external cohort)")

ck_max = df.groupby("ID")["CK"].max()
# LDH: descriptive only, never classified against any reference interval.
ldh_max = df.groupby("ID")["LDH"].max()
print(f"LDH (descriptive only, not classified): mean={ldh_max.mean():.1f} "
      f"median={ldh_max.median():.1f} SD={ldh_max.std():.1f} "
      f"range=[{ldh_max.min():.1f}, {ldh_max.max():.1f}]")

crp_wide = df.pivot(index="ID", columns="Timepoint", values="hsCRP")
crp_t0 = crp_wide["T0"]
crp_t3 = crp_wide["T3"]  # 24h post-match, analogue of our own CRP_24h

# Floor check is descriptive only here (no externally-documented LOQ for this cohort's assay --
# see module docstring); it does NOT select which composite is primary (CK is always primary).
crp_t0_min = crp_t0.min()
n_at_floor = int((crp_t0 == crp_t0_min).sum())
frac_at_floor = n_at_floor / n
print(f"hsCRP_T0: min={crp_t0_min}, {n_at_floor}/{n} at sample minimum "
      f"-> floor fraction {frac_at_floor*100:.1f}% (descriptive only)")

crp_fold = crp_t3 / crp_t0.clip(lower=max(crp_t0_min, 1e-3))

Z = pd.DataFrame({
    "CK": one_sided_z(ck_max, REFERENCES["CK"]["log_mu"], REFERENCES["CK"]["log_sd"]),
    "CRP_fold": one_sided_z(crp_fold, REFERENCES["CRP_fold"]["log_mu"], REFERENCES["CRP_fold"]["log_sd"]),
})

weights_full = {c: DEFAULT_WEIGHTS[c] for c in Z.columns}

composite_full = pd.Series(weighted_composite(Z, weights_full), index=Z.index)
# Primary composite is always CK alone -- a fixed, pre-specified choice, never selected at
# runtime based on this cohort's CRP behavior.
composite_primary = Z["CK"]

n_above_full = int((composite_full > ATTENTION_CUTOFF).sum())
n_above_primary = int((composite_primary > ATTENTION_CUTOFF).sum())

print(f"\nPrimary composite (CK alone): mean={composite_primary.mean():.3f}, above cutoff={n_above_primary}/{n}")
print(f"Full composite (CK+CRP, diagnostic only): mean={composite_full.mean():.3f}, above cutoff={n_above_full}/{n}")

print("\n--- Per-subject primary (CK-only) composite (sorted) ---")
for sid, val in composite_primary.sort_values(ascending=False).items():
    flag = " <-- ABOVE CUTOFF" if val > ATTENTION_CUTOFF else ""
    print(f"{sid}: CK_max={ck_max[sid]:.1f} LDH_max(descriptive)={ldh_max[sid]:.1f} composite={val:.3f}{flag}")

# Bootstrap over CK's reference-source uncertainty only (see 04_probabilistic_evidence_score.py's
# docstring for why CRP is excluded: no real subject-level sample size behind its "1.5-3x"
# summary). Vectorized over all draws at once, matching 04_probabilistic_evidence_score.py --
# an external review found the reported interval's upper bound sat within Monte Carlo noise of
# the 97.5th-percentile decision boundary at the earlier 3000-draw count; the vectorized draw
# makes a much larger count (20000, up from 3000) computationally free.
rng = np.random.default_rng(42)
n_bootstrap = 20000
ck_ref = REFERENCES["CK"]
se_mu, se_sd = published_ci_log_mu_sd_se(ck_ref)
mu_b = rng.normal(ck_ref["log_mu"], se_mu, size=n_bootstrap)
sd_b = np.clip(rng.normal(ck_ref["log_sd"], se_sd, size=n_bootstrap), 1e-3, None)
log_ck = np.log(ck_max.values)[:, None]
z_b = np.maximum((log_ck - mu_b[None, :]) / sd_b[None, :], 0.0)
boot_n_above = (z_b > ATTENTION_CUTOFF).sum(axis=0)

ci_lo, ci_hi = np.percentile(boot_n_above, [2.5, 97.5])
# Continuous companion metric to the discrete CI -- see 04_probabilistic_evidence_score.py's
# matching comment for why this is reported alongside it.
frac_at_least_1 = float((boot_n_above >= 1).mean())
print(f"\nBootstrap over CK's reference-source uncertainty only ({n_bootstrap} draws): "
      f"median n above cutoff={np.median(boot_n_above):.0f}, 95% CI=[{ci_lo:.0f}, {ci_hi:.0f}], "
      f"fraction of draws with >=1 subject above cutoff={frac_at_least_1*100:.1f}%")

# Also report CK against its reference interval directly (same style as manuscript's
# "Literature-Informed Classification" subsection). LDH is intentionally NOT compared against
# any threshold here -- see module docstring.
n_ck_exceed = int((ck_max > 1492).sum())
print(f"\nSubjects exceeding CK upper reference limit (1492 U/L): {n_ck_exceed}/{n}")

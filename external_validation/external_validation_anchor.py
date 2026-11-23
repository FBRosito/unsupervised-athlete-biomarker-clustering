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

Difference from our own cohort's timepoints: our data has Pre/Post/24h (3 points); this external
dataset has 5 points (T0=24h pre, T1=60min pre, T2=post, T3=24h post, T4=48h post). CK uses the
max across ALL available timepoints in both cases (consistent with our own script's logic, just
applied to a different number of timepoints). CRP fold-change uses T3 (24h post) / T0 (pre-match
baseline), the direct analogue of our own CRP_24h / CRP_Pre.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

# ---- Copied verbatim from notebooks/v2_reanalysis/04_probabilistic_evidence_score.py ----
REFERENCES = {
    "CK": {"n_reference": 182, "log_mu": float(np.log(np.sqrt(83 * 1492))),
           "log_sd": float((np.log(1492) - np.log(83)) / (2 * 1.96))},
    "CRP_fold": {"n_reference": 8, "log_mu": float(np.log(np.sqrt(1.5 * 3.0))),
                 "log_sd": float((np.log(3.0) - np.log(1.5)) / (2 * 1.96))},
}
DEFAULT_WEIGHTS = {"CK": 1.0, "CRP_fold": 1.0}
ATTENTION_CUTOFF = norm.ppf(0.95)


def one_sided_z(value, log_mu, log_sd):
    return np.maximum((np.log(value) - log_mu) / log_sd, 0.0)


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

# Assay-floor check on CRP_Pre (T0), same logic as common.py / 04_probabilistic_evidence_score.py:
# flag if a large fraction of baseline values sit at (or near) the sample minimum.
crp_t0_min = crp_t0.min()
n_at_floor = int((crp_t0 == crp_t0_min).sum())
frac_at_floor = n_at_floor / n
crp_floor_flag = frac_at_floor >= 0.3
print(f"hsCRP_T0: min={crp_t0_min}, {n_at_floor}/{n} at sample minimum "
      f"-> floor fraction {frac_at_floor*100:.1f}% flagged unstable: {crp_floor_flag}")

crp_fold = crp_t3 / crp_t0.clip(lower=max(crp_t0_min, 1e-3))

Z = pd.DataFrame({
    "CK": one_sided_z(ck_max, REFERENCES["CK"]["log_mu"], REFERENCES["CK"]["log_sd"]),
    "CRP_fold": one_sided_z(crp_fold, REFERENCES["CRP_fold"]["log_mu"], REFERENCES["CRP_fold"]["log_sd"]),
})

weights_full = {c: DEFAULT_WEIGHTS[c] for c in Z.columns}

composite_full = pd.Series(weighted_composite(Z, weights_full), index=Z.index)
composite_ck_only = Z["CK"]

n_above_full = int((composite_full > ATTENTION_CUTOFF).sum())
n_above_ck_only = int((composite_ck_only > ATTENTION_CUTOFF).sum())

print(f"\nComposite FULL (CK+CRP): mean={composite_full.mean():.3f}, above cutoff={n_above_full}/{n}")
print(f"Composite CK only: mean={composite_ck_only.mean():.3f}, above cutoff={n_above_ck_only}/{n}")
print(f"Recommended primary composite: {'CK only' if crp_floor_flag else 'FULL'}")

print("\n--- Per-subject CK-only composite (sorted) ---")
for sid, val in composite_ck_only.sort_values(ascending=False).items():
    flag = " <-- ABOVE CUTOFF" if val > ATTENTION_CUTOFF else ""
    print(f"{sid}: CK_max={ck_max[sid]:.1f} LDH_max(descriptive)={ldh_max[sid]:.1f} composite={val:.3f}{flag}")

# Bootstrap over reference-source uncertainty (identical procedure to our own script)
rng = np.random.default_rng(42)
n_bootstrap = 3000
boot_n_above = []
for _ in range(n_bootstrap):
    z_b = {}
    for name, ref in REFERENCES.items():
        se_mu = ref["log_sd"] / np.sqrt(ref["n_reference"])
        se_sd = ref["log_sd"] / np.sqrt(2 * max(ref["n_reference"] - 1, 1))
        mu_b = rng.normal(ref["log_mu"], se_mu)
        sd_b = max(rng.normal(ref["log_sd"], se_sd), 1e-3)
        val = {"CK": ck_max, "CRP_fold": crp_fold}[name]
        z_b[name] = one_sided_z(val, mu_b, sd_b)
    Zb = pd.DataFrame(z_b)
    comp_b = Zb["CK"]
    boot_n_above.append(int((comp_b > ATTENTION_CUTOFF).sum()))

boot_n_above = np.array(boot_n_above)
ci_lo, ci_hi = np.percentile(boot_n_above, [2.5, 97.5])
print(f"\nBootstrap over reference-source uncertainty ({n_bootstrap} draws): "
      f"median n above cutoff={np.median(boot_n_above):.0f}, 95% CI=[{ci_lo:.0f}, {ci_hi:.0f}]")

# Also report CK against its reference interval directly (same style as manuscript's
# "Literature-Informed Classification" subsection). LDH is intentionally NOT compared against
# any threshold here -- see module docstring.
n_ck_exceed = int((ck_max > 1492).sum())
print(f"\nSubjects exceeding CK upper reference limit (1492 U/L): {n_ck_exceed}/{n}")

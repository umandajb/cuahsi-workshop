# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 10:02:20 2026

@author: ov4dc
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nsEVDx as ns

from scipy.stats import genextreme
from pathlib import Path

# ============================================================
# 1. INPUT DATA
# ============================================================

INPUT_CSV = Path(r"06923950_PeakFlow.csv")

OUTPUT_FOLDER = INPUT_CSV.parent / "Stationary_GEV_nsEVDx"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_CSV)

df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
df["PeakFlow_cfs"] = pd.to_numeric(
    df["PeakFlow_cfs"], errors="coerce"
)

df = df.dropna(subset=["Year", "PeakFlow_cfs"])
df = df[df["PeakFlow_cfs"] > 0].copy()
df = df.sort_values("Year")

if df["Year"].duplicated().any():
    raise ValueError("Duplicate years found. Check annual peak  streamflow.")

Q = df["PeakFlow_cfs"].to_numpy(dtype=float)
n = len(Q)

if n < 20:
    raise ValueError("At least 20 observations are required.")

print(f"Number of observations: {n}")
print(f"Period: {int(df['Year'].min())} - "
      f"{int(df['Year'].max())}")

# ============================================================
# 2. STATIONARY GEV CONFIGURATION
# ============================================================

# No covariates in location, scale or shape
config = [0, 0, 0]

# Empty covariate matrix: no covariates needed
cov = np.empty((0, n))

# Standardize discharge for numerical stability
Q_mean = np.mean(Q)
Q_std = np.std(Q)

if Q_std <= 0:
    raise ValueError("Peak flows have zero variability.")

Qz = (Q - Q_mean) / Q_std

# ============================================================
# 3. INITIAL PARAMETERS
# ============================================================

# SciPy initial estimates on standardized data
c0, loc0, scale0 = genextreme.fit(Qz)

# nsEVDx parameter order:
# [location, scale, shape]
# Shape uses SciPy convention c = -xi

initial_params = [loc0, scale0, c0]

print("\nINITIAL PARAMETERS")
print(initial_params)

# ============================================================
# 4. FIT STATIONARY GEV USING nsEVDx
# ============================================================

model = ns.NonStationaryEVD(
    config=config,
    data=Qz,
    cov=cov,
    dist=genextreme
)

params = np.asarray(
    model.frequentist_nsEVD(
        initial_params=initial_params,
        max_retries=10
    ),
    dtype=float
)

if params.size != 3 or not np.all(np.isfinite(params)):
    raise RuntimeError("Invalid GEV parameter estimates.")

# Fitted parameters in standardized units
loc_z, scale_z, shape = params

if scale_z <= 0:
    raise RuntimeError("Estimated scale must be positive.")

# Transform back to original discharge units
mu = loc_z * Q_std + Q_mean
sigma = scale_z * Q_std
xi = -shape

print("\nSTATIONARY GEV PARAMETERS (nsEVDx MLE)")
print("-------------------------------------")
print(f"Location (mu): {mu:,.4f} cfs")
print(f"Scale (sigma): {sigma:,.4f} cfs")
print(f"Shape (xi): {xi:.6f}")

# ============================================================
# 5. LOG-LIKELIHOOD, AIC AND BIC
# ============================================================

logpdf = genextreme.logpdf(
    Q,
    shape,
    loc=mu,
    scale=sigma
)

if not np.all(np.isfinite(logpdf)):
    raise RuntimeError("Non-finite GEV log-likelihood.")

log_likelihood = np.sum(logpdf)

k = 3

AIC = 2 * k - 2 * log_likelihood
BIC = k * np.log(n) - 2 * log_likelihood

print("\nMODEL STATISTICS")
print("----------------------------")
print(f"Log-likelihood: {log_likelihood:.4f}")
print(f"AIC: {AIC:.4f}")
print(f"BIC: {BIC:.4f}")

# ============================================================
# 6. FLOOD QUANTILES
# ============================================================

return_periods = np.array([2, 5, 10, 25, 50, 100])

# Annual non-exceedance probability
F = 1 - 1 / return_periods

Q_T = genextreme.ppf(
    F,
    shape,
    loc=mu,
    scale=sigma
)

results = pd.DataFrame({
    "ReturnPeriod_years": return_periods,
    "AEP": 1 / return_periods,
    "FloodQuantile_cfs": Q_T
})

print("\nFLOOD FREQUENCY RESULTS")
print(results.to_string(index=False))

results.to_csv(
    OUTPUT_FOLDER / "GEV_Return_Levels.csv",
    index=False
)

# ============================================================
# 7. EMPIRICAL RETURN PERIODS
# ============================================================

# Rank annual peaks from largest to smallest
Q_sorted = np.sort(Q)[::-1]

rank = np.arange(1, n + 1)

# Weibull plotting positions
T_empirical = (n + 1) / rank

empirical = pd.DataFrame({
    "Rank": rank,
    "PeakFlow_cfs": Q_sorted,
    "ReturnPeriod_years": T_empirical
})

empirical.to_csv(
    OUTPUT_FOLDER / "Empirical_Return_Periods.csv",
    index=False
)

# ============================================================
# 8. FLOOD FREQUENCY PLOT
# ============================================================

T_plot = np.geomspace(1.01, 500, 500)

Q_plot = genextreme.ppf(
    1 - 1 / T_plot,
    shape,
    loc=mu,
    scale=sigma
)

fig, ax = plt.subplots(figsize=(9, 6))

ax.scatter(
    T_empirical,
    Q_sorted,
    color="black",
    s=35,
    label="Observed (Weibull)",
    zorder=3
)

ax.plot(
    T_plot,
    Q_plot,
    color="red",
    linewidth=2,
    label="Stationary GEV (nsEVDx MLE)"
)

ax.scatter(
    return_periods,
    Q_T,
    color="red",
    marker="s",
    s=40,
    zorder=4
)

ax.set_xscale("log")

ax.set_xlabel("Return Period (years)", fontsize=12)
ax.set_ylabel("Peak Discharge (cfs)", fontsize=12)

ax.set_title(
    "Stationary GEV Flood Frequency Analysis\n"
    "USGS 06923950",
    fontsize=13
)

ax.set_xticks([2, 5, 10, 25, 50, 100, 200, 500])
ax.set_xticklabels(["2", "5", "10", "25", "50",
                    "100", "200", "500"])

ax.grid(True, which="both", alpha=0.3)
ax.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_FOLDER / "Stationary_GEV_FFA.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 9. ANNUAL PEAK STREAMFLOW VS EXCEEDANCE PROBABILITY
# ============================================================

# Empirical AEP using Weibull plotting positions
AEP_empirical = rank / (n + 1)

# Theoretical AEP from fitted stationary GEV
AEP_plot = np.linspace(0.001, 0.999, 1000)

Q_AEP = genextreme.ppf(
    1 - AEP_plot,
    shape,
    loc=mu,
    scale=sigma
)

# AEP values corresponding to selected return periods
AEP_selected = 1 / return_periods

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 6))

# Observed annual peak discharge
ax.scatter(
    AEP_empirical * 100,
    Q_sorted,
    color="black",
    s=35,
    alpha=0.8,
    label="Observed (Weibull)",
    zorder=3
)

# Fitted stationary GEV
ax.plot(
    AEP_plot * 100,
    Q_AEP,
    color="red",
    linewidth=2,
    label="Stationary GEV (nsEVDx MLE)",
    zorder=2
)

# Selected return-period quantiles
ax.scatter(
    AEP_selected * 100,
    Q_T,
    color="red",
    marker="s",
    s=45,
    label="GEV Return Levels",
    zorder=4
)

# Axis configuration
ax.set_xlabel(
    "Annual Exceedance Probability (%)",
    fontsize=12
)

ax.set_ylabel(
    "Annual Peak Streamflow (cfs)",
    fontsize=12
)

ax.set_title(
    "Stationary GEV Flood Frequency Analysis\n"
    "USGS 01013500",
    fontsize=13
)

# Exceedance probability from 0% to 100%
ax.set_xlim(0, 100)

ax.set_xticks(np.arange(0, 101, 10))

ax.grid(
    True,
    linestyle="--",
    alpha=0.3
)

ax.legend(fontsize=10)

plt.tight_layout()

plt.savefig(
    OUTPUT_FOLDER / "Stationary_GEV_AEP.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ------------------------------------------------------------
# Export empirical AEP values
# ------------------------------------------------------------

empirical["AEP"] = AEP_empirical
empirical["AEP_percent"] = AEP_empirical * 100

empirical.to_csv(
    OUTPUT_FOLDER / "Empirical_Return_Periods.csv",
    index=False
)


# ============================================================
# 10. EXPORT MODEL PARAMETERS
# ============================================================

parameters = pd.DataFrame([{
    "Model": "Stationary GEV",
    "Method": "nsEVDx MLE",
    "N": n,
    "Location_mu": mu,
    "Scale_sigma": sigma,
    "Shape_xi": xi,
    "LogLikelihood": log_likelihood,
    "AIC": AIC,
    "BIC": BIC
}])

parameters.to_csv(
    OUTPUT_FOLDER / "GEV_Parameters.csv",
    index=False
)

print("\nAnalysis completed.")
print(f"Results saved in: {OUTPUT_FOLDER}")

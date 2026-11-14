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

from scipy.optimize import minimize



# ============================================================
# 1. INPUT DATA
# ============================================================

INPUT_CSV = Path(r"06923950_PeakFlow.csv")

OUTPUT_FOLDER = INPUT_CSV.parent / "Nonstationary_GEV_LS_nsEVDx"
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
# 2. NONSTATIONARY GEV CONFIGURATION
# ============================================================

# Location and scale vary with time; shape remains constant
config = [1, 0, 0]

# Standardize discharge
Q_mean = np.mean(Q)
Q_std = np.std(Q)

if Q_std <= 0:
    raise ValueError("Peak flows have zero variability.")

Qz = (Q - Q_mean) / Q_std

# Standardize time covariate
years = df["Year"].to_numpy(dtype=float)

year_mean = np.mean(years)
year_std = np.std(years)

tz = (years - year_mean) / year_std

# Covariate matrix: one time covariate
cov = tz.reshape(1, -1)


# ============================================================
# 3. INITIAL PARAMETERS
# ============================================================

# Stationary GEV estimates for initialization
c0, loc0, scale0 = genextreme.fit(Qz)

# Parameter order:
# [mu0, mu1, log_sigma0, shape]
#
# mu(t) = mu0 + mu1 * tz
# sigma(t) = exp(log_sigma0)
# shape = constant

initial_params = np.array([
    loc0,
    0.0,
    np.log(scale0),
    c0
])

print("\nINITIAL PARAMETERS")
print(initial_params)


# ============================================================
# 4. FIT NONSTATIONARY GEV USING MLE
# ============================================================

def negative_log_likelihood(params):

    mu0, mu1, log_sigma0, shape = params

    # Time-varying location
    mu_t = mu0 + mu1 * tz

    # Constant scale
    sigma_t = np.exp(log_sigma0)

    # GEV log-likelihood
    logpdf = genextreme.logpdf(
        Qz,
        shape,
        loc=mu_t,
        scale=sigma_t
    )

    if not np.all(np.isfinite(logpdf)):
        return 1e20

    return -np.sum(logpdf)


# Parameter bounds
bounds = [
    (None, None),  # Location intercept
    (None, None),  # Location slope
    (-10, 10),     # Log-scale intercept
    (-1, 1)        # Shape (SciPy convention)
]

# Optimize negative log-likelihood
result = minimize(
    negative_log_likelihood,
    initial_params,
    method="L-BFGS-B",
    bounds=bounds,
    options={
        "maxiter": 10000,
        "ftol": 1e-12
    }
)

if not result.success:
    raise RuntimeError(
        f"MLE optimization failed: {result.message}"
    )

# Estimated coefficients
params = result.x

mu0, mu1, log_sigma0, shape = params

# Annual location parameter
mu_z = mu0 + mu1 * tz

# Constant scale parameter
sigma_z = np.full(n, np.exp(log_sigma0))

# Transform back to original discharge units
mu = mu_z * Q_std + Q_mean
sigma = sigma_z * Q_std

# Convert SciPy shape to conventional GEV shape
xi = -shape

print("\nNONSTATIONARY GEV LOCATION MODEL (MLE)")
print("-------------------------------------")
print(f"Location intercept: {mu0:.6f}")
print(f"Location slope: {mu1:.6f}")
print(f"Log-scale intercept: {log_sigma0:.6f}")
print(f"Constant scale (cfs): {sigma[0]:.4f}")
print(f"Shape (xi): {xi:.6f}")

print("\nOPTIMIZATION RESULTS")
print("-------------------------------------")
print(f"Converged: {result.success}")
print(f"Iterations: {result.nit}")
print(f"Negative log-likelihood: {result.fun:.6f}")



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

k = 4

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

F = 1 - 1 / return_periods

Q_T = genextreme.ppf(
    F[None, :],
    shape,
    loc=mu[:, None],
    scale=sigma[:, None]
)

results = pd.DataFrame(
    Q_T,
    columns=[f"Q{T}_cfs" for T in return_periods]
)

results.insert(0, "Year", years.astype(int))

print("\nNONSTATIONARY FLOOD FREQUENCY RESULTS")
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

# Reference year: last observed year
reference_year = int(years[-1])

mu_ref = mu[-1]
sigma_ref = sigma[-1]
Q_T_ref = Q_T[-1, :]

T_plot = np.geomspace(1.01, 500, 500)

Q_plot = genextreme.ppf(
    1 - 1 / T_plot,
    shape,
    loc=mu_ref,
    scale=sigma_ref
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
    label="Nontationary GEV"
)

ax.scatter(
    return_periods,
    Q_T_ref,
    color="red",
    marker="s",
    s=40,
    zorder=4
)

ax.set_xscale("log")

ax.set_xlabel("Return Period (years)", fontsize=12)
ax.set_ylabel("Peak Discharge (cfs)", fontsize=12)

ax.set_title(
    "Nonstationary GEV Flood Frequency Analysis\n"
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
    OUTPUT_FOLDER / "Nonstationary_GEV_FFA.png",
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
    loc=mu_ref,
    scale=sigma_ref
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
    label="Nonstationary GEV (nsEVDx MLE)",
    zorder=2
)

# Selected return-period quantiles
ax.scatter(
    AEP_selected * 100,
    Q_T_ref,
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
    "Nonstationary GEV Flood Frequency Analysis\n"
    "USGS 06923950",
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
    OUTPUT_FOLDER / "Nonstationary_GEV_AEP.png",
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
    "Model": "Nonstationary GEV L",
    "Method": "Direct MLE (L-BFGS-B)",
    "N": n,
    "Location_intercept": mu0,
    "Location_slope": mu1,
    "LogScale_intercept": log_sigma0,
    "Scale_sigma_cfs": sigma[0],
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


# ============================================================
# 11. NONSTATIONARY GEV ENVELOPE - ALL YEARS
# ============================================================

# Annual exceedance probability
AEP_plot = np.linspace(0.001, 0.999, 1000)

# Non-exceedance probability
F_plot = 1 - AEP_plot

# Calculate theoretical GEV curves for every year
# Rows = years
# Columns = exceedance probabilities

Q_all_years = genextreme.ppf(
    F_plot[None, :],
    shape,
    loc=mu[:, None],
    scale=sigma[:, None]
)

# Lower and upper envelope across all years
Q_lower = np.min(Q_all_years, axis=0)
Q_upper = np.max(Q_all_years, axis=0)

# GEV curves for first and last observed years
Q_first = Q_all_years[0, :]
Q_last = Q_all_years[-1, :]

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 6))

# Shaded area between minimum and maximum annual GEV curves
ax.fill_between(
    AEP_plot * 100,
    Q_lower,
    Q_upper,
    color="red",
    alpha=0.15,
    label="Nonstationary GEV Envelope",
    zorder=1
)

# First-year GEV curve
ax.plot(
    AEP_plot * 100,
    Q_first,
    color="blue",
    linewidth=2,
    label=f"GEV {int(years[0])}",
    zorder=2
)

# Last-year GEV curve
ax.plot(
    AEP_plot * 100,
    Q_last,
    color="red",
    linewidth=2,
    label=f"GEV {int(years[-1])}",
    zorder=3
)

# Observed annual peak streamflow (Weibull)
ax.scatter(
    AEP_empirical * 100,
    Q_sorted,
    color="black",
    edgecolor="black",
    s=35,
    alpha=0.8,
    label="Observed (Weibull)",
    zorder=4
)

# ------------------------------------------------------------
# Axis configuration
# ------------------------------------------------------------

ax.set_xlabel(
    "Annual Exceedance Probability (%)",
    fontsize=12
)

ax.set_ylabel(
    "Annual Peak Streamflow (cfs)",
    fontsize=12
)

ax.set_title(
    "Nonstationary GEV Flood Frequency Envelope\n"
    "USGS 06923950",
    fontsize=13
)

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
    OUTPUT_FOLDER / "Nonstationary_GEV_Envelope.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

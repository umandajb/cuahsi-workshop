# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 10:19:05 2026

@author: ov4dc
"""

# if not installed, first install.

import dataretrieval.nwis as nwis
import pandas as pd


# USER SETTINGS
#-------------------

GAUGE_ID = "06923950"

MODE = "all"  # Options: "range" or "all"
START_YEAR = 1985
END_YEAR = 2024

OUTPUT_CSV = f"{GAUGE_ID}_PeakFlow.csv"


# DOWNLOAD ANNUAL PEAK FLOW
#-----------------------------

if MODE == "range":
    df, md = nwis.get_discharge_peaks(
        sites=GAUGE_ID,
        start=f"{START_YEAR}-01-01",
        end=f"{END_YEAR}-12-31"
    )
else:
    df, md = nwis.get_discharge_peaks(sites=GAUGE_ID)

# Extract water year and peak discharge (cfs)
df["peak_dt"] = pd.to_datetime(df["peak_dt"])

df["Year"] = df["peak_dt"].dt.year + (
    df["peak_dt"].dt.month >= 10
).astype(int)

df["PeakFlow_cfs"] = pd.to_numeric(
    df["peak_va"], errors="coerce"
)

# Filter by water year
if MODE == "range":
    df = df[df["Year"].between(START_YEAR, END_YEAR)]

# Save CSV
df = df[["Year", "PeakFlow_cfs"]].sort_values("Year")
df.to_csv(OUTPUT_CSV, index=False)

print(f"Downloaded {len(df)} annual peaks")
print(f"Saved to: {OUTPUT_CSV}")

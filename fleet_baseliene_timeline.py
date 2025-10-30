# fleet_baseline_timeline.py
# Timeline NOx: baseline vs SSW scenarios (+ N-1 taxi variants), Oct-2025 → May-2031
# Uses Nexa font & brand colors; cumulative savings vs baseline.

from __future__ import annotations
import datetime as dt
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

# ---------- Branding / Fonts ----------
PRIMARY_COLOR   = "#0cd66d"     # your primary
SECONDARY_COLOR = "#2d099d"     # your secondary
ALT1_COLOR      = "#fc3968"     # extra scenario color
ALT2_COLOR      = "#fce939"     # extra scenario color
BASELINE_COLOR  = "#888888"

FONT_PATH = "Nexa-ExtraLight.ttf"  # ensure this path is valid in your env

def use_nexa():
    try:
        fm.fontManager.addfont(FONT_PATH)
        fam = fm.FontProperties(fname=FONT_PATH).get_name()
        mpl.rcParams["font.family"] = fam
        print(f"[INFO] Using custom font: {fam}")
    except Exception as e:
        print(f"[WARN] Could not load '{FONT_PATH}': {e}. Using default font.")

use_nexa()
mpl.rc("grid", color="darkgray", linestyle=":")
mpl.rcParams["font.size"] = 11

# ---------- Imports from your SSW module ----------
# We rely on your improved generator and NOx computation here.
import ssw_vs_blended737 as ssw

# ---------- Configuration ----------
MEAN_KM = 1325.47
SD_KM   = 640.22
DT_SECONDS = 100
FLIGHTS_PER_AIRCRAFT_PER_MONTH = 100

# Start/end months
START_MONTH = dt.date(2025, 10, 1)   # Oct 2025
END_MONTH   = dt.date(2031, 5, 1)    # May 2031

# Fleet (starting month)
START_A321 = 13
START_B738 = 37

# B738 phase-out schedule (YYYY-MM in your table)
# If multiple aircraft leave the same month, put both entries with that month.
PHASEOUTS_YM = [
    "2025-09", "2025-10", "2025-11",
    "2026-01", "2026-04", "2026-07", "2026-12",
    "2027-08", "2027-09", "2027-10", "2027-11", "2027-12",
    "2028-01", "2028-08", "2028-09", "2028-10", "2028-12",
    "2029-01", "2029-07", "2029-08", "2029-12",
    "2030-01", "2030-03", "2030-04", "2030-05", "2030-06", "2030-08", "2030-09", "2030-12",
    "2031-01", "2031-02", "2031-03", "2031-05", "2031-08",
]
# The list above is built from your phase-out table lines; add/remove to match exactly.
# Important: we only apply replacements from the first month >= START_MONTH.

# 5 aircraft are already SSW today (honor this for stagger scenario)
ALREADY_SSW_COUNT = 5

# ---------- N-1 taxi parameters (NOx savings only; kg per flight) ----------
# From your ICAS-based constants: 209.714 g per taxi-out; taxi-in scaled by 6.0/12.3
NOX_SAVING_PER_TAXI_OUT_KG = 0.209714
TAXI_IN_SCALE = 6.0 / 12.3   # ≈ 0.4878
NOX_SAVING_PER_TAXI_IN_KG  = NOX_SAVING_PER_TAXI_OUT_KG * TAXI_IN_SCALE
NOX_SAVING_OUT_ONLY_PER_FLIGHT_KG = NOX_SAVING_PER_TAXI_OUT_KG
NOX_SAVING_OUT_IN_PER_FLIGHT_KG   = NOX_SAVING_PER_TAXI_OUT_KG * (1.0 + TAXI_IN_SCALE)

# ---------- Helpers ----------
def month_range(start_month: dt.date, end_month: dt.date) -> List[pd.Timestamp]:
    months = []
    y, m = start_month.year, start_month.month
    while (y, m) <= (end_month.year, end_month.month):
        months.append(pd.Timestamp(year=y, month=m, day=1))
        m += 1
        if m > 12:
            m = 1; y += 1
    return months

def phaseout_counts_by_month(months: List[pd.Timestamp]) -> Dict[pd.Timestamp, int]:
    # Count phaseouts per YYYY-MM and map onto the months list (>= START_MONTH)
    counts = {}
    from collections import Counter
    c = Counter(PHASEOUTS_YM)
    for ts in months:
        ym = f"{ts.year:04d}-{ts.month:02d}"
        counts[ts] = c.get(ym, 0)
    return counts

# ---------- Calibrate per-type NOx for 100 flights (normal dist) ----------
@dataclass
class PerTypeNOx:
    # per 100 flights totals (kg)
    b738_bw_kg: float
    b738_ssw_kg: float
    a321neo_kg: float

def simulate_month_per_type() -> PerTypeNOx:
    # Generate one 100-flight kinematics set
    flights_kin, _ = ssw.generate_flights_with_range_distribution(
        ssw.PARAMS,
        mean_km=MEAN_KM,
        sd_km=SD_KM,
        min_km=80.0,
        max_km=6000.0,
        max_tries_per_flight=40,
    )

    # Compute NOx for B738-BW vs B738-SSW vs A21N on SAME kinematics.
    # We reuse the ssw.compute_fuel_and_nox_multi: pass two variants for 738 flavors,
    # and a separate call for A21N (or extend the function call to 3 variants at once).
    # Simpler: do two calls — (BW vs SSW) + (A21N).

    # (A) BW vs SSW
    combined_738 = ssw.compute_fuel_and_nox_multi(
        flights=flights_kin,
        variants=ssw.VARIANTS,  # B738-BW and B738-SSW
        mass_takeoff_default=ssw.PARAMS["mass_takeoff"],
        mass_takeoff_by_ac=ssw.PARAMS.get("mass_takeoff_by_ac", {}),
        dt=DT_SECONDS,
        vs_clip_fpm=ssw.PARAMS["vs_fpm_clip"],
        tas_min_flight=ssw.PARAMS["tas_min_flight"],
        alt_min_flight=ssw.PARAMS["alt_min_flight"],
        ef_co2_kg_per_kg=ssw.PARAMS["ef_co2_kg_per_kg"],
    )
    sum_738 = (combined_738.groupby("aircraft")
               .agg(total_nox_g=("nox_step_g", "sum"))
               .total_nox_g.to_dict())
    bw_kg  = sum_738.get("B738-BW", 0.0) / 1000.0
    ssw_kg = sum_738.get("B738-SSW", 0.0) / 1000.0

    # (B) A321neo (A21N)
    # Quick one-off via openap on the same flights (reuse the helper approach)
    import openap
    emis_A21N = openap.Emission("A21N", use_synonym=True)
    # We need a fuel model to get NOx; ssw.compute* uses fuel->NOx internally.
    # Here, approximate using B738 fuel flow model for mass stepping would be wrong.
    # Better: reuse ssw.compute_fuel_and_nox_multi but with a fake VARIANTS list for A21N code-path.

    variants_a21n = [{"label": "A21N", "ac_code": "A21N", "ff_scale": 1.0}]
    combined_a21n = ssw.compute_fuel_and_nox_multi(
        flights=flights_kin,
        variants=variants_a21n,
        mass_takeoff_default=ssw.PARAMS["mass_takeoff"],
        mass_takeoff_by_ac={"A21N": 70_000.0},
        dt=DT_SECONDS,
        vs_clip_fpm=ssw.PARAMS["vs_fpm_clip"],
        tas_min_flight=ssw.PARAMS["tas_min_flight"],
        alt_min_flight=ssw.PARAMS["alt_min_flight"],
        ef_co2_kg_per_kg=ssw.PARAMS["ef_co2_kg_per_kg"],
    )
    a21n_kg = combined_a21n["nox_step_g"].sum() / 1000.0

    return PerTypeNOx(b738_bw_kg=bw_kg, b738_ssw_kg=ssw_kg, a321neo_kg=a21n_kg)

# Cache once for speed
_per_type_cache = None
def per_type_nox_100() -> PerTypeNOx:
    global _per_type_cache
    if _per_type_cache is None:
        _per_type_cache = simulate_month_per_type()
        print("[INFO] Calibrated per-100-flight NOx (kg):",
              _per_type_cache)
    return _per_type_cache

# ---------- Fleet evolution ----------
def evolve_fleet(months: List[pd.Timestamp]) -> pd.DataFrame:
    """
    Returns a frame with columns:
      month, a321_count, b738_count
    applying all 737 phase-outs (1:1 replaced by A321) from START_MONTH onward.
    """
    df = pd.DataFrame({"month": months})
    df["a321_count"] = START_A321
    df["b738_count"] = START_B738

    po_counts = phaseout_counts_by_month(months)
    for i, m in enumerate(months):
        if i == 0:
            continue
        # propagate previous month's counts
        df.loc[i, "a321_count"] = df.loc[i-1, "a321_count"]
        df.loc[i, "b738_count"] = df.loc[i-1, "b738_count"]

        # apply this month's phase-outs (if month >= START_MONTH)
        n_po = po_counts.get(m, 0)
        if n_po > 0:
            df.loc[i, "b738_count"] = max(0, df.loc[i, "b738_count"] - n_po)
            df.loc[i, "a321_count"] = df.loc[i, "a321_count"] + n_po

    return df

# ---------- Staggered SSW plan (–25 months) ----------
def month_offset(ts: pd.Timestamp, months: int) -> pd.Timestamp:
    y, m = ts.year, ts.month
    m += months
    while m <= 0:
        m += 12; y -= 1
    while m > 12:
        m -= 12; y += 1
    return pd.Timestamp(year=y, month=m, day=1)

def stagger_ssw_activation_months() -> List[pd.Timestamp]:
    """
    For each 737 retirement in PHASEOUTS_YM, activate SSW 25 months earlier.
    Also assume ALREADY_SSW_COUNT active at START_MONTH.
    Returns a list of activation timestamps (can include duplicates if multiple in same month).
    """
    acts = []
    for ym in PHASEOUTS_YM:
        y, m = map(int, ym.split("-"))
        retire = pd.Timestamp(year=y, month=m, day=1)
        act = month_offset(retire, -25)
        if act >= pd.Timestamp(START_MONTH):
            acts.append(act)
    # We'll handle the “already SSW” by counting them as active from START_MONTH.
    return acts

# ---------- Monthly NOx totals per scenario ----------
def compute_monthly_totals() -> pd.DataFrame:
    months = month_range(START_MONTH, END_MONTH)
    fleet = evolve_fleet(months)
    per = per_type_nox_100()

    # per-aircraft per-month NOx (kg) for 100 flights
    nox_b738_bw_kg  = per.b738_bw_kg
    nox_b738_ssw_kg = per.b738_ssw_kg
    nox_a21n_kg     = per.a321neo_kg

    # Baseline: BW for all 737s; A321s as-is
    baseline_kg = []
    for _, row in fleet.iterrows():
        total = (row.b738_count * nox_b738_bw_kg) + (row.a321_count * nox_a21n_kg)
        baseline_kg.append(total)

    # All SSW now: all 737s use SSW from t=0
    allssw_kg = []
    for _, row in fleet.iterrows():
        total = (row.b738_count * nox_b738_ssw_kg) + (row.a321_count * nox_a21n_kg)
        allssw_kg.append(total)

    # Staggered SSW: –25 mo activation, honoring 5 already SSW at t0
    ssw_acts = stagger_ssw_activation_months()
    # Build cumulative SSW-active 737 count month-by-month
    # Start with min(ALREADY_SSW_COUNT, current 737s)
    ssw_active_counts = []
    active = 0
    for i, m in enumerate(months):
        # start with previous
        if i == 0:
            active = min(ALREADY_SSW_COUNT, START_B738)
        else:
            active = min(active, int(fleet.loc[i, "b738_count"]))  # cap by current 737s

        # add any activations this month
        n_new = sum(1 for ts in ssw_acts if ts == m)
        if n_new:
            active = min(active + n_new, int(fleet.loc[i, "b738_count"]))

        ssw_active_counts.append(active)

    stagger_kg = []
    for i, row in fleet.iterrows():
        ssw_737 = ssw_active_counts[i]
        bw_737  = max(0, int(row.b738_count) - ssw_737)
        total = (ssw_737 * nox_b738_ssw_kg) + (bw_737 * nox_b738_bw_kg) + (row.a321_count * nox_a21n_kg)
        stagger_kg.append(total)

    # N-1 taxi savings layered on top of ALL-SSW NOW
    # Compute per-month NOx saved by N-1, applied to ALL aircraft (737 + A321),
    # since taxi measure is operational and aircraft-agnostic.
    n1_out_only_saved_kg    = []
    n1_out_in_saved_kg      = []
    for _, row in fleet.iterrows():
        flights_month_total = int(row.b738_count + row.a321_count) * FLIGHTS_PER_AIRCRAFT_PER_MONTH
        save_out   = flights_month_total * NOX_SAVING_OUT_ONLY_PER_FLIGHT_KG
        save_outin = flights_month_total * NOX_SAVING_OUT_IN_PER_FLIGHT_KG
        n1_out_only_saved_kg.append(save_out)
        n1_out_in_saved_kg.append(save_outin)

    # Scenario timelines (absolute totals per month)
    # - baseline_kg
    # - allssw_kg
    # - stagger_kg
    # - allssw + n1 out only (subtract savings from totals)
    # - allssw + n1 out & in
    allssw_n1out_kg   = np.array(allssw_kg) - np.array(n1_out_only_saved_kg)
    allssw_n1outin_kg = np.array(allssw_kg) - np.array(n1_out_in_saved_kg)

    df = pd.DataFrame({
        "month": months,
        "a321": fleet["a321_count"].to_numpy(),
        "b738": fleet["b738_count"].to_numpy(),
        "baseline_kg": np.array(baseline_kg),
        "allssw_kg":   np.array(allssw_kg),
        "stagger_kg":  np.array(stagger_kg),
        "allssw_n1out_kg":   allssw_n1out_kg,
        "allssw_n1outin_kg": allssw_n1outin_kg,
    })
    return df

# ---------- Plot: cumulative savings vs baseline ----------
def plot_cumulative_savings(df: pd.DataFrame, save_pdf: str | None = None):
    # Savings per month vs baseline
    s_allssw    = df["baseline_kg"] - df["allssw_kg"]
    s_stagger   = df["baseline_kg"] - df["stagger_kg"]
    s_n1out     = df["baseline_kg"] - df["allssw_n1out_kg"]
    s_n1outin   = df["baseline_kg"] - df["allssw_n1outin_kg"]

    # Cumulative
    cs_allssw   = s_allssw.cumsum()
    cs_stagger  = s_stagger.cumsum()
    cs_n1out    = s_n1out.cumsum()
    cs_n1outin  = s_n1outin.cumsum()

    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    # Baseline: zero line
    ax.axhline(0, color=BASELINE_COLOR, linestyle="--", linewidth=1.5, label="Baseline (no retrofit, no N–1)")

    ax.plot(df["month"], cs_allssw,   color=ALT2_COLOR,   linewidth=1, label="All SSW now")
    ax.plot(df["month"], cs_stagger,  color=SECONDARY_COLOR, linewidth=1, label="Staggered SSW (–25 mo)")
    ax.plot(df["month"], cs_n1out,    color=ALT1_COLOR,      linewidth=1, label="All SSW now + N–1 (out only)")
    ax.plot(df["month"], cs_n1outin,  color=PRIMARY_COLOR,      linewidth=1, label="All SSW now + N–1 (out & in)")

    ax.set_ylabel("Cumulative NOx saved vs baseline (kg)")
    ax.set_xlabel("Month")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    if save_pdf:
        fig.savefig(save_pdf, format="pdf", dpi=300, bbox_inches="tight")
        print(f"[INFO] Saved cumulative savings plot → {save_pdf}")
    plt.show()

def plot_monthly_savings(df: pd.DataFrame, save_pdf: str | None = None):
    """
    Monthly NOx savings vs baseline (kg per month)
    """
    # Monthly differences
    s_allssw    = df["baseline_kg"] - df["allssw_kg"]
    s_stagger   = df["baseline_kg"] - df["stagger_kg"]
    s_n1out     = df["baseline_kg"] - df["allssw_n1out_kg"]
    s_n1outin   = df["baseline_kg"] - df["allssw_n1outin_kg"]

    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    ax.axhline(0, color=BASELINE_COLOR, linestyle="--", linewidth=1.5, label="Baseline (no retrofit, no N–1)")

    ax.plot(df["month"], s_allssw,   color=ALT2_COLOR,   linewidth=1, label="All SSW now")
    ax.plot(df["month"], s_stagger,  color=SECONDARY_COLOR, linewidth=1, label="Staggered SSW (–25 mo)")
    ax.plot(df["month"], s_n1out,    color=ALT1_COLOR,      linewidth=1, label="All SSW now + N–1 (out only)")
    ax.plot(df["month"], s_n1outin,  color=PRIMARY_COLOR,      linewidth=1, label="All SSW now + N–1 (out & in)")

    ax.set_ylabel("Monthly NOx saved vs baseline (kg)")
    ax.set_xlabel("Month")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    if save_pdf:
        fig.savefig(save_pdf, format="pdf", dpi=300, bbox_inches="tight")
        print(f"[INFO] Saved monthly savings plot → {save_pdf}")
    plt.show()

def plot_monthly_emissions(df: pd.DataFrame, save_pdf: str | None = None):
    """
    Plot absolute NOx emitted per month (kg) for each scenario,
    with translucent fills under curves and Nexa font.
    """
    fig, ax = plt.subplots(figsize=(11.5, 5.2))

    # --- Plot lines ---
    ax.plot(df["month"], df["baseline_kg"],
            color=BASELINE_COLOR, linewidth=1, linestyle="--", label="Baseline (no retrofit, no N–1)")
    ax.plot(df["month"], df["allssw_kg"],
            color=ALT2_COLOR, linewidth=1, label="All SSW now")
    ax.plot(df["month"], df["stagger_kg"],
            color=SECONDARY_COLOR, linewidth=1, label="Staggered SSW (–25 mo)")
    ax.plot(df["month"], df["allssw_n1out_kg"],
            color=ALT1_COLOR, linewidth=1, label="All SSW now + N–1 (out only)")
    # ✅ Best-case: All SSW + N-1 out & in → bright green line + filled area
    ax.plot(df["month"], df["allssw_n1outin_kg"],
            color=PRIMARY_COLOR, linewidth=1, label="All SSW now + N–1 (out & in)")

    # --- Labels & grid ---
    ax.set_ylabel("Monthly NOx emitted (kg)")
    ax.set_xlabel("Month")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()

    # --- Save high-quality PDF ---
    if save_pdf:
        fig.savefig(save_pdf, format="pdf", dpi=300, bbox_inches="tight")
        print(f"[INFO] Saved monthly emissions plot → {save_pdf}")
    plt.show()

def plot_cumulative_emissions(df, save_pdf: str | None = None):
    """
    Cumulative NOx emitted (kg) per scenario (absolute, not savings), with fills to x-axis for scenarios.
    If cumulative columns are missing, compute them on the fly.
    Expects df monthly totals per scenario; uses columns listed below.
    """
    # Ensure cumulative columns exist (compute if needed)
    def ensure_cum(col_name):
        ccol = col_name.replace("_kg", "_cum")
        if ccol not in df.columns:
            df[ccol] = df[col_name].cumsum()
        return ccol

    c_base   = ensure_cum("baseline_kg")
    c_all    = ensure_cum("allssw_kg")
    c_stag   = ensure_cum("stagger_kg")
    c_n1out  = ensure_cum("allssw_n1out_kg")
    c_n1oi   = ensure_cum("allssw_n1outin_kg")

    fig, ax = plt.subplots(figsize=(11.5, 5.2))

    # Baseline absolute cumulative (dashed)
    ax.plot(df["month"], df[c_base], color=BASELINE_COLOR, linestyle="--", linewidth=1.6, label="Baseline (no retrofit, no N–1)")

    # Scenario lines
    ax.plot(df["month"], df[c_all],   color=ALT2_COLOR,    linewidth=1.0, label="All SSW now")
    ax.plot(df["month"], df[c_stag],  color=SECONDARY_COLOR, linewidth=1.0, label="Staggered SSW (–25 mo)")
    ax.plot(df["month"], df[c_n1out], color=ALT1_COLOR,  linewidth=1.0, label="All SSW now + N–1 (out only)")
    ax.plot(df["month"], df[c_n1oi],  color=PRIMARY_COLOR,  linewidth=1, label="All SSW now + N–1 (out & in)")

    ax.set_ylabel("Cumulative NOx emitted (kg)")
    ax.set_xlabel("Month")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()

    if save_pdf:
        fig.savefig(save_pdf, format="pdf", dpi=300, bbox_inches="tight")
        print(f"[INFO] Saved cumulative emissions plot → {save_pdf}")
    plt.show()

# ---------- Main ----------
if __name__ == "__main__":
    timeline = compute_monthly_totals()
    # Quick peek:
    print(timeline.head())

    # Cumulative savings vs baseline (PDF optional)
    # --- Plots ---
    plot_monthly_emissions(timeline, save_pdf="nox_monthly_emissions_timeline.pdf")
    plot_monthly_savings(timeline, save_pdf="nox_monthly_savings_timeline.pdf")
    plot_cumulative_savings(timeline, save_pdf="nox_cumulative_savings_timeline.pdf")
    plot_cumulative_emissions(timeline, save_pdf="nox_cumulative_emissions_timeline.pdf")

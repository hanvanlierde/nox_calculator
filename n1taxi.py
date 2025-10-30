"""
N-1 (single-engine) taxiing — Fleet totals with ETS, Fuel, and CO₂ savings
Saves both scenario and roadmap figures as high-quality PDF (300 dpi vector).
"""

from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import matplotlib as mpl
import os

# ===================== Font (Nexa) =====================
FONT_PATH = "Nexa-ExtraLight.ttf"  # <- update this path if needed

def configure_font(font_path: str):
    try:
        fm.fontManager.addfont(font_path)
        family_name = fm.FontProperties(fname=font_path).get_name()
        mpl.rcParams["font.family"] = family_name
        print(f"[INFO] Using custom font: {family_name}")
    except Exception as e:
        print(f"[WARN] Could not load '{font_path}': {e}. Using default font.")

configure_font(FONT_PATH)

# ===================== File saving =====================
SAVE_DIR = "./outputs"
os.makedirs(SAVE_DIR, exist_ok=True)

SCENARIO_FIG_PATH = os.path.join(SAVE_DIR, "n1_taxi_scenario_comparison.pdf")
ROADMAP_FIG_PATH  = os.path.join(SAVE_DIR, "n1_taxi_roadmap_2025_2030.pdf")

# --------------------- User inputs ---------------------
N_FLIGHTS             = 1200      # flights/year considered
ADOPTION_TAXI_OUT     = 1.00
ADOPTION_TAXI_IN      = 1.00

TARGET_TAXI_IN_MIN    = 6.0
TARGET_TAXI_OUT_MIN   = 12.3
TAXI_OUT_TIME_SCALE   = 1.00
TAXI_IN_TIME_SCALE    = TARGET_TAXI_IN_MIN / TARGET_TAXI_OUT_MIN  # ≈ 0.488

# --------------------- Economics (EUR) ---------------------
FUEL_PRICE_EUR_PER_L   = 1.20
FUEL_DENSITY_KG_PER_L  = 0.84
FUEL_PRICE_EUR_PER_KG  = FUEL_PRICE_EUR_PER_L / FUEL_DENSITY_KG_PER_L
ETS_PRICE_EUR_PER_TCO2 = 87.90
EURO = "€"

ETS_PRICE_BY_YEAR = {
    2025: 87.90,
    2026: 100.74,
    2027: 113.58,
    2028: 126.42,
    2029: 139.26,
    2030: 152.10,
}

# --------------------- ICAS constants -----------------
FUEL_SAVING_PER_TAXI_KG = 22.31
NOX_SAVING_PER_TAXI_G   = 209.714
CO2_PER_KG_FUEL         = 3.155

# --------------------- Colors --------------------------
COLOR_OUT_ONLY = "#2d099d"
COLOR_BOTH     = "#0cd66d"
COLOR_BASELINE = "#888888"

# --------------------- Data structure ------------------
@dataclass
class Totals:
    flights: int
    adopted_out: int
    adopted_in: int
    fuel_saved_kg: float
    nox_saved_kg: float
    co2_saved_kg: float
    fuel_value_eur: float
    ets_value_eur: float
    total_value_eur: float
    taxi_in_scale: float
    taxi_out_scale: float
    target_taxi_in_min: float
    target_taxi_out_min: float

# --------------------- Core calculator -----------------
def estimate_totals(
    n_flights: int,
    adopt_out: float,
    adopt_in: float,
    scale_out: float,
    scale_in: float,
    fuel_price_eur_per_kg: float,
    ets_price_eur_per_tco2: float,
    target_in_min: float,
    target_out_min: float,
) -> Totals:
    n_out = int(round(n_flights * adopt_out))
    n_in  = int(round(n_flights * adopt_in))

    fuel_save_out = FUEL_SAVING_PER_TAXI_KG * scale_out
    fuel_save_in  = FUEL_SAVING_PER_TAXI_KG * scale_in
    nox_save_out_g = NOX_SAVING_PER_TAXI_G * scale_out
    nox_save_in_g  = NOX_SAVING_PER_TAXI_G * scale_in

    fuel_saved_kg = n_out * fuel_save_out + n_in * fuel_save_in
    nox_saved_kg  = (n_out * nox_save_out_g + n_in * nox_save_in_g) / 1000.0
    co2_saved_kg  = fuel_saved_kg * CO2_PER_KG_FUEL

    fuel_value_eur = fuel_saved_kg * fuel_price_eur_per_kg
    ets_value_eur  = (co2_saved_kg / 1000.0) * ets_price_eur_per_tco2
    total_value_eur = fuel_value_eur + ets_value_eur

    return Totals(
        flights=n_flights,
        adopted_out=n_out,
        adopted_in=n_in,
        fuel_saved_kg=fuel_saved_kg,
        nox_saved_kg=nox_saved_kg,
        co2_saved_kg=co2_saved_kg,
        fuel_value_eur=fuel_value_eur,
        ets_value_eur=ets_value_eur,
        total_value_eur=total_value_eur,
        taxi_in_scale=scale_in,
        taxi_out_scale=scale_out,
        target_taxi_in_min=target_in_min,
        target_taxi_out_min=target_out_min,
    )

# --------------------- Scenario plotting ----------------
def _annotate_bars(ax, rects, fmt="{:,.0f}", top_margin_frac=0.14):
    heights = [r.get_height() for r in rects]
    vmax = max(heights) if heights else 1.0
    ax.set_ylim(-0.05 * vmax, (1.0 + top_margin_frac) * vmax)
    for r in rects:
        h = r.get_height()
        ax.text(r.get_x() + r.get_width()/2.0, h + 0.02 * vmax,
                fmt.format(h), ha="center", va="bottom", fontsize=10)

def _compute_scenario_totals(n_flights, adopt_out, adopt_in, ets):
    return estimate_totals(
        n_flights, adopt_out, adopt_in,
        TAXI_OUT_TIME_SCALE, TAXI_IN_TIME_SCALE,
        FUEL_PRICE_EUR_PER_KG, ets,
        TARGET_TAXI_IN_MIN, TARGET_TAXI_OUT_MIN
    )

def plot_n1_scenarios(n_flights, ets=ETS_PRICE_EUR_PER_TCO2, save_path=None):
    scen = [
        ("N-1 Taxi-Out Only", 1.0, 0.0, COLOR_OUT_ONLY),
        ("N-1 Out & In",      1.0, 1.0, COLOR_BOTH),
    ]
    results = [(name, _compute_scenario_totals(n_flights, ao, ai, ets), c)
               for name, ao, ai, c in scen]

    names = [n for n, _, _ in results]
    colors = [c for _, _, c in results]
    x = np.arange(len(names))
    width = 0.6

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5))
    (ax1, ax2), (ax3, ax4) = axes

    def baseline(ax):
        ax.axhline(0, color=COLOR_BASELINE, linestyle="--", linewidth=1.25, label="Baseline")

    # Fuel saved
    vals = [t.fuel_saved_kg for _, t, _ in results]
    baseline(ax1)
    b = ax1.bar(x, vals, width, color=colors)
    ax1.set_title("Fuel saved (kg)")
    ax1.set_xticks(x, names, rotation=10)
    ax1.grid(axis="y", alpha=0.25)
    _annotate_bars(ax1, b)
    ax1.legend(fontsize=9)

    # NOx saved
    vals = [t.nox_saved_kg for _, t, _ in results]
    baseline(ax2)
    b = ax2.bar(x, vals, width, color=colors)
    ax2.set_title("NOx saved (kg)")
    ax2.set_xticks(x, names, rotation=10)
    ax2.grid(axis="y", alpha=0.25)
    _annotate_bars(ax2, b, fmt="{:,.1f}")
    ax2.legend(fontsize=9)

    # CO₂ saved
    vals = [t.co2_saved_kg / 1000.0 for _, t, _ in results]
    baseline(ax3)
    b = ax3.bar(x, vals, width, color=colors)
    ax3.set_title("CO₂ saved (t)")
    ax3.set_xticks(x, names, rotation=10)
    ax3.grid(axis="y", alpha=0.25)
    _annotate_bars(ax3, b, fmt="{:,.1f}")
    ax3.legend(fontsize=9)

    # € saved
    vals = [t.total_value_eur for _, t, _ in results]
    baseline(ax4)
    b = ax4.bar(x, vals, width, color=colors)
    ax4.set_title("Total benefit (€)")
    ax4.set_xticks(x, names, rotation=10)
    ax4.grid(axis="y", alpha=0.25)
    _annotate_bars(ax4, b, fmt="€{:,.0f}")
    ax4.legend(fontsize=9)

    fig.suptitle(
        "N-1 Taxi — 1,200 flights | 2022 Average taxi times: in 6.0 min, out 12.3 min |",
        y=0.995
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])

    if save_path:
        plt.savefig(save_path, format="pdf", dpi=300, bbox_inches="tight")
        print(f"[INFO] Scenario figure saved → {save_path}")
    plt.show()

# --------------------- Roadmap plot ---------------------
def plot_roadmap_2025_2030(n_flights, save_path=None, cumulative=False):
    years = sorted(ETS_PRICE_BY_YEAR.keys())
    out_only, both = [], []
    for y in years:
        p = ETS_PRICE_BY_YEAR[y]
        out_only.append(_compute_scenario_totals(n_flights, 1.0, 0.0, p))
        both.append(_compute_scenario_totals(n_flights, 1.0, 1.0, p))

    def arr(sel):
        a = np.array([sel(t) for t in out_only])
        b = np.array([sel(t) for t in both])
        if cumulative:
            a, b = np.cumsum(a), np.cumsum(b)
        return a, b

    nox_out, nox_both = arr(lambda t: t.nox_saved_kg)
    co2_out, co2_both = arr(lambda t: t.co2_saved_kg / 1000.0)
    eur_out, eur_both = arr(lambda t: t.total_value_eur)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for ax in (ax1, ax2, ax3):
        ax.grid(True, alpha=0.25)

    ax1.plot(years, nox_out, marker="o", color=COLOR_OUT_ONLY, label="N-1 Out only")
    ax1.plot(years, nox_both, marker="o", color=COLOR_BOTH, label="N-1 Out & In")
    ax1.set_ylabel("NOx saved (kg)")
    ax1.legend()

    ax2.plot(years, co2_out, marker="o", color=COLOR_OUT_ONLY)
    ax2.plot(years, co2_both, marker="o", color=COLOR_BOTH)
    ax2.set_ylabel("CO₂ saved (t)")

    ax3.plot(years, eur_out, marker="o", color=COLOR_OUT_ONLY)
    ax3.plot(years, eur_both, marker="o", color=COLOR_BOTH)
    ax3.set_ylabel("Total benefit (€)")
    ax3.set_xlabel("Year")

    mode = "Cumulative totals" if cumulative else "Annual totals"
    fig.suptitle(
        f"N-1 Taxi roadmap to 2030 — {mode} for {n_flights:,} flights/year",
        y=0.99
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])

    if save_path:
        plt.savefig(save_path, format="pdf", dpi=300, bbox_inches="tight")
        print(f"[INFO] Roadmap figure saved → {save_path}")
    plt.show()

# --------------------- Main ----------------------------
if __name__ == "__main__":
    plot_n1_scenarios(N_FLIGHTS, save_path=SCENARIO_FIG_PATH)
    plot_roadmap_2025_2030(N_FLIGHTS, save_path=ROADMAP_FIG_PATH, cumulative=False)

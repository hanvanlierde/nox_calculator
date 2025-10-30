#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Aircraft performance regression — Stage Length vs DATA - FUEL BURN.

• x-axis: Stage Length
• y-axis: DATA - FUEL BURN
• Scatter: light gray
• Regression colors:
    32Q (NEO)  -> #0CD66D
    73H (737)  -> #2D099D
    320        -> #fc3968
• All plots saved as PDF.
• Font: Nexa-ExtraLight.ttf (if available)
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.ticker import PercentFormatter

# ---------------------- CONFIG ----------------------
PRIMARY_COLOR   = "#0CD66D"   # 32Q (NEO)
SECONDARY_COLOR = "#2D099D"   # 73H (737)
THIRD_COLOR     = "#fc3968"   # 320
POINT_GRAY      = "#C8C8C8"

TYPE_COLOR_MAP = {
    "32Q": PRIMARY_COLOR,
    "73H": SECONDARY_COLOR,
    "320": THIRD_COLOR,
}

DPI = 150
LINEWIDTH = 2.4
SCATTER_SIZE = 48
FILL_ALPHA = 0.18  # for filled areas

OUTPUT_PER_TYPE = "stage_vs_datafuel_{type_name}.pdf"
OUTPUT_ALL      = "stage_vs_datafuel_all_regressions.pdf"
OUTPUT_PCTDIFF  = "stage_vs_datafuel_pct_additional_vs_32Q.pdf"
OUTPUT_ABSDIFF  = "stage_vs_datafuel_abs_additional_vs_32Q.pdf"

# % / absolute diff range (km) for regression-based comparison
XMIN = 0.0
XMAX = 3000.0
NPOINTS = 601
EPS_DENOM = 1e-9
# ----------------------------------------------------


RAW_DATA = r"""
320
Fuel Burn (ltr)    Kilometers    Stage Length    Trips    DATA - FUEL BURN
252,043    48,937    1,359    36    7001.207153
186,957    40,240    1,437    28    6677.037471
451,873    90,366    1,458    62    7288.272653
174,208    37,607    1,709    22    7918.544582
503,867    107,590    1,735    62    8126.88536
355,425    77,006    1,833    42    8462.500998
151,347    33,003    1,833    18    8408.158828
143,974    32,488    2,321    14    10283.89014
282,286    62,261    2,395    26    10857.1396
22,595    4,823    2,411    2    11297.5
459,929    105,220    3,095    34    13527.3185

32Q
Fuel Burn (ltr)    Kilometers    Stage Length    Trips    DATA - FUEL BURN
2,000    367    367    1    2000
35,515    7,445    1,241    6    5919.198317
341,987    73,873    1,539    48    7124.735431
303,711    67,750    1,613    42    7231.208569
130,586    30,769    1,709    18    7254.759828
48,523    10,968    1,828    6    8087.130817
196,103    44,048    1,835    24    8170.968383
86,597    19,928    1,993    10    8659.69388
241,977    56,290    2,165    26    9306.820931
113,310    26,180    2,182    12    9442.470667
57,339    13,390    2,232    6    9556.47245
306,762    72,339    2,411    30    10225.40833
144,308    34,225    2,445    14    10307.7057
274,013    63,694    2,654    24    11417.19961
42,094    10,823    2,706    4    10523.41773
49,501    12,023    3,006    4    12375.30685

73H
Fuel Burn (ltr)    Kilometers    Stage Length    Trips    DATA - FUEL BURN
6,001    519    104    5    1200.25318
762,995    147,660    1,241    119    6411.721649
5,708    1,411    1,411    1    5707.5949
241,017    49,248    1,539    32    7531.790875
1,012,581    212,928    1,613    132    7671.064977
284,986    62,149    1,828    34    8381.943644
325,845    69,743    1,835    38    8574.861945
32,508    7,823    1,956    4    8126.898725
732,966    169,465    1,971    86    8522.865577
163,104    35,870    1,993    18    9061.318228
53,867    11,957    1,993    6    8977.8481
601,399    130,899    2,182    60    10023.31004
447,248    102,430    2,328    44    10164.73569
172,135    38,581    2,411    16    10758.46814
85,270    19,557    2,445    8    10658.70255
290,408    66,402    2,554    26    11169.5575
204,633    46,404    2,578    18    11368.52442
268,999    64,939    2,706    24    11208.28059
279,462    66,128    3,006    22    12702.81933
""".strip()


# ---------- Utility & plotting functions ----------
def load_custom_font(ttf_name="Nexa-ExtraLight.ttf"):
    p = Path(ttf_name)
    if p.exists():
        try:
            fm.fontManager.addfont(str(p))
            plt.rcParams["font.family"] = fm.FontProperties(fname=str(p)).get_name()
        except Exception:
            pass


def style_matplotlib():
    load_custom_font()
    plt.rcParams.update({
        "figure.dpi": DPI,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
    })


def parse_blocks(raw: str) -> pd.DataFrame:
    rows = []
    current_type = None
    header_re = re.compile(r"fuel\s*burn", re.IGNORECASE)
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        if header_re.search(line):
            continue
        if re.fullmatch(r"[A-Za-z0-9]+", line):
            current_type = line
            continue
        parts = re.split(r"[\s\t]+", line)
        if len(parts) < 5:
            continue
        try:
            stage = float(parts[2].replace(",", ""))
            data_fuel = float(parts[4].replace(",", ""))
        except ValueError:
            continue
        if current_type:
            rows.append((current_type, stage, data_fuel))
    return pd.DataFrame(rows, columns=["type", "stage_length", "data_fuel"])


def linfit(x, y):
    m, b = np.polyfit(x, y, 1)
    yhat = m * x + b
    r2 = 1 - np.sum((y - yhat) ** 2) / np.sum((y - np.mean(y)) ** 2)
    return m, b, r2


def type_color(t): return TYPE_COLOR_MAP.get(t, THIRD_COLOR)


def plot_one_type(t, sub):
    x = sub["stage_length"].to_numpy(float)
    y = sub["data_fuel"].to_numpy(float)
    m, b, r2 = linfit(x, y)
    xg = np.linspace(x.min(), x.max(), 200)
    yg = m * xg + b
    fig, ax = plt.subplots()
    ax.scatter(x, y, s=SCATTER_SIZE, color=POINT_GRAY, alpha=0.9)
    ax.plot(xg, yg, color=type_color(t), lw=LINEWIDTH,
            label=f"Regression: fuel = {m:.3f}·stage + {b:.1f} (R²={r2:.3f})")
    ax.set_title(f"Fuel Burn vs Stage Length — {t}")
    ax.set_xlabel("Stage Length (Km)")
    ax.set_ylabel("Fuel Burn (l)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_PER_TYPE.format(type_name=t))
    return m, b, r2


def plot_all_regressions(df, fits):
    xmin, xmax = df["stage_length"].min(), df["stage_length"].max()
    xg = np.linspace(xmin, xmax, 400)
    fig, ax = plt.subplots()
    for t in ["32Q", "73H", "320"]:
        if t in fits:
            m, b, r2 = fits[t]
            yg = m * xg + b
            ax.plot(xg, yg, color=type_color(t), lw=LINEWIDTH + 0.4,
                    label=f"{t} regression (R²={r2:.3f})")
    ax.set_title("Fuel Burn vs Stage Length — all regressions")
    ax.set_xlabel("Stage Length (Km)")
    ax.set_ylabel("Fuel Burn (l)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_ALL)


def plot_pct_additional(fits):
    """Plot % Additional Fuel Burn vs 32Q (regression-based)."""
    def yhat(m, b, x): return m * x + b
    xg = np.linspace(XMIN, XMAX, NPOINTS)

    if "32Q" not in fits:
        raise RuntimeError("32Q regression missing")
    m_q, b_q, _ = fits["32Q"]
    y_q = yhat(m_q, b_q, xg)
    denom = np.where(np.abs(y_q) < EPS_DENOM, np.sign(y_q)*EPS_DENOM, y_q)

    fig, ax = plt.subplots()
    # draw A32Q baseline
    ax.plot(xg, np.zeros_like(xg), "--", color=PRIMARY_COLOR, lw=2, label="32Q baseline")

    if "320" in fits:
        m, b, _ = fits["320"]
        diff = (yhat(m, b, xg) - y_q) / denom * 100
        ax.plot(xg, diff, color=THIRD_COLOR, lw=LINEWIDTH, label="320 additional fuel burn")
        ax.fill_between(xg, 0, diff, color=THIRD_COLOR, alpha=FILL_ALPHA)

    if "73H" in fits:
        m, b, _ = fits["73H"]
        diff = (yhat(m, b, xg) - y_q) / denom * 100
        ax.plot(xg, diff, color=SECONDARY_COLOR, lw=LINEWIDTH, label="73H additional fuel burn")
        ax.fill_between(xg, 0, diff, color=SECONDARY_COLOR, alpha=FILL_ALPHA)

    ax.set_title("Fuel Burn % Additional vs 32Q (regression-based)")
    ax.set_xlabel("Stage Length (Km)")
    ax.set_ylabel("% Additional Fuel Burn")
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.axhline(0, color="#999999", lw=1.0, alpha=0.6)
    ax.set_xlim(XMIN, XMAX)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_PCTDIFF)
    print(f"Saved: {OUTPUT_PCTDIFF}")


def plot_abs_additional(fits):
    """Plot absolute Additional Fuel Burn vs 32Q (regression-based)."""
    def yhat(m, b, x): return m * x + b
    xg = np.linspace(XMIN, XMAX, NPOINTS)

    if "32Q" not in fits:
        raise RuntimeError("32Q regression missing")
    m_q, b_q, _ = fits["32Q"]
    y_q = yhat(m_q, b_q, xg)

    fig, ax = plt.subplots()
    # draw A32Q baseline
    ax.plot(xg, np.zeros_like(xg), "--", color=PRIMARY_COLOR, lw=2, label="32Q baseline")

    if "320" in fits:
        m, b, _ = fits["320"]
        diff = yhat(m, b, xg) - y_q
        ax.plot(xg, diff, color=THIRD_COLOR, lw=LINEWIDTH, label="320 additional fuel burn")
        ax.fill_between(xg, 0, diff, color=THIRD_COLOR, alpha=FILL_ALPHA)

    if "73H" in fits:
        m, b, _ = fits["73H"]
        diff = yhat(m, b, xg) - y_q
        ax.plot(xg, diff, color=SECONDARY_COLOR, lw=LINEWIDTH, label="73H additional fuel burn")
        ax.fill_between(xg, 0, diff, color=SECONDARY_COLOR, alpha=FILL_ALPHA)

    ax.set_title("Fuel Burn Additional vs 32Q (regression-based)")
    ax.set_xlabel("Stage Length (Km)")
    ax.set_ylabel("Additional Fuel Burn (l)")
    ax.axhline(0, color="#999999", lw=1.0, alpha=0.6)
    ax.set_xlim(XMIN, XMAX)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_ABSDIFF)
    print(f"Saved: {OUTPUT_ABSDIFF}")

def plot_all_scatter(df: pd.DataFrame):
    """Plot all aircraft data points in one scatter plot, colored by type."""
    fig, ax = plt.subplots()
    for t in ["32Q", "73H", "320"]:
        sub = df[df["type"] == t]
        ax.scatter(
            sub["stage_length"], sub["data_fuel"],
            s=SCATTER_SIZE, color=type_color(t), alpha=0.8, label=t
        )
    ax.set_title("Fuel Burn vs Stage Length — All Aircraft Data")
    ax.set_xlabel("Stage Length (Km)")
    ax.set_ylabel("Fuel Burn (l)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("stage_vs_datafuel_all_scatter.pdf")
    print("Saved: stage_vs_datafuel_all_scatter.pdf")

def plot_savings_errorbars_vs_32Q(df: pd.DataFrame, fits: Dict[str, Tuple[float, float, float]]):
    """
    For every non-32Q flight, draw a vertical 'savings' bar equal to:
        y_hat(other, stage_length) - y_hat(32Q, stage_length)
    Plot only non-32Q flights as scatter points, and make all savings
    bars green to indicate potential improvement if flown by 32Q.
    """
    def yhat(m, b, x): return m * x + b

    if "32Q" not in fits:
        raise RuntimeError("32Q regression not available for savings plot.")
    m_q, b_q, _ = fits["32Q"]

    fig, ax = plt.subplots()

    # Scatter only non-32Q flights
    for t in ["73H", "320"]:
        sub = df[df["type"] == t]
        ax.scatter(
            sub["stage_length"], sub["data_fuel"],
            s=SCATTER_SIZE, color=type_color(t),
            alpha=0.9, label=f"{t} data"
        )

    # Draw green "savings" bars
    for t in ["320", "73H"]:
        if t not in fits:
            continue
        m_o, b_o, _ = fits[t]
        sub = df[df["type"] == t]
        if sub.empty:
            continue

        x = sub["stage_length"].to_numpy(float)
        y_other = yhat(m_o, b_o, x)
        y_neo   = yhat(m_q, b_q, x)
        savings = y_other - y_neo

        mask = savings > 0
        x_draw = x[mask]
        y_bottom = y_neo[mask]
        y_top = y_other[mask]

        if x_draw.size:
            # Green vertical "savings" bars
            ax.vlines(
                x_draw, y_bottom, y_top,
                color=PRIMARY_COLOR, linewidth=2.4, alpha=0.85
            )

    # Styling and labels
    ax.set_title("Fuel Burn — Potential Savings vs 32Q (regression-based)")
    ax.set_xlabel("Stage Length (Km)")
    ax.set_ylabel("Fuel Burn (l)")
    ax.set_xlim(df["stage_length"].min(), df["stage_length"].max())

    # Custom legend
    from matplotlib.lines import Line2D
    legend_items = [
        Line2D([0], [0], color=PRIMARY_COLOR, lw=2.4, label="Potential savings (if flown by 32Q)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=THIRD_COLOR,
               label="320 actual data", markersize=7),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=SECONDARY_COLOR,
               label="73H actual data", markersize=7),
    ]
    ax.legend(handles=legend_items, loc="upper left", frameon=True)

    fig.tight_layout()
    fig.savefig("stage_vs_datafuel_savings_errorbars_vs_32Q.pdf")
    print("Saved: stage_vs_datafuel_savings_errorbars_vs_32Q.pdf")

def plot_non32Q_scatter_with_32Q_regression(df: pd.DataFrame, fits: Dict[str, Tuple[float, float, float]]):
    """
    Plot A320 and 73H scatter points with ONLY the A32Q regression line overlaid.
    Saves: stage_vs_datafuel_scatter_non32Q_with_32Q_reg.pdf
    """
    def yhat(m, b, x): return m * x + b

    if "32Q" not in fits:
        raise RuntimeError("32Q regression not available for overlay plot.")
    m_q, b_q, _ = fits["32Q"]

    fig, ax = plt.subplots()

    # Scatter for A320 and 73H only
    for t in ["320", "73H"]:
        sub = df[df["type"] == t]
        if sub.empty:
            continue
        ax.scatter(
            sub["stage_length"], sub["data_fuel"],
            s=SCATTER_SIZE, color=type_color(t), alpha=0.9, label=f"{t} data"
        )

    # A32Q regression line across full domain of observed stages
    xmin, xmax = df["stage_length"].min(), df["stage_length"].max()
    xgrid = np.linspace(xmin, xmax, 600)
    ygrid_32q = yhat(m_q, b_q, xgrid)
    ax.plot(xgrid, ygrid_32q, linestyle="--", linewidth=LINEWIDTH+0.2,
            color=PRIMARY_COLOR, label="32Q regression")

    # Titles & styling (kept consistent with your latest)
    ax.set_title("Fuel Burn vs Stage Length — A320 & 73H with 32Q regression")
    ax.set_xlabel("Stage Length (Km)")
    ax.set_ylabel("Fuel Burn (l)")
    ax.legend()
    ax.set_xlim(xmin, xmax)
    fig.tight_layout()
    fig.savefig("stage_vs_datafuel_scatter_non32Q_with_32Q_reg.pdf")
    print("Saved: stage_vs_datafuel_scatter_non32Q_with_32Q_reg.pdf")


def main():
    style_matplotlib()
    df = parse_blocks(RAW_DATA)
    fits = {t: plot_one_type(t, sub) for t, sub in df.groupby("type")}
    plot_all_regressions(df, fits)
    plot_all_scatter(df)
    plot_pct_additional(fits)
    plot_abs_additional(fits)
    plot_savings_errorbars_vs_32Q(df, fits)
    plot_non32Q_scatter_with_32Q_regression(df, fits)  # ← NEW
    plt.show()





if __name__ == "__main__":
    main()

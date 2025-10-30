# === B738 Blended Winglet (BW) vs Shortened/Scimitar-Style Winglet (SSW) Comparison ===
# Adds Normal range distribution (μ=1325.47 km, σ=640.22 km) by trimming each generated flight
# at its target distance. OpenAP models remain unchanged; SSW uses a simple fuel-flow scale.

from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from openap.gen import FlightGenerator
import openap

PARAMS = {
    # Kinematic flight generation (used once; emissions computed per-variant on same kinematics)
    "dt": 100,                  # time step [s]
    "n_flights": 1108,          # number of sample flights
    "random_init": True,        # False avoids degenerate on-ground profiles
    "enable_noise": True,       # add generator noise
    "gen_aircraft": "b738",     # aircraft used ONLY for generating kinematics

    # Mass settings (per-aircraft code; both variants use B738 under the hood)
    "mass_takeoff": 65_000.0,   # fallback if not in mass_takeoff_by_ac
    "mass_takeoff_by_ac": {     # keys are aircraft codes (case-insensitive)
        "B738": 65_000.0,
        "A21N": 70_000.0,
    },

    # Fuel-flow reduction for SSW (percent). Positive means SSW burns that much less fuel.
    "ff_reduction_pct_ssw": 1.74,

    # Emission factors
    "ef_co2_kg_per_kg": 3.16,   # CO2 factor [kg CO2 / kg fuel]

    # Plotting & branding (plots only used when __main__)
    "mpl_backend": "TkAgg",
    "font_family": "Nexa",                     # fallback family name
    "font_directory": "Nexa-ExtraLight.ttf",   # path to your Nexa TTF file
    "font_size": 11,
    "grid_color": "darkgray",
    "grid_linestyle": ":",
    "plot_lw": 0.2,
    "figsize": (8, 4),

    # Brand colors
    "primary_color": "0cd66d",   # B738-BW
    "secondary_color": "2d099d", # B738-SSW

    # Extra plots (toggle) — only used when running this file directly
    "plot_fuel": False,
    "plot_mass": False,
    "plot_nox":  False,
    "plot_co2":  False,
    "fuel_figsize": (6, 6),

    # Histogram export (used only if plotting)
    "save_hist": False,
    "hist_filename": "range_histogram",
    "hist_format": "pdf",
    "hist_dpi": 300,
    "hist_bins": 40,
    "show_normal_pdf": True,

    # Guardrails for calling the models
    "vs_fpm_clip": 6000,       # clip vertical rate to +/- this [ft/min]
    "tas_min_flight": 80.0,    # kt; below -> on ground
    "alt_min_flight": 100.0,   # ft; below -> on ground

    # Saving (when __main__)
    "save_csv": False,
    "csv_filename": "flights_combined_compare.csv",
}

# ---------------- Font configuration ----------------
def configure_fonts(params: dict):
    mpl.rc("grid", color=params["grid_color"], linestyle=params["grid_linestyle"])
    mpl.rcParams["font.size"] = params["font_size"]
    font_path = params.get("font_directory")
    fallback_family = params.get("font_family", "sans-serif")
    if not font_path:
        mpl.rcParams["font.family"] = fallback_family
        return
    try:
        fm.fontManager.addfont(font_path)
        family_name = fm.FontProperties(fname=font_path).get_name()
        mpl.rcParams["font.family"] = family_name
    except Exception:
        mpl.rcParams["font.family"] = fallback_family

# Silence RuntimeWarnings thrown inside openap.fuel for extreme inputs
warnings.filterwarnings("ignore", category=RuntimeWarning, module=r".*openap\.fuel")

# ---------------- Variant builder (exported) ----------------
def build_b738_variants(params: dict) -> list[dict]:
    """Two B738 variants sharing OpenAP models; SSW gets a fuel-flow multiplier."""
    scale_ssw = 1.0 - float(params["ff_reduction_pct_ssw"]) / 100.0  # e.g., 0.9826
    variants = [
        {"label": "B738-BW",  "ac_code": "B738", "ff_scale": 1.0},
        {"label": "B738-SSW", "ac_code": "B738", "ff_scale": scale_ssw},
    ]
    return variants

VARIANTS = build_b738_variants(PARAMS)  # <-- exported symbol used by your timeline

# ---------------- Colors ----------------
def _hex(c: str) -> str:
    return c if c.startswith("#") else f"#{c}"

def color_for_aircraft(label: str, params: dict) -> str:
    return _hex(params["secondary_color"]) if "SSW" in label.upper() else _hex(params["primary_color"])

# ---------------- Range-distribution helpers (exported) ----------------
def _sample_targets_normal_km(n, mean_km, sd_km, min_km=80.0, max_km=6000.0):
    """Draw n positive targets from N(mean, sd) with truncation to [min_km, max_km]."""
    targets = []
    need = n
    while need > 0:
        x = np.random.normal(loc=mean_km, scale=sd_km, size=max(need * 2, 100))
        x = x[(x >= min_km) & (x <= max_km)]
        if len(x) == 0:
            continue
        take = min(need, len(x))
        targets.extend(x[:take])
        need -= take
    return np.array(targets, dtype=float)

def _trim_to_distance_m(df: pd.DataFrame, target_m: float) -> pd.DataFrame:
    """Trim a single generated flight to stop exactly at target_m (linear interpolate last step)."""
    s = df["s"].to_numpy()
    if s[-1] <= target_m:
        return df  # nothing to trim; caller should ensure s_end >= target_m
    idx = np.searchsorted(s, target_m, side="left")
    if idx == 0:
        return df.iloc[[0]].copy()
    if s[idx] == target_m:
        return df.iloc[: idx + 1].copy()
    a = df.iloc[idx - 1]; b = df.iloc[idx]
    w = (target_m - a.s) / (b.s - a.s + 1e-12)
    new = b.copy()
    for col in ("t", "h", "s", "altitude", "vertical_rate", "groundspeed"):
        if col in df.columns:
            new[col] = float(a[col] + w * (b[col] - a[col]))
    out = pd.concat([df.iloc[:idx], new.to_frame().T], ignore_index=True)
    return out

def generate_flights_with_range_distribution(
    params: dict,
    mean_km: float = 1325.47,
    sd_km: float = 640.22,
    min_km: float = 80.0,
    max_km: float = 6000.0,
    max_tries_per_flight: int = 40,
):
    """
    Generate kinematic flights whose total distances follow a Normal(mean, sd), truncated.
    Strategy: draw target distances (km), generate complete flights, and trim at target s.
    If a generated flight ends shorter than target, re-generate up to max_tries_per_flight.
    Returns (flights, targets_km).
    """
    fgen = FlightGenerator(ac=params["gen_aircraft"])
    if params.get("enable_noise", False):
        fgen.enable_noise()
    targets_km = _sample_targets_normal_km(params["n_flights"], mean_km, sd_km, min_km, max_km)
    flights = []
    for tk in targets_km:
        target_m = float(tk) * 1000.0
        last = None
        for _ in range(max_tries_per_flight):
            f = fgen.complete(dt=params["dt"], random=params["random_init"])
            last = f
            if f["s"].iloc[-1] >= target_m:
                f_trim = _trim_to_distance_m(f, target_m)
                flights.append(f_trim)
                break
        else:
            flights.append(last)  # fallback
    return flights, targets_km

# ---------------- Core computation: fuel, NOx & CO2 (exported) ----------------
def compute_fuel_and_nox_multi(
    flights: list[pd.DataFrame],
    variants: list[dict],                 # list with 'label', 'ac_code', 'ff_scale'
    mass_takeoff_default: float,
    mass_takeoff_by_ac: dict,
    dt: int,
    vs_clip_fpm: float | None = None,
    tas_min_flight: float = 80.0,
    alt_min_flight: float = 100.0,
    ef_co2_kg_per_kg: float = PARAMS["ef_co2_kg_per_kg"],
) -> pd.DataFrame:
    """
    Returns long-form DataFrame with:
      flight_id, aircraft (variant label), mass0_kg, t, altitude, groundspeed, vertical_rate,
      fuel_flow (kg/s), fuel_step (kg), mass_after_step (kg),
      nox_flow_gps (g/s), nox_step_g (g), nox_cum_g (g),
      co2_flow_kgps (kg/s), co2_step_kg (kg), co2_cum_kg (kg)
    """
    all_rows = []

    mass_map = {str(k).upper(): float(v) for k, v in (mass_takeoff_by_ac or {}).items()}

    fuel_models, emis_models = {}, {}
    ac_codes = {v["ac_code"].upper() for v in variants}
    for ac_code in ac_codes:
        try:
            fuel_models[ac_code] = openap.FuelFlow(ac_code, use_synonym=True)
            emis_models[ac_code]  = openap.Emission(ac_code, use_synonym=True)
        except Exception:
            pass

    if not fuel_models:
        raise RuntimeError("No aircraft models could be initialized. Check codes/synonyms.")

    for flight_id, f in enumerate(flights, start=1):
        t_arr   = f["t"].to_numpy()
        gs_arr  = f["groundspeed"].to_numpy(dtype=float)   # kt
        alt_arr = f["altitude"].to_numpy(dtype=float)      # ft
        vs_arr  = f["vertical_rate"].to_numpy(dtype=float) # ft/min
        s_arr   = f["s"].to_numpy(dtype=float) if "s" in f.columns else None
        h_arr   = f["h"].to_numpy(dtype=float) if "h" in f.columns else None

        if vs_clip_fpm is not None:
            vs_arr = np.clip(vs_arr, -vs_clip_fpm, vs_clip_fpm)

        in_flight_mask = (gs_arr >= tas_min_flight) & (alt_arr >= alt_min_flight)

        for var in variants:
            label = var["label"]
            ac    = var["ac_code"].upper()
            scale = float(var.get("ff_scale", 1.0))
            if ac not in fuel_models:
                continue

            mass0 = mass_map.get(ac, mass_takeoff_default)
            mass  = float(mass0)

            ff_model = fuel_models[ac]
            em_model = emis_models[ac]

            ff = np.zeros_like(gs_arr, dtype=float)         # scaled fuel flow (kg/s)
            nox_flow = np.zeros_like(gs_arr, dtype=float)   # g/s
            fuel_step = np.zeros_like(gs_arr, dtype=float)  # kg
            mass_series = np.zeros_like(gs_arr, dtype=float)
            nox_step = np.zeros_like(gs_arr, dtype=float)   # g

            co2_flow = np.zeros_like(gs_arr, dtype=float)   # kg/s
            co2_step = np.zeros_like(gs_arr, dtype=float)   # kg

            for i in range(len(gs_arr)):
                if not in_flight_mask[i] or mass <= 0.0:
                    ff_raw = 0.0
                else:
                    tas_kt = float(np.clip(gs_arr[i], 50.0, 520.0))
                    alt_ft = float(np.clip(alt_arr[i], 0.0, 45000.0))
                    vs_fpm = float(np.clip(vs_arr[i], -8000.0, 8000.0))

                    val = ff_model.enroute(mass=mass, tas=tas_kt, alt=alt_ft, vs=vs_fpm)
                    ff_raw = val if np.isfinite(val) and val >= 0 else 0.0

                ff[i] = scale * ff_raw  # <-- 1.74% reduction for SSW fuel flow

                nv = em_model.nox(ff[i], tas=float(gs_arr[i]), alt=float(alt_arr[i]))
                nox_flow[i] = nv if np.isfinite(nv) and nv >= 0 else 0.0

                fuel_step[i] = ff[i] * dt
                nox_step[i]  = nox_flow[i] * dt
                mass = max(mass - fuel_step[i], 0.0)
                mass_series[i] = mass

                co2_flow[i] = ff[i] * ef_co2_kg_per_kg          # kg/s
                co2_step[i] = fuel_step[i] * ef_co2_kg_per_kg   # kg

            nox_cum = np.cumsum(nox_step)
            co2_cum = np.cumsum(co2_step)

            df_var = pd.DataFrame({
                "flight_id": flight_id,
                "aircraft": label,           # labels: B738-BW vs B738-SSW (or A21N if passed)
                "mass0_kg": mass0,
                "t": t_arr,
                "altitude": alt_arr,
                "groundspeed": gs_arr,
                "vertical_rate": vs_arr,
                "fuel_flow": ff,
                "fuel_step": fuel_step,
                "mass_after_step": mass_series,
                "nox_flow_gps": nox_flow,
                "nox_step_g": nox_step,
                "nox_cum_g": nox_cum,
                "co2_flow_kgps": co2_flow,
                "co2_step_kg": co2_step,
                "co2_cum_kg": co2_cum,
                "ff_scale": scale,
            })
            if s_arr is not None: df_var["s"] = s_arr
            if h_arr is not None: df_var["h"] = h_arr

            all_rows.append(df_var)

    return pd.concat(all_rows, ignore_index=True)

# ===========================
# Optional plotting if run directly (kept minimal)
# ===========================
if __name__ == "__main__":
    mpl.use(PARAMS["mpl_backend"])
    configure_fonts(PARAMS)

    # Quick smoke test (no heavy plotting)
    flights_kin, _ = generate_flights_with_range_distribution(
        PARAMS, mean_km=1325.47, sd_km=640.22, min_km=80.0, max_km=6000.0
    )
    combined = compute_fuel_and_nox_multi(
        flights=flights_kin,
        variants=VARIANTS,
        mass_takeoff_default=PARAMS["mass_takeoff"],
        mass_takeoff_by_ac=PARAMS.get("mass_takeoff_by_ac", {}),
        dt=PARAMS["dt"],
        vs_clip_fpm=PARAMS["vs_fpm_clip"],
        tas_min_flight=PARAMS["tas_min_flight"],
        alt_min_flight=PARAMS["alt_min_flight"],
        ef_co2_kg_per_kg=PARAMS["ef_co2_kg_per_kg"],
    )
    print(combined.groupby("aircraft")["nox_step_g"].sum() / 1000.0)

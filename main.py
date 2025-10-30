# Improved main.py — now uses the Normal-distribution flight generator with trimming
# (μ = 1325.47 km, σ = 640.22 km), matching the 737 blended script.
# Plotting remains OFF; exposes simulate_per_model_totals() for import.

PARAMS = {
    # Kinematics generation
    "dt": 10,
    "n_flights": 100,             # per-aircraft flights used by simulate_per_model_totals()
    "random_init": True,
    "enable_noise": True,
    "gen_aircraft": "b738",

    # Aircraft models to compare
    "aircraft_models": ["b738", "a21n"],

    # Masses
    "mass_takeoff": 65_000.0,
    "mass_takeoff_by_ac": {"B738": 65_000.0, "A21N": 70_000.0},

    # Range distribution (NEW – same as in 737 blended script)
    "range_mean_km": 1325.47,
    "range_sd_km": 640.22,
    "range_min_km": 80.0,
    "range_max_km": 6000.0,
    "max_tries_per_flight": 40,

    # Matplotlib (kept quiet)
    "mpl_backend": "TkAgg",
    "font_family": "Helvetica",
    "font_size": 11,
    "grid_color": "darkgray",
    "grid_linestyle": ":",

    # Plot toggles (OFF)
    "plot_lw": 0.2,
    "figsize": (8, 4),
    "plot_fuel": False,
    "plot_mass": False,
    "plot_nox": False,
    "fuel_figsize": (6, 6),

    # Model guards
    "vs_fpm_clip": 6000,
    "tas_min_flight": 80.0,
    "alt_min_flight": 100.0,

    # CSV (OFF by default)
    "save_csv": False,
    "csv_filename": "flights_combined_compare.csv",
}

import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use(PARAMS["mpl_backend"])
import matplotlib.pyplot as plt  # (not used when imported; safe)
from openap.gen import FlightGenerator
import openap

# Styling (quiet)
matplotlib.rc("font", size=PARAMS["font_size"], family=PARAMS["font_family"])
matplotlib.rc("grid", color=PARAMS["grid_color"], linestyle=PARAMS["grid_linestyle"])
warnings.filterwarnings("ignore", category=RuntimeWarning, module=r".*openap\.fuel")

# ---------- Range-distribution helpers (copied from 737 blended script) ----------
def _sample_targets_normal_km(n, mean_km, sd_km, min_km=80.0, max_km=6000.0):
    """Draw n targets from N(mean, sd), truncated to [min_km, max_km]."""
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
    """Trim a generated flight to stop exactly at target_m (linear interpolation on last step)."""
    s = df["s"].to_numpy()
    if s[-1] <= target_m:
        return df
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

def generate_flights_with_range_distribution(params: dict) -> list[pd.DataFrame]:
    """
    Generate kinematic flights whose total distances follow a Normal(mean, sd), truncated.
    Strategy: draw target distances (km), generate complete flights, and trim at target s.
    """
    fgen = FlightGenerator(ac=params["gen_aircraft"])
    if params.get("enable_noise", False):
        fgen.enable_noise()
    targets_km = _sample_targets_normal_km(
        params["n_flights"],
        params["range_mean_km"], params["range_sd_km"],
        params["range_min_km"], params["range_max_km"],
    )
    flights = []
    for tk in targets_km:
        target_m = float(tk) * 1000.0
        last = None
        for _ in range(params.get("max_tries_per_flight", 40)):
            f = fgen.complete(dt=params["dt"], random=params["random_init"])
            last = f
            if f["s"].iloc[-1] >= target_m:
                flights.append(_trim_to_distance_m(f, target_m))
                break
        else:
            flights.append(last)  # fallback if we couldn’t reach target
    return flights

# ---------- Fuel & NOx computation (unchanged) ----------
def compute_fuel_and_nox_multi(
    flights: list[pd.DataFrame],
    aircraft_models: list[str],
    mass_takeoff_default: float,
    mass_takeoff_by_ac: dict,
    dt: int,
    vs_clip_fpm: float | None = None,
    tas_min_flight: float = 80.0,
    alt_min_flight: float = 100.0,
) -> pd.DataFrame:
    all_rows = []
    mass_map = {str(k).upper(): float(v) for k, v in (mass_takeoff_by_ac or {}).items()}
    fuel_models = {}; emis_models = {}; resolved_aircraft = []
    for ac in aircraft_models:
        ac_code = str(ac).upper()
        fuel_models[ac_code] = openap.FuelFlow(ac_code, use_synonym=True)
        emis_models[ac_code]  = openap.Emission(ac_code, use_synonym=True)
        resolved_aircraft.append(ac_code)

    for flight_id, f in enumerate(flights, start=1):
        t_arr   = f["t"].to_numpy()
        gs_arr  = f["groundspeed"].to_numpy(dtype=float)
        alt_arr = f["altitude"].to_numpy(dtype=float)
        vs_arr  = f["vertical_rate"].to_numpy(dtype=float)
        if vs_clip_fpm is not None:
            vs_arr = np.clip(vs_arr, -vs_clip_fpm, vs_clip_fpm)
        in_flight_mask = (gs_arr >= tas_min_flight) & (alt_arr >= alt_min_flight)

        for ac in resolved_aircraft:
            mass0 = mass_map.get(ac, mass_takeoff_default)
            mass = float(mass0)
            ff = np.zeros_like(gs_arr, dtype=float)
            nox_flow = np.zeros_like(gs_arr, dtype=float)
            fuel_step = np.zeros_like(gs_arr, dtype=float)
            nox_step = np.zeros_like(gs_arr, dtype=float)

            ff_model = fuel_models[ac]
            em_model = emis_models[ac]

            for i in range(len(gs_arr)):
                if not in_flight_mask[i] or mass <= 0.0:
                    ff[i] = 0.0
                    nox_flow[i] = 0.0
                else:
                    val = ff_model.enroute(mass=mass, tas=float(gs_arr[i]), alt=float(alt_arr[i]), vs=float(vs_arr[i]))
                    ff[i] = val if np.isfinite(val) and val >= 0 else 0.0
                    nv = em_model.nox(ff[i], tas=float(gs_arr[i]), alt=float(alt_arr[i]))
                    nox_flow[i] = nv if np.isfinite(nv) and nv >= 0 else 0.0

                fuel_step[i] = ff[i] * dt
                nox_step[i]  = nox_flow[i] * dt
                mass = max(mass - fuel_step[i], 0.0)

            nox_cum = np.cumsum(nox_step)
            df_ac = pd.DataFrame({
                "flight_id": flight_id,
                "aircraft": ac,
                "t": t_arr,
                "fuel_step": fuel_step,
                "nox_step_g": nox_step,
                "nox_cum_g": nox_cum,
            })
            all_rows.append(df_ac)

    return pd.concat(all_rows, ignore_index=True)

# ---------- Public helper for orchestrator ----------
def simulate_per_model_totals(
    n_flights: int = PARAMS["n_flights"],
    params: dict = PARAMS,
) -> pd.DataFrame:
    """
    Returns a compact DataFrame with per-model totals for the requested number of flights:
      columns: aircraft, total_fuel_kg, total_nox_g
    Uses the Normal-distribution range generator with trimming (as in 737 blended script).
    """
    local = dict(params)
    local["n_flights"] = n_flights
    flights = generate_flights_with_range_distribution(local)
    df = compute_fuel_and_nox_multi(
        flights=flights,
        aircraft_models=local["aircraft_models"],
        mass_takeoff_default=local["mass_takeoff"],
        mass_takeoff_by_ac=local.get("mass_takeoff_by_ac", {}),
        dt=local["dt"],
        vs_clip_fpm=local["vs_fpm_clip"],
        tas_min_flight=local["tas_min_flight"],
        alt_min_flight=local["alt_min_flight"],
    )
    tot = df.groupby("aircraft", as_index=False).agg(
        total_fuel_kg=("fuel_step", "sum"),
        total_nox_g=("nox_step_g", "sum"),
    )
    return tot

if __name__ == "__main__":
    # Quick dry run (no plots)
    print(simulate_per_model_totals())

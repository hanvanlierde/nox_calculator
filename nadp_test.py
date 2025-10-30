import numpy as np
import pandas as pd
from openap.gen import FlightGenerator
from openap import FuelFlow, Emission
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# style
matplotlib.rc("font", size=11)
matplotlib.rc("font", family="Helvetica")
matplotlib.rc("grid", color="darkgray", linestyle=":")

def enforce_nadp_speeds(df, *, v2p=170, acc_alt_ft=3000,
                         target_below10k=250, ramp_s=90):
    """
    Return a copy with groundspeed adjusted to emulate a NADP schedule:
      - hold v2p (kt) until acc_alt_ft
      - linearly ramp to target_below10k (kt) over ramp_s seconds
    """
    g = df.copy()
    t = g.t.to_numpy()
    alt = g.altitude.to_numpy()
    gs = g.groundspeed.to_numpy().copy()

    # index where we reach acceleration altitude
    i_acc = np.argmax(alt >= acc_alt_ft)
    if alt[i_acc] < acc_alt_ft and i_acc == len(alt)-1:
        i_acc = len(alt)-1

    # up to accel: cap at V2+
    gs[: i_acc + 1] = np.minimum(gs[: i_acc + 1], v2p)

    # ramp to target
    if i_acc < len(gs) - 1:
        t0, t1 = t[i_acc], t[i_acc] + ramp_s
        ramp_mask = (t >= t0) & (t <= t1)
        alpha = (t[ramp_mask] - t0) / max(1.0, (t1 - t0))
        gs[ramp_mask] = (1 - alpha) * v2p + alpha * target_below10k
        gs[t > t1] = np.maximum(gs[t > t1], target_below10k)

    g.groundspeed = gs
    g["nadp_acc_alt_ft"] = acc_alt_ft
    g["nadp_v2p_kt"] = v2p
    return g

def simulate_emissions(df, *, ac="B738", mass0_kg=68000):
    """
    Step through the profile, compute fuel flow (kg/s) and NOx flow (g/s),
    and integrate totals (kg). Assumes groundspeed≈TAS (no wind).
    """
    fuelflow = FuelFlow(ac=ac)
    emission = Emission(ac=ac)

    d = df.copy()
    d["d_ts"] = d["t"].diff().fillna(d["t"].iloc[0] if len(d) else 0)

    mass = mass0_kg
    FF, NOX = [], []
    for _, r in d.iterrows():
        ff = fuelflow.enroute(mass=mass, tas=r.groundspeed,
                              alt=r.altitude, vs=r.vertical_rate)
        FF.append(ff)  # kg/s
        NOX.append(emission.nox(ff, tas=r.groundspeed, alt=r.altitude))  # g/s
        mass -= ff * r.d_ts

    d["fuel_flow"] = FF
    d["nox_flow"] = NOX
    d["fuel"] = d["fuel_flow"] * d["d_ts"]                # kg
    d["nox"] = d["nox_flow"] * d["d_ts"] / 1000.0         # kg
    d["nox_cum"] = d["nox"].cumsum()
    return d

# -------- Generate and compare NADP 1 vs NADP 2 --------
fgen = FlightGenerator(ac="b738")

# Simulate climb to 10,000 ft (focus on departure segment)
dt = 2
base = fgen.climb(dt=dt, alt_cr=10000)  # CAS/Mach params irrelevant below 10k
# (OpenAP climb generation: see handbook 5.3 for params; emissions in 6.2.)  # ref

# Assumptions for a typical B738; change per weight/SOP
V2_PLUS = 170   # kt (≈ V2+15..20 for a mid-weight B738)
TARGET_10K = 250  # kt
MASS0 = 68000  # kg, assumed TOW for this example

# NADP 1: accelerate at ~3000 ft AFE
nadp1 = enforce_nadp_speeds(base, v2p=V2_PLUS, acc_alt_ft=3000,
                            target_below10k=TARGET_10K, ramp_s=90)
# NADP 2: accelerate early (~1000 ft AFE)
nadp2 = enforce_nadp_speeds(base, v2p=V2_PLUS, acc_alt_ft=1000,
                            target_below10k=TARGET_10K, ramp_s=90)

# Emissions (fuel + NOx)
nadp1_e = simulate_emissions(nadp1, ac="B738", mass0_kg=MASS0)
nadp2_e = simulate_emissions(nadp2, ac="B738", mass0_kg=MASS0)

# -------- Plots --------
def format_ax(ax):
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.grid(True)

# Altitude vs time
fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(nadp1_e.t, nadp1_e.altitude, label="NADP 1")
ax.plot(nadp2_e.t, nadp2_e.altitude, label="NADP 2")
ax.set_ylabel("Altitude (ft)"); ax.set_xlabel("Time (s)")
ax.legend(); format_ax(ax); fig.tight_layout()

# Speed vs time
fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(nadp1_e.t, nadp1_e.groundspeed, label="NADP 1")
ax.plot(nadp2_e.t, nadp2_e.groundspeed, label="NADP 2")
ax.set_ylabel("Speed (kt)"); ax.set_xlabel("Time (s)")
ax.legend(); format_ax(ax); fig.tight_layout()

# NOx flow (g/s) vs time
fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(nadp1_e.t, nadp1_e.nox_flow, label="NADP 1")
ax.plot(nadp2_e.t, nadp2_e.nox_flow, label="NADP 2")
ax.set_ylabel("NOx (g/s)"); ax.set_xlabel("Time (s)")
ax.legend(); format_ax(ax); fig.tight_layout()

# Cumulative NOx (kg) vs time
fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(nadp1_e.t, nadp1_e.nox_cum, label="NADP 1")
ax.plot(nadp2_e.t, nadp2_e.nox_cum, label="NADP 2")
ax.set_ylabel("Cumulative NOx (kg)"); ax.set_xlabel("Time (s)")
ax.legend(); format_ax(ax); fig.tight_layout()

print("Total NOx to 10,000 ft:")
print(f"  NADP 1: {nadp1_e.nox.sum():.3f} kg")
print(f"  NADP 2: {nadp2_e.nox.sum():.3f} kg")

plt.show()

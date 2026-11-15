import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# IONA | IONOSPHERIC GNSS PROPAGATION MODEL
# First-order group delay caused by free electrons
# ============================================================

C = 299792458.0
F_L1 = 1575.42e6
F_L2 = 1227.60e6
K_IONO = 40.3

EARTH_RADIUS_KM = 6371.0
SHELL_HEIGHT_KM = 350.0
ELEVATION_DEG = 30.0

OUTPUT = Path("results/ionosphere")
OUTPUT.mkdir(parents=True, exist_ok=True)


def mapping_factor(elevation_deg):
    """Thin-shell mapping approximation: vertical TEC -> slant TEC."""
    elevation = np.radians(elevation_deg)
    ratio = (
        EARTH_RADIUS_KM
        * np.cos(elevation)
        / (EARTH_RADIUS_KM + SHELL_HEIGHT_KM)
    )
    return 1.0 / np.sqrt(1.0 - ratio**2)


def group_delay_m(tec_electrons_m2, frequency_hz):
    """First-order ionospheric group delay in meters."""
    return K_IONO * tec_electrons_m2 / frequency_hz**2


def tecu_to_electrons(tecu):
    return tecu * 1e16


def simulate_observations(vtec, rng, geometric_range=20_200_000.0):
    """
    Synthetic code pseudoranges at GPS L1 and L2.
    Independent code measurement noise is included.
    """
    slant_tec = tecu_to_electrons(vtec * mapping_factor(ELEVATION_DEG))

    delay_l1 = group_delay_m(slant_tec, F_L1)
    delay_l2 = group_delay_m(slant_tec, F_L2)

    noise_l1 = rng.normal(0, 0.30)
    noise_l2 = rng.normal(0, 0.30)

    p1 = geometric_range + delay_l1 + noise_l1
    p2 = geometric_range + delay_l2 + noise_l2

    # First-order ionosphere-free dual-frequency combination
    p_if = (
        F_L1**2 * p1 - F_L2**2 * p2
    ) / (F_L1**2 - F_L2**2)

    return {
        "vtec": vtec,
        "slant_tec": slant_tec / 1e16,
        "delay_l1": delay_l1,
        "delay_l2": delay_l2,
        "differential_delay": delay_l2 - delay_l1,
        "p1": p1,
        "p2": p2,
        "p_if": p_if,
        "if_residual": p_if - geometric_range
    }


print("\n" + "=" * 76)
print("IONA | IONOSPHERIC GNSS PROPAGATION MODEL")
print("=" * 76)

mapping = mapping_factor(ELEVATION_DEG)

print(f"GPS L1: {F_L1/1e6:.2f} MHz")
print(f"GPS L2: {F_L2/1e6:.2f} MHz")
print(f"Satellite elevation: {ELEVATION_DEG:.1f} degrees")
print(f"Thin-shell mapping factor: {mapping:.4f}")

# ------------------------------------------------------------
# 1. Static TEC scenarios
# ------------------------------------------------------------

rng = np.random.default_rng(2026)
tec_scenarios = [5, 10, 20, 30, 50]
static_results = []

print("\nIONOSPHERIC DELAY BY VERTICAL TEC")
print("-" * 76)
print(
    f"{'VTEC':>8} | {'Slant TEC':>10} | "
    f"{'L1 delay':>10} | {'L2 delay':>10} | {'L2-L1':>10}"
)
print("-" * 76)

for tec in tec_scenarios:
    result = simulate_observations(tec, rng)
    static_results.append(result)

    print(
        f"{tec:8.1f} | "
        f"{result['slant_tec']:10.2f} | "
        f"{result['delay_l1']:9.3f} m | "
        f"{result['delay_l2']:9.3f} m | "
        f"{result['differential_delay']:9.3f} m"
    )

# ------------------------------------------------------------
# 2. Time-varying synthetic ionosphere
# ------------------------------------------------------------

TIME = np.arange(0, 120, 1.0)

# Illustrative VTEC variation, not real ionospheric observations.
vtec_quiet = 10.0 + 1.5 * np.sin(2*np.pi*TIME/120)
vtec_disturbed = (
    25.0
    + 8.0 * np.sin(2*np.pi*TIME/80)
    + 3.0 * np.sin(2*np.pi*TIME/17)
)

def delay_series(vtec_series, frequency):
    slant = tecu_to_electrons(vtec_series * mapping)
    return group_delay_m(slant, frequency)


quiet_l1 = delay_series(vtec_quiet, F_L1)
disturbed_l1 = delay_series(vtec_disturbed, F_L1)
quiet_l2 = delay_series(vtec_quiet, F_L2)
disturbed_l2 = delay_series(vtec_disturbed, F_L2)

# ------------------------------------------------------------
# 3. Export results
# ------------------------------------------------------------

csv_path = OUTPUT / "tec_delay_scenarios.csv"

with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(static_results[0].keys()))
    writer.writeheader()
    writer.writerows(static_results)

# ------------------------------------------------------------
# 4. Visualization
# ------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(tec_scenarios, [r["delay_l1"] for r in static_results],
             marker="o", label="GPS L1")
axes[0].plot(tec_scenarios, [r["delay_l2"] for r in static_results],
             marker="o", label="GPS L2")
axes[0].set_title("First-order ionospheric delay")
axes[0].set_xlabel("Vertical TEC (TECU)")
axes[0].set_ylabel("Group delay (m)")
axes[0].grid(True, alpha=0.3)
axes[0].legend()

axes[1].plot(TIME, quiet_l1, label="L1 — illustrative quiet scenario")
axes[1].plot(TIME, disturbed_l1, label="L1 — illustrative disturbed scenario")
axes[1].plot(TIME, quiet_l2, "--", label="L2 — quiet")
axes[1].plot(TIME, disturbed_l2, "--", label="L2 — disturbed")
axes[1].set_title("Time-varying synthetic TEC")
axes[1].set_xlabel("Mission time (s)")
axes[1].set_ylabel("Group delay (m)")
axes[1].grid(True, alpha=0.3)
axes[1].legend()

fig.suptitle("IONA | Ionospheric Effects on GNSS Signals")
fig.tight_layout()

figure_path = OUTPUT / "ionospheric_delay.png"
fig.savefig(figure_path, dpi=180, bbox_inches="tight")

print("\nDUAL-FREQUENCY CORRECTION")
print("-" * 76)

for result in static_results:
    print(
        f"VTEC {result['vtec']:5.1f} TECU | "
        f"ionosphere-free residual: {result['if_residual']:+.3f} m"
    )

print("\nFiles generated:")
print(csv_path.resolve())
print(figure_path.resolve())
print("\nIONOSPHERIC MODEL COMPLETED")

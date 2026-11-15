import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# IONA | GNSS + IMU NAVIGATION EXPERIMENT
# Controlled simulation of GNSS outage and inertial fusion
# ============================================================

SEED = 42
rng = np.random.default_rng(SEED)

DT = 0.02
DURATION = 120.0
TIME = np.arange(0, DURATION, DT)
N = len(TIME)

GNSS_INTERVAL = 1.0
GNSS_NOISE_STD = 3.0

IMU_NOISE_STD = 0.08
IMU_BIAS = np.array([0.025, -0.018])

OUTAGE_START = 45.0
OUTAGE_END = 75.0

OUTPUT = Path("results/gnss_imu")
OUTPUT.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# 1. Ground-truth trajectory
# ------------------------------------------------------------

ax_true = (
    0.12 * np.sin(0.11 * TIME)
    + 0.05 * np.sin(0.035 * TIME)
)

ay_true = (
    0.10 * np.cos(0.09 * TIME)
    + 0.04 * np.sin(0.04 * TIME)
)

position_true = np.zeros((N, 2))
velocity_true = np.zeros((N, 2))

for k in range(1, N):
    acceleration = np.array([ax_true[k-1], ay_true[k-1]])

    position_true[k] = (
        position_true[k-1]
        + velocity_true[k-1] * DT
        + 0.5 * acceleration * DT**2
    )

    velocity_true[k] = (
        velocity_true[k-1] + acceleration * DT
    )

# ------------------------------------------------------------
# 2. Simulated IMU measurements
# ------------------------------------------------------------

acceleration_true = np.column_stack((ax_true, ay_true))

imu_measurements = (
    acceleration_true
    + IMU_BIAS
    + rng.normal(0, IMU_NOISE_STD, (N, 2))
)

# ------------------------------------------------------------
# 3. Simulated GNSS measurements
# ------------------------------------------------------------

gnss_indices = np.arange(0, N, round(GNSS_INTERVAL / DT))

gnss_measurements = (
    position_true[gnss_indices]
    + rng.normal(0, GNSS_NOISE_STD, (len(gnss_indices), 2))
)

gnss_available = (
    (TIME[gnss_indices] < OUTAGE_START)
    | (TIME[gnss_indices] >= OUTAGE_END)
)

gnss_measurements[~gnss_available] = np.nan

# GNSS-only: hold last available position during outage.
gnss_only = np.zeros((N, 2))
last_position = np.array([0.0, 0.0])

for k in range(N):
    if k in set(gnss_indices):
        j = np.searchsorted(gnss_indices, k)

        if j < len(gnss_indices) and gnss_available[j]:
            last_position = gnss_measurements[j].copy()

    gnss_only[k] = last_position

# ------------------------------------------------------------
# 4. IMU-only dead reckoning
# ------------------------------------------------------------

imu_only_position = np.zeros((N, 2))
imu_only_velocity = np.zeros((N, 2))

for k in range(1, N):
    acceleration = imu_measurements[k-1]

    imu_only_position[k] = (
        imu_only_position[k-1]
        + imu_only_velocity[k-1] * DT
        + 0.5 * acceleration * DT**2
    )

    imu_only_velocity[k] = (
        imu_only_velocity[k-1] + acceleration * DT
    )

# ------------------------------------------------------------
# 5. Extended Kalman Filter
# State: [x, y, vx, vy, bias_ax, bias_ay]
# Measurement: GNSS position [x, y]
# ------------------------------------------------------------

state = np.zeros(6)

P = np.diag([1.0, 1.0, 0.5, 0.5, 0.05, 0.05])

R = np.diag([
    GNSS_NOISE_STD**2,
    GNSS_NOISE_STD**2
])

H = np.zeros((2, 6))
H[0, 0] = 1.0
H[1, 1] = 1.0

estimated_position = np.zeros((N, 2))

for k in range(1, N):
    ax, ay = imu_measurements[k-1]
    bx, by = state[4], state[5]

    ax_corrected = ax - bx
    ay_corrected = ay - by

    # State prediction
    state[0] += state[2] * DT + 0.5 * ax_corrected * DT**2
    state[1] += state[3] * DT + 0.5 * ay_corrected * DT**2
    state[2] += ax_corrected * DT
    state[3] += ay_corrected * DT

    F = np.eye(6)
    F[0, 2] = DT
    F[1, 3] = DT
    F[0, 4] = -0.5 * DT**2
    F[1, 5] = -0.5 * DT**2
    F[2, 4] = -DT
    F[3, 5] = -DT

    Q = np.diag([
        0.25 * IMU_NOISE_STD**2 * DT**4,
        0.25 * IMU_NOISE_STD**2 * DT**4,
        IMU_NOISE_STD**2 * DT**2,
        IMU_NOISE_STD**2 * DT**2,
        1e-7 * DT,
        1e-7 * DT
    ])

    P = F @ P @ F.T + Q

    # GNSS measurement update, when available
    if k in set(gnss_indices):
        j = np.searchsorted(gnss_indices, k)

        if j < len(gnss_indices) and gnss_available[j]:
            z = gnss_measurements[j]
            innovation = z - H @ state
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)

            state = state + K @ innovation

            # Joseph covariance update for numerical stability
            I = np.eye(6)
            P = (
                (I - K @ H) @ P @ (I - K @ H).T
                + K @ R @ K.T
            )

    estimated_position[k] = state[:2]

# ------------------------------------------------------------
# 6. Error analysis
# ------------------------------------------------------------

def position_error(estimate):
    return np.linalg.norm(estimate - position_true, axis=1)

errors = {
    "GNSS only": position_error(gnss_only),
    "IMU only": position_error(imu_only_position),
    "GNSS + IMU": position_error(estimated_position)
}

outage_mask = (
    (TIME >= OUTAGE_START)
    & (TIME < OUTAGE_END)
)

print("\n" + "=" * 58)
print("IONA | GNSS + IMU NAVIGATION EXPERIMENT")
print("=" * 58)

print(f"Simulation duration: {DURATION:.1f} s")
print(f"IMU sampling rate: {1/DT:.1f} Hz")
print(f"GNSS sampling rate: {1/GNSS_INTERVAL:.1f} Hz")
print(f"GNSS outage: {OUTAGE_START:.1f}–{OUTAGE_END:.1f} s")

print("\nPOSITION ERROR DURING GNSS OUTAGE")
print("-" * 58)

for name, error in errors.items():
    rmse = np.sqrt(np.mean(error[outage_mask]**2))
    maximum = np.max(error[outage_mask])

    print(f"{name:15s} | RMSE: {rmse:8.3f} m | Max: {maximum:8.3f} m")

print("\nPOSITION ERROR OVER FULL SIMULATION")
print("-" * 58)

for name, error in errors.items():
    rmse = np.sqrt(np.mean(error**2))
    print(f"{name:15s} | RMSE: {rmse:8.3f} m")

# ------------------------------------------------------------
# 7. Scientific visualization
# ------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

ax = axes[0]

ax.plot(
    position_true[:, 0],
    position_true[:, 1],
    label="Reference trajectory",
    linewidth=2.5
)

ax.plot(
    gnss_only[:, 0],
    gnss_only[:, 1],
    label="GNSS only",
    alpha=0.8
)

ax.plot(
    imu_only_position[:, 0],
    imu_only_position[:, 1],
    label="IMU only"
)

ax.plot(
    estimated_position[:, 0],
    estimated_position[:, 1],
    label="GNSS + IMU (EKF)",
    linewidth=2
)

ax.axvspan(
    position_true[outage_mask, 0].min(),
    position_true[outage_mask, 0].max(),
    alpha=0.08
)

ax.set_title("Navigation trajectory comparison")
ax.set_xlabel("East position (m)")
ax.set_ylabel("North position (m)")
ax.axis("equal")
ax.grid(True, alpha=0.3)
ax.legend()

ax = axes[1]

for name, error in errors.items():
    ax.plot(TIME, error, label=name)

ax.axvspan(
    OUTAGE_START,
    OUTAGE_END,
    alpha=0.15,
    label="GNSS outage"
)

ax.set_title("Position estimation error")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Position error (m)")
ax.grid(True, alpha=0.3)
ax.legend()

fig.suptitle("IONA | Autonomous Navigation Laboratory", fontsize=15)
fig.tight_layout()

figure_path = OUTPUT / "gnss_imu_comparison.png"
fig.savefig(figure_path, dpi=180, bbox_inches="tight")

print("\nFigure saved:", figure_path.resolve())
print("EXPERIMENT COMPLETED")

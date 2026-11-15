import csv
import numpy as np
from pathlib import Path

# ============================================================
# IONA | MONTE CARLO VALIDATION
# GNSS vs IMU vs GNSS + IMU (EKF)
# ============================================================

SEED = 2026
TRIALS = 100

DT = 0.02
DURATION = 120.0
TIME = np.arange(0, DURATION, DT)
N = len(TIME)

GNSS_INTERVAL = 1.0
GNSS_STEP = round(GNSS_INTERVAL / DT)
GNSS_NOISE_STD = 3.0

IMU_NOISE_STD = 0.08
OUTAGE_START = 45.0
OUTAGE_END = 75.0

OUTAGE = (TIME >= OUTAGE_START) & (TIME < OUTAGE_END)
GNSS_INDICES = np.arange(0, N, GNSS_STEP)
GNSS_LOOKUP = {int(k): j for j, k in enumerate(GNSS_INDICES)}

OUTPUT = Path("results/gnss_imu")
OUTPUT.mkdir(parents=True, exist_ok=True)

# Fixed reference trajectory
ax_true = (
    0.12 * np.sin(0.11 * TIME)
    + 0.05 * np.sin(0.035 * TIME)
)
ay_true = (
    0.10 * np.cos(0.09 * TIME)
    + 0.04 * np.sin(0.04 * TIME)
)

acc_true = np.column_stack((ax_true, ay_true))
position_true = np.zeros((N, 2))
velocity_true = np.zeros((N, 2))

for k in range(1, N):
    position_true[k] = (
        position_true[k-1]
        + velocity_true[k-1] * DT
        + 0.5 * acc_true[k-1] * DT**2
    )
    velocity_true[k] = velocity_true[k-1] + acc_true[k-1] * DT


def run_trial(seed):
    rng = np.random.default_rng(seed)

    # IMU: white noise plus a slightly varying constant bias
    imu_bias = np.array([0.025, -0.018]) + rng.normal(0, 0.01, 2)
    imu = acc_true + imu_bias + rng.normal(
        0, IMU_NOISE_STD, (N, 2)
    )

    # GNSS observations
    gnss = position_true[GNSS_INDICES] + rng.normal(
        0, GNSS_NOISE_STD, (len(GNSS_INDICES), 2)
    )

    gnss_times = TIME[GNSS_INDICES]
    available = (
        (gnss_times < OUTAGE_START)
        | (gnss_times >= OUTAGE_END)
    )
    gnss[~available] = np.nan

    # A. GNSS only: hold last received position during outage
    gnss_only = np.zeros((N, 2))
    last_position = np.zeros(2)

    for k in range(N):
        j = GNSS_LOOKUP.get(k)
        if j is not None and available[j]:
            last_position = gnss[j].copy()
        gnss_only[k] = last_position

    # B. IMU-only dead reckoning
    imu_only = np.zeros((N, 2))
    imu_velocity = np.zeros((N, 2))

    for k in range(1, N):
        imu_only[k] = (
            imu_only[k-1]
            + imu_velocity[k-1] * DT
            + 0.5 * imu[k-1] * DT**2
        )
        imu_velocity[k] = imu_velocity[k-1] + imu[k-1] * DT

    # C. GNSS + IMU Extended Kalman Filter
    # State: x, y, vx, vy, accelerometer bias x, accelerometer bias y
    state = np.zeros(6)
    P = np.diag([1.0, 1.0, 0.5, 0.5, 0.05, 0.05])

    H = np.zeros((2, 6))
    H[0, 0] = 1.0
    H[1, 1] = 1.0

    R = np.eye(2) * GNSS_NOISE_STD**2
    I = np.eye(6)

    fused = np.zeros((N, 2))

    for k in range(1, N):
        ax, ay = imu[k-1]
        bx, by = state[4], state[5]

        # Prediction
        state[0] += state[2] * DT + 0.5 * (ax-bx) * DT**2
        state[1] += state[3] * DT + 0.5 * (ay-by) * DT**2
        state[2] += (ax-bx) * DT
        state[3] += (ay-by) * DT

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

        # Measurement update whenever GNSS is available
        j = GNSS_LOOKUP.get(k)
        if j is not None and available[j]:
            innovation = gnss[j] - H @ state
            S = H @ P @ H.T + R
            K = np.linalg.solve(S, H @ P).T

            state = state + K @ innovation
            P = (
                (I - K @ H) @ P @ (I - K @ H).T
                + K @ R @ K.T
            )

        fused[k] = state[:2]

    def rmse(estimate, mask):
        error = np.linalg.norm(
            estimate[mask] - position_true[mask], axis=1
        )
        return float(np.sqrt(np.mean(error**2)))

    return {
        "gnss_outage": rmse(gnss_only, OUTAGE),
        "imu_outage": rmse(imu_only, OUTAGE),
        "ekf_outage": rmse(fused, OUTAGE),
        "gnss_full": rmse(gnss_only, np.ones(N, dtype=bool)),
        "imu_full": rmse(imu_only, np.ones(N, dtype=bool)),
        "ekf_full": rmse(fused, np.ones(N, dtype=bool))
    }


# Run independent trials
results = [run_trial(SEED + i) for i in range(TRIALS)]

fields = list(results[0].keys())
csv_path = OUTPUT / "monte_carlo_results.csv"

with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(results)


def report(label, key):
    values = np.array([r[key] for r in results])
    print(
        f"{label:15s} | "
        f"Mean: {values.mean():8.3f} m | "
        f"SD: {values.std(ddof=1):7.3f} m | "
        f"Median: {np.median(values):8.3f} m | "
        f"P95: {np.percentile(values, 95):8.3f} m"
    )


gnss = np.array([r["gnss_outage"] for r in results])
ekf = np.array([r["ekf_outage"] for r in results])
improvement = 100 * (1 - ekf / gnss)
wins = np.mean(ekf < gnss) * 100

print("\n" + "=" * 76)
print("IONA | MONTE CARLO NAVIGATION VALIDATION")
print("=" * 76)
print(f"Independent trials: {TRIALS}")
print(f"GNSS outage: {OUTAGE_START:.0f} to {OUTAGE_END:.0f} seconds")
print(f"IMU rate: {1/DT:.0f} Hz | GNSS rate: {1/GNSS_INTERVAL:.0f} Hz")

print("\nRMSE DURING GNSS OUTAGE")
print("-" * 76)
report("GNSS only", "gnss_outage")
report("IMU only", "imu_outage")
report("GNSS + IMU", "ekf_outage")

print("\nRMSE OVER FULL SIMULATION")
print("-" * 76)
report("GNSS only", "gnss_full")
report("IMU only", "imu_full")
report("GNSS + IMU", "ekf_full")

print("\nROBUSTNESS")
print("-" * 76)
print(f"Mean paired RMSE reduction vs GNSS: {improvement.mean():.2f}%")
print(f"Median paired RMSE reduction:       {np.median(improvement):.2f}%")
print(f"Trials where EKF beats GNSS:        {wins:.1f}%")
print(f"CSV saved: {csv_path.resolve()}")
print("MONTE CARLO VALIDATION COMPLETED")

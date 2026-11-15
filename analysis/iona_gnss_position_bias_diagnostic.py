import numpy as np
import iona_velocity_consistency_diagnostic as diag


def main():
    vtec = diag.sim.make_tec_scenario(diag.SCENARIO)

    phases = ["pre", "outage", "post"]
    records = {
        phase: {
            "gnss": [],
            "ekf_position": [],
            "ekf_velocity": []
        }
        for phase in phases
    }

    print("IONA | GNSS POSITION BIAS DIAGNOSTIC")
    print("Trials:", diag.TRIALS)

    for trial in range(diag.TRIALS):
        seed = diag.BASE_SEED + trial

        fixes, R_series = diag.simulate_gnss(
            vtec,
            diag.TEC_ERROR_PERCENT,
            seed
        )

        estimates, _ = diag.run_filter(
            fixes,
            R_series,
            seed + 1000003
        )

        gnss_errors = (
            fixes - diag.true_state[
                diag.sim.GNSS_INDICES, :2
            ]
        )

        for phase in phases:
            time_mask = diag.phase_mask(phase)

            gnss_mask = time_mask[
                diag.sim.GNSS_INDICES
            ]

            records[phase]["gnss"].append(
                np.mean(gnss_errors[gnss_mask], axis=0)
            )

            records[phase]["ekf_position"].append(
                np.mean(
                    estimates[time_mask, :2]
                    - diag.true_state[time_mask, :2],
                    axis=0
                )
            )

            records[phase]["ekf_velocity"].append(
                np.mean(
                    estimates[time_mask, 2:]
                    - diag.true_state[time_mask, 2:],
                    axis=0
                )
            )

        if (trial + 1) % 10 == 0:
            print(f"Completed {trial + 1}/{diag.TRIALS}")

    print("\nRESULTS: MEAN PER TRIAL")
    print("z = mean / standard error")

    for phase in phases:
        print("\n" + "=" * 55)
        print("PHASE:", phase.upper())
        print("=" * 55)

        for name, values in records[phase].items():
            values = np.asarray(values)

            mean = np.mean(values, axis=0)
            sd = np.std(values, axis=0, ddof=1)
            se = sd / np.sqrt(diag.TRIALS)

            z = np.divide(
                mean,
                se,
                out=np.zeros_like(mean),
                where=se > 0
            )

            print("\n" + name.upper())
            print("Mean:", mean)
            print("Across-trial SD:", sd)
            print("Standard error:", se)
            print("Mean / SE:", z)

    print("\nGNSS EPOCHS PER PHASE")
    for phase in phases:
        mask = diag.phase_mask(phase)
        print(
            phase,
            int(np.sum(mask[diag.sim.GNSS_INDICES]))
        )


if __name__ == "__main__":
    main()

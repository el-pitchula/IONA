import numpy as np
import iona_velocity_consistency_diagnostic as diag


def main():
    vtec = diag.sim.make_tec_scenario(diag.SCENARIO)

    results = {
        phase: {
            "position": [],
            "velocity": [],
            "conditional": []
        }
        for phase in ["pre", "outage", "post"]
    }

    print("IONA | CONDITIONAL BIAS DIAGNOSTIC")
    print("Trials:", diag.TRIALS)

    for trial in range(diag.TRIALS):
        seed = diag.BASE_SEED + trial

        fixes, R_series = diag.simulate_gnss(
            vtec, diag.TEC_ERROR_PERCENT, seed
        )

        estimates, covariances = diag.run_filter(
            fixes, R_series, seed + 1000003
        )

        errors = estimates - diag.true_state

        for phase in results:
            mask = diag.phase_mask(phase)
            selected = errors[mask]
            selected_P = covariances[mask]

            conditional = []

            for error, P in zip(selected, selected_P):
                Ppp = P[:2, :2]
                Ppv = P[:2, 2:]
                Pvp = P[2:, :2]
                Pvv = P[2:, 2:]

                ep = error[:2]
                ev = error[2:]

                conditional.append(
                    ev - Pvp @ np.linalg.solve(Ppp, ep)
                )

            results[phase]["position"].append(
                np.mean(selected[:, :2], axis=0)
            )
            results[phase]["velocity"].append(
                np.mean(selected[:, 2:], axis=0)
            )
            results[phase]["conditional"].append(
                np.mean(conditional, axis=0)
            )

        if (trial + 1) % 10 == 0:
            print(f"Completed {trial + 1}/{diag.TRIALS}")

    print("\nRESULTS: MEAN ERROR PER INDEPENDENT TRIAL")
    print("z = mean / standard error")
    print("Values are descriptive; temporal samples are not treated as independent.")

    for phase, groups in results.items():
        print("\n" + "=" * 60)
        print("PHASE:", phase.upper())
        print("=" * 60)

        for name, values in groups.items():
            values = np.asarray(values)

            mean = np.mean(values, axis=0)
            sd = np.std(values, axis=0, ddof=1)
            se = sd / np.sqrt(len(values))
            z = np.divide(
                mean, se,
                out=np.zeros_like(mean),
                where=se > 0
            )

            print(f"\n{name.upper()}")
            print("Mean:", mean)
            print("Across-trial SD:", sd)
            print("Standard error:", se)
            print("Mean / SE:", z)

        conditional = np.asarray(groups["conditional"])

        print("\nCONDITIONAL BIAS MAGNITUDE")
        print("Mean vector norm:", np.linalg.norm(
            np.mean(conditional, axis=0)
        ))
        print("Mean trial-vector norm:", np.mean(
            np.linalg.norm(conditional, axis=1)
        ))


if __name__ == "__main__":
    main()

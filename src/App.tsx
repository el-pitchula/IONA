
import { useEffect, useMemo, useState } from "react";
import "./App.css";

type Sample = {
  time_s: number;
  true_east_m: number;
  true_north_m: number;
  gnss_east_m: number;
  gnss_north_m: number;
  imu_east_m: number;
  imu_north_m: number;
  ekf_l1_east_m: number;
  ekf_l1_north_m: number;
  ekf_corrected_east_m: number;
  ekf_corrected_north_m: number;
  ekf_dual_east_m: number;
  ekf_dual_north_m: number;
  gnss_available: number;
};

type SolutionKey = "dual" | "l1" | "tec" | "imu";

const SOLUTIONS: {
  id: SolutionKey;
  label: string;
  description: string;
  east: keyof Sample;
  north: keyof Sample;
}[] = [
  {
    id: "dual",
    label: "Kalman Â· L1 + L2",
    description: "Dual-frequency GNSS",
    east: "ekf_dual_east_m",
    north: "ekf_dual_north_m",
  },
  {
    id: "l1",
    label: "Kalman Â· L1",
    description: "Single-frequency GNSS",
    east: "ekf_l1_east_m",
    north: "ekf_l1_north_m",
  },
  {
    id: "tec",
    label: "Kalman Â· ideal TEC",
    description: "Idealized ionospheric correction",
    east: "ekf_corrected_east_m",
    north: "ekf_corrected_north_m",
  },
  {
    id: "imu",
    label: "IMU only",
    description: "Inertial dead reckoning",
    east: "imu_east_m",
    north: "imu_north_m",
  },
];

const OUTAGE_START = 45;
const OUTAGE_END = 75;
const DURATION = 120;

function parseCSV(text: string): Sample[] {
  const lines = text.trim().replace(/^\uFEFF/, "").split(/\r?\n/);
  const headers = lines[0].split(",");

  return lines.slice(1).filter(Boolean).map((line) => {
    const values = line.split(",");
    const row: Record<string, number> = {};

    headers.forEach((header, index) => {
      const value = values[index]?.trim();
      row[header.trim()] =
        value === undefined || value === "" ? Number.NaN : Number(value);
    });

    return row as Sample;
  });
}

function formatMeters(value: number): string {
  return Number.isFinite(value) ? `${value.toFixed(2)} m` : "â€”";
}

function App() {
  const [samples, setSamples] = useState<Sample[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState("");
  const [solution, setSolution] = useState<SolutionKey>("dual");
  const [sampleIndex, setSampleIndex] = useState(0);
  const [playing, setPlaying] = useState(false);

  useEffect(() => {
    fetch("/data/mission_timeseries.csv")
      .then((response) => {
        if (!response.ok) throw new Error("Mission dataset not found.");
        return response.text();
      })
      .then((text) => {
        const parsed = parseCSV(text);
        if (!parsed.length) throw new Error("The mission dataset is empty.");
        setSamples(parsed);
        setLoading(false);
      })
      .catch((error: Error) => {
        setLoadError(error.message);
        setLoading(false);
      });
  }, []);

  useEffect(() => {
    if (!playing || samples.length === 0) return;

    const timer = window.setInterval(() => {
      setSampleIndex((current) => {
        if (current >= samples.length - 1) {
          setPlaying(false);
          return 0;
        }
        return Math.min(current + 5, samples.length - 1);
      });
    }, 50);

    return () => window.clearInterval(timer);
  }, [playing, samples.length]);

  const selectedSolution = SOLUTIONS.find((item) => item.id === solution)!;
  const currentSample = samples[sampleIndex];

  const metrics = useMemo(() => {
    if (!samples.length) return { full: NaN, outage: NaN };

    let fullSquaredError = 0;
    let fullCount = 0;
    let outageSquaredError = 0;
    let outageCount = 0;

    samples.forEach((sample) => {
      const east = sample[selectedSolution.east] as number;
      const north = sample[selectedSolution.north] as number;

      if (!Number.isFinite(east) || !Number.isFinite(north)) return;

      const errorSquared =
        (east - sample.true_east_m) ** 2 +
        (north - sample.true_north_m) ** 2;

      fullSquaredError += errorSquared;
      fullCount++;

      if (sample.time_s >= OUTAGE_START && sample.time_s <= OUTAGE_END) {
        outageSquaredError += errorSquared;
        outageCount++;
      }
    });

    return {
      full: fullCount ? Math.sqrt(fullSquaredError / fullCount) : NaN,
      outage: outageCount ? Math.sqrt(outageSquaredError / outageCount) : NaN,
    };
  }, [samples, selectedSolution]);

  const geometry = useMemo(() => {
    if (!samples.length) return null;

    const coordinates: number[] = [];

    samples.forEach((sample) => {
      coordinates.push(sample.true_east_m, sample.true_north_m);

      const east = sample[selectedSolution.east] as number;
      const north = sample[selectedSolution.north] as number;

      if (Number.isFinite(east) && Number.isFinite(north)) {
        coordinates.push(east, north);
      }
    });

    const eastValues = coordinates.filter((_, index) => index % 2 === 0);
    const northValues = coordinates.filter((_, index) => index % 2 === 1);

    const minEast = Math.min(...eastValues);
    const maxEast = Math.max(...eastValues);
    const minNorth = Math.min(...northValues);
    const maxNorth = Math.max(...northValues);

    const width = 820;
    const height = 440;

    const mapX = (value: number) =>
      55 + ((value - minEast) / (maxEast - minEast || 1)) * 710;

    const mapY = (value: number) =>
      height - 45 -
      ((value - minNorth) / (maxNorth - minNorth || 1)) * 350;

    const makePath = (
      data: Sample[],
      eastKey: keyof Sample,
      northKey: keyof Sample,
    ) =>
      data
        .filter(
          (sample) =>
            Number.isFinite(sample[eastKey]) &&
            Number.isFinite(sample[northKey]),
        )
        .map((sample, index) => {
          const x = mapX(sample[eastKey] as number);
          const y = mapY(sample[northKey] as number);
          return `${index === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`;
        })
        .join(" ");

    const outageSamples = samples.filter(
      (sample) =>
        sample.time_s >= OUTAGE_START && sample.time_s <= OUTAGE_END,
    );

    const gnssPoints = samples.filter(
      (sample) =>
        sample.gnss_available === 1 &&
        Number.isFinite(sample.gnss_east_m) &&
        Number.isFinite(sample.gnss_north_m),
    );

    return {
      width,
      height,
      mapX,
      mapY,
      truthPath: makePath(samples, "true_east_m", "true_north_m"),
      estimatePath: makePath(
        samples,
        selectedSolution.east,
        selectedSolution.north,
      ),
      outagePath: makePath(
        outageSamples,
        selectedSolution.east,
        selectedSolution.north,
      ),
      gnssPoints,
    };
  }, [samples, selectedSolution]);

  const errorChart = useMemo(() => {
    if (!samples.length) return null;

    const errors = samples.map((sample) =>
      Math.hypot(
        (sample[selectedSolution.east] as number) - sample.true_east_m,
        (sample[selectedSolution.north] as number) - sample.true_north_m,
      ),
    );

    const finiteErrors = errors.filter(Number.isFinite);
    const maxError = Math.max(1, ...finiteErrors) * 1.12;

    const points = errors
      .map((error, index) => {
        if (!Number.isFinite(error)) return "";
        const x = 45 + (index / (samples.length - 1)) * 730;
        const y = 155 - (error / maxError) * 120;
        return `${x.toFixed(2)},${y.toFixed(2)}`;
      })
      .filter(Boolean)
      .join(" ");

    return { points, maxError };
  }, [samples, selectedSolution]);

  const currentTime = currentSample?.time_s ?? 0;
  const inOutage =
    currentTime >= OUTAGE_START && currentTime <= OUTAGE_END;

  const currentError = currentSample
    ? Math.hypot(
        (currentSample[selectedSolution.east] as number) -
          currentSample.true_east_m,
        (currentSample[selectedSolution.north] as number) -
          currentSample.true_north_m,
      )
    : NaN;
const handlePlay = () => {
    if (sampleIndex >= samples.length - 1) setSampleIndex(0);
    setPlaying((value) => !value);
  };

  return (
    <div className="iona">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">I</div>
          <div>
            <div className="brand-name">IONA</div>
            <div className="brand-subtitle">INTELLIGENT NAVIGATION ARCHITECTURE</div>
          </div>
        </div>

        <div className="topbar-center">
          <span className="live-dot" />
          MISSION SIMULATION
        </div>

        <div className="topbar-right">
          <span className="status-pill">RESEARCH PROTOTYPE</span>
          <span className="version-label">v1.0</span>
        </div>
      </header>

      <main className="main-content">
        <section className="intro">
          <div className="eyebrow">RESILIENT POSITIONING SYSTEMS / 001</div>
          <h1>Navigation beyond the signal.</h1>
          <p>
            Multi-sensor navigation under GNSS outages and ionospheric
            uncertainty. Explore how inertial measurements and satellite
            positioning work together when the signal disappears.
          </p>
        </section>

        <section className="mission-strip">
          <div className="mission-info">
            <span className="section-kicker">MISSION PROFILE</span>
            <strong>2D Integrated Navigation Test</strong>
            <span className="muted">Synthetic local trajectory Â· 120 seconds</span>
          </div>
          <div className="mission-stat">
            <span>DURATION</span>
            <strong>120 <small>SEC</small></strong>
          </div>
          <div className="mission-stat">
            <span>GNSS OUTAGE</span>
            <strong>30 <small>SEC</small></strong>
          </div>
          <div className="mission-stat">
            <span>IMU RATE</span>
            <strong>50 <small>HZ</small></strong>
          </div>
          <div className="mission-stat">
            <span>DATA POINTS</span>
            <strong>{samples.length.toLocaleString("en-US")}</strong>
          </div>
        </section>

        {loading && <div className="notice">Loading mission telemetryâ€¦</div>}

        {loadError && (
          <div className="notice error">
            Unable to load mission data: {loadError}
            <br />
            Check that the CSV is available at /data/mission_timeseries.csv.
          </div>
        )}

        {!loading && !loadError && currentSample && geometry && (
          <>
            <section className="workspace">
              <div className="panel trajectory-panel">
                <div className="panel-heading">
                  <div>
                    <div className="section-kicker">LIVE TELEMETRY</div>
                    <h2>Trajectory reconstruction</h2>
                    <p>Reference path versus estimated navigation solution</p>
                  </div>
                  <div className={`signal-state ${inOutage ? "outage" : ""}`}>
                    <span />
                    {inOutage ? "GNSS SIGNAL LOST" : "GNSS SIGNAL AVAILABLE"}
                  </div>
                </div>

                <div className="chart-wrap">
                  <svg
                    className="trajectory-chart"
                    viewBox={`0 0 ${geometry.width} ${geometry.height}`}
                    role="img"
                    aria-label="Two-dimensional navigation trajectory"
                  >
                    {[0, 1, 2, 3, 4].map((index) => (
                      <g key={index}>
                        <line
                          x1={55 + index * 177.5}
                          y1="35"
                          x2={55 + index * 177.5}
                          y2="395"
                          className="grid-line"
                        />
                        <line
                          x1="55"
                          y1={45 + index * 87.5}
                          x2="765"
                          y2={45 + index * 87.5}
                          className="grid-line"
                        />
                      </g>
                    ))}

                    <text x="55" y="22" className="axis-label">NORTH / m</text>
                    <text x="765" y="425" textAnchor="end" className="axis-label">EAST / m</text>

                    <path d={geometry.truthPath} className="truth-path" />
                    <path d={geometry.estimatePath} className="estimate-path" />
                    <path d={geometry.outagePath} className="outage-path" />

                    {geometry.gnssPoints.map((point, index) => (
                      <circle
                        key={index}
                        cx={geometry.mapX(point.gnss_east_m)}
                        cy={geometry.mapY(point.gnss_north_m)}
                        r="2.7"
                        className="gnss-point"
                      />
                    ))}

                    <circle
                      cx={geometry.mapX(currentSample.true_east_m)}
                      cy={geometry.mapY(currentSample.true_north_m)}
                      r="6"
                      className="reference-marker"
                    />
                    <circle
                      cx={geometry.mapX(currentSample[selectedSolution.east] as number)}
                      cy={geometry.mapY(currentSample[selectedSolution.north] as number)}
                      r="6"
                      className="estimate-marker"
                    />
                  </svg>
                </div>

                <div className="legend">
                  <span><i className="legend-line reference" /> Reference trajectory</span>
                  <span><i className="legend-line estimate" /> Navigation estimate</span>
                  <span><i className="legend-dot gnss" /> GNSS fixes</span>
                  <span><i className="legend-line outage" /> Outage interval</span>
                </div>
              </div>

              <aside className="side-column">
                <div className="panel solution-panel">
                  <div className="section-kicker">ESTIMATION MODEL</div>
                  <h2>Navigation solution</h2>
                  <p className="panel-description">
                    Select the estimator to inspect its reconstructed trajectory.
                  </p>

                  <div className="solution-options">
                    {SOLUTIONS.map((item) => (
                      <button
                        key={item.id}
                        className={`solution-option ${solution === item.id ? "active" : ""}`}
                        onClick={() => setSolution(item.id)}
                      >
                        <span className="option-radio" />
                        <span className="option-copy">
                          <strong>{item.label}</strong>
                          <small>{item.description}</small>
                        </span>
                      </button>
                    ))}
                  </div>
                </div>

                <div className="panel metrics-panel">
                  <div className="section-kicker">PERFORMANCE</div>
                  <h2>Navigation accuracy</h2>

                  <div className="metric-block">
                    <span>FULL MISSION RMSE</span>
                    <strong>{formatMeters(metrics.full)}</strong>
                  </div>
                  <div className="metric-block highlight">
                    <span>OUTAGE RMSE Â· 45â€“75 SEC</span>
                    <strong>{formatMeters(metrics.outage)}</strong>
                  </div>
                  <div className="metric-footnote">
                    Position error against the simulated reference trajectory.
                  </div>
                </div>
              </aside>
            </section>

            <section className="panel timeline-panel">
              <div className="timeline-heading">
                <div>
                  <div className="section-kicker">MISSION CONTROL</div>
                  <h2>Time-domain playback</h2>
                </div>
                <div className="mission-clock">
                  <strong>{currentTime.toFixed(2)}</strong>
                  <span> / 120.00 SEC</span>
                </div>
              </div>

              <div className="timeline-track-wrap">
                <div className="timeline-track">
                  <div
                    className="outage-band"
                    style={{
                      left: `${(OUTAGE_START / DURATION) * 100}%`,
                      width: `${((OUTAGE_END - OUTAGE_START) / DURATION) * 100}%`,
                    }}
                  />
                  <div
                    className="timeline-progress"
                    style={{ width: `${(currentTime / DURATION) * 100}%` }}
                  />
                </div>
                <input
                  className="time-slider"
                  type="range"
                  min="0"
                  max={samples.length - 1}
                  value={sampleIndex}
                  onChange={(event) => {
                    setPlaying(false);
                    setSampleIndex(Number(event.target.value));
                  }}
                  aria-label="Mission playback time"
                />
              </div>

              <div className="timeline-labels">
                <span>00:00</span>
                <span className="outage-label">GNSS OUTAGE Â· 00:45â€“01:15</span>
                <span>02:00</span>
              </div>

              <div className="playback-controls">
                <button className="play-button" onClick={handlePlay}>
                  {playing ? "â…¡ PAUSE" : "â–¶ PLAY MISSION"}
                </button>
                <button
                  className="reset-button"
                  onClick={() => {
                    setPlaying(false);
                    setSampleIndex(0);
                  }}
                >
                  RESET
                </button>

                <div className="current-readout">
                  <span>INSTANTANEOUS POSITION ERROR</span>
                  <strong>{formatMeters(currentError)}</strong>
                </div>
                <div className="current-readout">
                  <span>ESTIMATED POSITION Â· EAST</span>
                  <strong>{formatMeters(currentSample[selectedSolution.east] as number)}</strong>
                </div>
                <div className="current-readout">
                  <span>ESTIMATED POSITION Â· NORTH</span>
                  <strong>{formatMeters(currentSample[selectedSolution.north] as number)}</strong>
                </div>
              </div>
            </section>

            <section className="panel error-chart-panel">
              <div className="panel-heading">
                <div>
                  <div className="section-kicker">ESTIMATION PERFORMANCE</div>
                  <h2>Position error over mission time</h2>
                  <p>Euclidean error relative to the simulated reference trajectory</p>
                </div>
                <div className="error-current">
                  {formatMeters(currentError)}
                  <small>CURRENT ERROR</small>
                </div>
              </div>

              {errorChart && (
                <div className="error-chart-wrap">
                  <svg viewBox="0 0 820 190" className="error-chart">
                    {[35, 75, 115, 155].map((y) => (
                      <line
                        key={y}
                        x1="45"
                        y1={y}
                        x2="775"
                        y2={y}
                        className="grid-line"
                      />
                    ))}

                    <rect
                      x={45 + (OUTAGE_START / DURATION) * 730}
                      y="25"
                      width={((OUTAGE_END - OUTAGE_START) / DURATION) * 730}
                      height="130"
                      className="error-outage-region"
                    />

                    <polyline points={errorChart.points} className="error-line" />

                    <line
                      x1={45 + (currentTime / DURATION) * 730}
                      y1="25"
                      x2={45 + (currentTime / DURATION) * 730}
                      y2="155"
                      className="error-cursor"
                    />

                    <text x="45" y="178" className="axis-label">0 s</text>
                    <text x="410" y="178" textAnchor="middle" className="axis-label">60 s</text>
                    <text x="775" y="178" textAnchor="end" className="axis-label">120 s</text>
                    <text x="775" y="17" textAnchor="end" className="axis-label">
                      MAX {errorChart.maxError.toFixed(1)} m
                    </text>
                  </svg>
                </div>
              )}

              <div className="error-legend">
                <span><i /> Position error</span>
                <span className="outage-key">GNSS outage Â· 45â€“75 s</span>
                <span>Vertical marker Â· selected time</span>
              </div>
            </section>

            <section className="lower-grid">
              <div className="panel concept-panel">
                <div className="section-kicker">SYSTEM BEHAVIOR</div>
                <h2>When GNSS disappears</h2>
                <p>
                  During the simulated outage, satellite measurements are
                  unavailable. The navigation filter propagates the state
                  using inertial measurements and resumes GNSS updates when
                  the signal returns.
                </p>
                <div className="process-flow">
                  <div className="process-node">
                    <span>01</span>
                    <strong>GNSS</strong>
                    <small>Position fixes</small>
                  </div>
                  <div className="process-arrow">â†’</div>
                  <div className="process-node">
                    <span>02</span>
                    <strong>IMU</strong>
                    <small>Motion propagation</small>
                  </div>
                  <div className="process-arrow">â†’</div>
                  <div className="process-node">
                    <span>03</span>
                    <strong>KALMAN</strong>
                    <small>State estimation</small>
                  </div>
                </div>
              </div>

              <div className="panel experiment-panel">
                <div className="section-kicker">EXPERIMENTAL SCOPE</div>
                <h2>What this prototype validates</h2>
                <ul>
                  <li>Position estimation during a 30-second GNSS outage.</li>
                  <li>Comparison of L1, dual-frequency and idealized TEC cases.</li>
                  <li>Performance against a known synthetic reference trajectory.</li>
                </ul>
                <div className="scope-note">
                  Synthetic 2D simulation. The idealized TEC case assumes exact
                  knowledge of the simulated ionospheric delay; it is a
                  reference scenario, not an operational correction.
                </div>
              </div>
            </section>
          </>
        )}

        <footer className="footer">
          <div>IONA <span>Â·</span> INTELLIGENT NAVIGATION ARCHITECTURE</div>
          <div>RESEARCH PROTOTYPE Â· SYNTHETIC DATA Â· 2026</div>
        </footer>
      </main>
    </div>
  );
}

export default App;



import { useState } from "react";

type AnnualRecord = {
  year: number;
  precipitation_mm: number;
};

type Props = {
  records: AnnualRecord[];
};

export default function AnnualPrecipitationPlot({
  records,
}: Props) {
  const [selected, setSelected] = useState<number | null>(null);

  const data = [...records]
    .filter(
      (item) =>
        Number.isFinite(item.year) &&
        Number.isFinite(item.precipitation_mm) &&
        item.precipitation_mm >= 0
    )
    .sort((a, b) => a.year - b.year);

  if (data.length === 0) {
    return <p>Nenhum dado disponível para o gráfico.</p>;
  }

  const width = 900;
  const height = 350;
  const left = 70;
  const right = 25;
  const top = 25;
  const bottom = 295;

  const plotWidth = width - left - right;
  const plotHeight = bottom - top;

  const maxValue =
    Math.ceil(
      Math.max(...data.map((item) => item.precipitation_mm), 1) / 250
    ) * 250;

  const getX = (index: number) =>
    left + (index / Math.max(data.length - 1, 1)) * plotWidth;

  const getY = (value: number) =>
    bottom - (value / maxValue) * plotHeight;

  const points = data
    .map(
      (item, index) =>
        `${getX(index)},${getY(item.precipitation_mm)}`
    )
    .join(" ");

  const active = selected === null ? null : data[selected];

  return (
    <div className="iona-precip-chart">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        role="group"
        aria-label="Gráfico interativo de precipitação anual"
        onPointerLeave={() => setSelected(null)}
        style={{ width: "100%", height: "auto" }}
      >
        <defs>
          <linearGradient id="iona-rain-fill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#61e0c0" stopOpacity="0.24" />
            <stop offset="100%" stopColor="#61e0c0" stopOpacity="0" />
          </linearGradient>
        </defs>

        {[0, 0.25, 0.5, 0.75, 1].map((fraction) => {
          const value = fraction * maxValue;
          const y = getY(value);

          return (
            <g key={fraction}>
              <line
                x1={left}
                y1={y}
                x2={width - right}
                y2={y}
                stroke="#35515d"
                strokeDasharray="4 6"
              />
              <text
                x={left - 12}
                y={y + 4}
                textAnchor="end"
                fill="#9db8c6"
                fontSize="12"
              >
                {value.toFixed(0)}
              </text>
            </g>
          );
        })}

        <polygon
          points={`${left},${bottom} ${points} ${getX(
            data.length - 1
          )},${bottom}`}
          fill="url(#iona-rain-fill)"
        />

        <polyline
          points={points}
          fill="none"
          stroke="#61e0c0"
          strokeWidth="3"
          strokeLinecap="round"
          strokeLinejoin="round"
        />

        {data.map((item, index) => {
          const x = getX(index);
          const y = getY(item.precipitation_mm);

          return (
            <g key={item.year}>
              {(index % 4 === 0 || index === data.length - 1) && (
                <text
                  x={x}
                  y={bottom + 27}
                  textAnchor="middle"
                  fill="#9db8c6"
                  fontSize="12"
                >
                  {item.year}
                </text>
              )}

              <circle
                cx={x}
                cy={y}
                r={selected === index ? 6 : 3.5}
                fill="#61e0c0"
                pointerEvents="none"
              />

              <circle
                cx={x}
                cy={y}
                r="14"
                fill="transparent"
                tabIndex={0}
                aria-label={`${item.year}: ${item.precipitation_mm.toFixed(2)} milímetros`}
                onPointerEnter={() => setSelected(index)}
                onFocus={() => setSelected(index)}
                onBlur={() => setSelected(null)}
              />
            </g>
          );
        })}
      </svg>

      <p aria-live="polite" style={{ color: "#9db8c6" }}>
        {active
          ? `${active.year}: ${active.precipitation_mm.toLocaleString(
              "pt-BR",
              { maximumFractionDigits: 2 }
            )} mm`
          : "Passe o mouse sobre os pontos para consultar os valores."}
      </p>
    </div>
  );
}

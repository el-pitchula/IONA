import { useMemo, useState } from "react";
import "./SpatialGrid.css";

export type SpatialFeature = {
  type: "Feature";
  geometry: { type: "Point"; coordinates: [number, number] };
  properties: {
    sen_slope: number;
    q_value: number;
    significant_fdr: boolean;
    queen_cluster: string;
    rook_cluster: string;
  };
};

export type SpatialGeoJSON = {
  type: "FeatureCollection";
  features: SpatialFeature[];
};

type View = "slope" | "queen" | "rook";

const pt = (value: number, digits = 2) =>
  value.toLocaleString("pt-BR", { maximumFractionDigits: digits });

function clusterColor(cluster: string) {
  switch (cluster) {
    case "high-high": return "#e7ae72";
    case "low-low": return "#74b5d4";
    case "high-low": return "#d87f88";
    case "low-high": return "#b3a1ec";
    default: return "#294553";
  }
}

function slopeColor(slope: number) {
  if (!Number.isFinite(slope)) return "#294553";
  if (slope >= 0) return `hsl(24 72% ${Math.max(45, 67 - slope * 2)}%)`;
  return `hsl(196 45% ${Math.max(31, 64 + slope * 1.5)}%)`;
}

export default function SpatialGrid({ features }: { features: SpatialFeature[] }) {
  const [view, setView] = useState<View>("slope");
  const [selected, setSelected] = useState<number | null>(null);

  const cells = useMemo(() => features.filter((f) =>
    f?.geometry?.type === "Point" &&
    Array.isArray(f.geometry.coordinates) &&
    f.geometry.coordinates.length === 2 &&
    Number.isFinite(f.geometry.coordinates[0]) &&
    Number.isFinite(f.geometry.coordinates[1]) &&
    Number.isFinite(f.properties?.sen_slope)
  ), [features]);

  const bounds = useMemo(() => {
    const lons = cells.map((f) => f.geometry.coordinates[0]);
    const lats = cells.map((f) => f.geometry.coordinates[1]);
    return {
      west: Math.min(...lons), east: Math.max(...lons),
      south: Math.min(...lats), north: Math.max(...lats),
    };
  }, [cells]);

  const active = selected === null ? null : cells[selected];

  if (!cells.length) return <p>Sem coordenadas geográficas válidas para exibição.</p>;

  const cellSize = 19;
  const size = 520;
  const inner = size - cellSize - 32;
  const rangeX = Math.max(bounds.east - bounds.west, Number.EPSILON);
  const rangeY = Math.max(bounds.north - bounds.south, Number.EPSILON);

  return (
    <div className="iona-spatial">
      <div className="iona-spatial__toolbar">
        <label htmlFor="iona-map-mode">Camada científica</label>
        <select id="iona-map-mode" value={view} onChange={(event) => {
          setView(event.target.value as View);
          setSelected(null);
        }}>
          <option value="slope">Sen's slope por célula</option>
          <option value="queen">Clusters LISA: Queen</option>
          <option value="rook">Clusters LISA: Rook</option>
        </select>
      </div>
      <div className="iona-spatial__map" role="group" aria-label="Grade geográfica do estudo piloto">
        <span className="iona-spatial__north">N ↑</span>
        <svg viewBox={`0 0 ${size} ${size}`} aria-label="625 células com coordenadas reais do estudo piloto" preserveAspectRatio="xMidYMid meet">
          {cells.map((feature, index) => {
            const [lon, lat] = feature.geometry.coordinates;
            const x = 16 + ((lon - bounds.west) / rangeX) * inner;
            const y = 16 + ((bounds.north - lat) / rangeY) * inner;
            const cluster = view === "queen" ? feature.properties.queen_cluster : feature.properties.rook_cluster;
            const color = view === "slope" ? slopeColor(feature.properties.sen_slope) : clusterColor(cluster);
            const isActive = index === selected;
            return (
              <rect
                key={`${lon}-${lat}`}
                x={x} y={y} width={cellSize - 1} height={cellSize - 1} rx={1}
                fill={color} stroke={isActive ? "#ffffff" : "#0b1a24"}
                strokeWidth={isActive ? 2.5 : 0.9}
                role="button" tabIndex={0}
                aria-label={`Célula ${index + 1}, latitude ${pt(lat, 3)}, longitude ${pt(lon, 3)}, inclinação ${pt(feature.properties.sen_slope)} mm por ano ao quadrado`}
                onClick={() => setSelected(index)}
                onFocus={() => setSelected(index)}
                onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); setSelected(index); } }}
              >
                <title>{`Lat ${pt(lat, 3)}, Lon ${pt(lon, 3)} | Sen ${pt(feature.properties.sen_slope)} mm/ano²`}</title>
              </rect>
            );
          })}
        </svg>
      </div>
      <div className="iona-spatial__legend">
        {view === "slope" ? (
          <><span className="iona-spatial__swatch" style={{ background: "#74b5d4" }} /> Inclinação negativa <span className="iona-spatial__swatch" style={{ background: "#e7ae72" }} /> Inclinação positiva</>
        ) : (
          <><span className="iona-spatial__swatch" style={{ background: "#e7ae72" }} /> High-High <span className="iona-spatial__swatch" style={{ background: "#74b5d4" }} /> Low-Low <span className="iona-spatial__swatch" style={{ background: "#294553" }} /> Não significativo (LISA)</>
        )}
      </div>
      <div className="iona-spatial__detail" aria-live="polite">
        {active ? (
          <>
            <strong>Célula selecionada</strong>
            <span>Lat {pt(active.geometry.coordinates[1], 3)} · Lon {pt(active.geometry.coordinates[0], 3)}</span>
            <span>Sen: {pt(active.properties.sen_slope)} mm/ano²</span>
            <span>Trend FDR: {active.properties.significant_fdr ? "Significativa" : "Não significativa"}</span>
            <span>Queen: {active.properties.queen_cluster} · Rook: {active.properties.rook_cluster}</span>
            <button type="button" className="iona-spatial__clear" onClick={() => setSelected(null)}>Limpar seleção</button>
          </>
        ) : "Selecione uma célula para consultar seus indicadores e coordenadas."}
      </div>
      <p className="iona-spatial__disclaimer">
        Grade de pontos do estudo piloto do DF e entorno. As cores de Sen's slope representam estimativas,
        não tendências temporalmente significativas. Clusters LISA são uma análise espacial distinta.
        Não corresponde à delimitação oficial do bioma Cerrado.
      </p>
    </div>
  );
}

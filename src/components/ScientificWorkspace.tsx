import { useEffect, useMemo, useState, type FormEvent } from "react";
import AnnualPrecipitationPlot from "./AnnualPrecipitationPlot";
import SpatialGrid, { type SpatialGeoJSON } from "./SpatialGrid";
import { navigateToIONASection } from "../ionaNavigation";
import "./ScientificWorkspace.css";

type AnnualRecord = { year: number; precipitation_mm: number };
type AnnualSeries = {
  metadata: { period: string; unit: string; aggregation?: string };
  data: AnnualRecord[];
};
type Overview = {
  study_area: { name: string; is_official_cerrado_polygon: boolean };
  dataset: { period: string; months: number; spatial_cells: number; resolution_degrees: number; doi: string };
  regional_trend: { sen_slope_mm_per_year_per_year: number; conventional_p_value: number; sen_ci_95: [number, number] };
  spatial_results: { conventional_significant_fdr: number; tfpw_significant_fdr: number };
  spatial_autocorrelation: { observed_moran_i: number; p_two_sided: number; variable: string };
  local_spatial_analysis: { significant_cells: number; cluster_counts: Record<string, number> };
  robustness: { queen: { significant_cells_fdr: number }; rook: { significant_cells_fdr: number } };
};
type DataQuality = {
  dataset: { observations: number; missing_observations: number; valid_observations: number };
  source: { provider: string; product: string; version: string; doi: string };
};
type DataBundle = {
  annual: AnnualRecord[];
  overview: Overview;
  quality: DataQuality;
  spatial: SpatialGeoJSON;
};
const fmt = (number: number, digits = 2) => number.toLocaleString("pt-BR", {
  minimumFractionDigits: digits, maximumFractionDigits: digits,
});

async function readJson(path: string, signal: AbortSignal): Promise<unknown> {
  const response = await fetch(`${import.meta.env.BASE_URL}data/${path}`, { signal });
  if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
  return response.json() as Promise<unknown>;
}

function validateData(annual: unknown, overview: unknown, quality: unknown, spatial: unknown): DataBundle {
  const a = annual as AnnualSeries;
  const o = overview as Overview;
  const q = quality as DataQuality;
  const s = spatial as SpatialGeoJSON;
  if (!Array.isArray(a?.data) || !a.data.length || !a.data.every((x) =>
    Number.isInteger(x.year) && Number.isFinite(x.precipitation_mm) && x.precipitation_mm >= 0)) {
    throw new Error("annual_series.json não contém uma série anual válida.");
  }
  if (!o?.dataset?.period || !o?.regional_trend || !o?.spatial_results ||
      !o?.spatial_autocorrelation || !o?.local_spatial_analysis || !o?.robustness) {
    throw new Error("overview.json não possui os resultados científicos esperados.");
  }
  if (!q?.dataset || !q?.source) {
    throw new Error("data_quality.json não possui os metadados esperados.");
  }
  if (s?.type !== "FeatureCollection" || !Array.isArray(s.features) || !s.features.length) {
    throw new Error("spatial_cells.geojson não contém células geográficas válidas.");
  }
  return { annual: [...a.data].sort((x, y) => x.year - y.year), overview: o, quality: q, spatial: s };
}

export default function ScientificWorkspace() {
  const [dataset, setDataset] = useState<DataBundle | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [start, setStart] = useState(2001);
  const [end, setEnd] = useState(2024);
  const [applied, setApplied] = useState({ start: 2001, end: 2024 });
  const [filterError, setFilterError] = useState("");

  useEffect(() => {
    const controller = new AbortController();
    async function load() {
      try {
        const [annual, overview, quality, spatial] = await Promise.all([
          readJson("annual_series.json", controller.signal),
          readJson("overview.json", controller.signal),
          readJson("data_quality.json", controller.signal),
          readJson("spatial_cells.geojson", controller.signal),
        ]);
        const bundle = validateData(annual, overview, quality, spatial);
        if (!controller.signal.aborted) {
          setDataset(bundle);
          setStart(bundle.annual[0].year);
          setEnd(bundle.annual[bundle.annual.length - 1].year);
          setApplied({ start: bundle.annual[0].year, end: bundle.annual[bundle.annual.length - 1].year });
        }
      } catch (caught) {
        if (!controller.signal.aborted) setError(caught instanceof Error ? caught.message : "Falha desconhecida ao carregar os dados.");
      } finally {
        if (!controller.signal.aborted) setLoading(false);
      }
    }
    void load();
    return () => controller.abort();
  }, []);

  // Scroll to deep links after the asynchronous scientific datasets finish loading.
  useEffect(() => {
    if (!dataset) return;
    const id = window.location.hash.slice(1);
    if (!id || id === "overview") return;
    const frame = window.requestAnimationFrame(() => {
      if (document.getElementById(id)) navigateToIONASection(id, false);
    });
    return () => window.cancelAnimationFrame(frame);
  }, [dataset]);

  const filtered = useMemo(() => dataset?.annual.filter((item) =>
    item.year >= applied.start && item.year <= applied.end) ?? [], [dataset, applied]);
  const mean = useMemo(() => filtered.length ?
    filtered.reduce((sum, item) => sum + item.precipitation_mm, 0) / filtered.length : null,
  [filtered]);

  function applyFilter(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!dataset) return;
    if (start > end) {
      setFilterError("O ano inicial não pode ser posterior ao ano final.");
      return;
    }
    setFilterError("");
    setApplied({ start, end });
  }

  function resetFilter() {
    if (!dataset) return;
    const first = dataset.annual[0].year;
    const last = dataset.annual[dataset.annual.length - 1].year;
    setStart(first);
    setEnd(last);
    setApplied({ start: first, end: last });
    setFilterError("");
  }

  if (loading) return <section className="iona-loading" role="status">Carregando séries e dados espaciais do IONA...</section>;
  if (error || !dataset) return <section className="iona-loading" role="alert">Falha ao carregar dados: {error ?? "conjunto indisponível"}. Verifique os arquivos em public/data/.</section>;

  const { overview, quality, spatial } = dataset;
  const years = dataset.annual.map((record) => record.year);
  const pValue = overview.regional_trend.conventional_p_value;
  const significant = pValue < 0.05;
  const noTemporalFdr = overview.spatial_results.conventional_significant_fdr === 0 &&
    overview.spatial_results.tfpw_significant_fdr === 0;

  return (
    <div className="iona-workspace" id="explorer">
      <div className="iona-workspace__heading">
        <div>
          <h2>Painel de investigação</h2>
          <p>Dados NASA GPM IMERG · Região piloto do Distrito Federal e entorno</p>
        </div>
        <span className="iona-workspace__live">DADOS PROCESSADOS · NÃO SIMULADOS</span>
      </div>

      <form className="iona-filter" onSubmit={applyFilter} aria-label="Filtrar série anual">
        <div className="iona-filter__field">
          <span className="iona-filter__label">Variável ambiental</span>
          <div className="iona-filter__readonly" title="Somente precipitação está disponível neste estudo">Precipitação</div>
        </div>
        <div className="iona-filter__field">
          <span className="iona-filter__label">Região de estudo</span>
          <div className="iona-filter__readonly" title="Somente a região piloto foi processada">DF e entorno (área piloto)</div>
        </div>
        <div className="iona-filter__field">
          <label htmlFor="iona-start">Ano inicial</label>
          <select id="iona-start" value={start} onChange={(e) => setStart(Number(e.target.value))}>
            {years.map((year) => <option key={year} value={year}>{year}</option>)}
          </select>
        </div>
        <div className="iona-filter__field">
          <label htmlFor="iona-end">Ano final</label>
          <select id="iona-end" value={end} onChange={(e) => setEnd(Number(e.target.value))}>
            {years.map((year) => <option key={year} value={year}>{year}</option>)}
          </select>
        </div>
        <div className="iona-filter__actions">
          <button type="submit" className="iona-filter__submit">Aplicar período</button>
          <button type="button" className="iona-filter__reset" onClick={resetFilter}>Restaurar 2001–2024</button>
        </div>
        {filterError && <p role="alert" className="iona-filter__error">{filterError}</p>}
        <p className="iona-filter__note" role="status" aria-live="polite">
          Série anual selecionada: {applied.start}–{applied.end}. Os resultados espaciais e a tendência regional continuam referentes ao estudo completo (2001–2024).
        </p>
      </form>

      <div className="iona-stats">
        <article className="iona-stat"><span>Período selecionado</span><strong>{filtered.length} anos</strong><small>{applied.start} a {applied.end} · série anual</small></article>
        <article className="iona-stat"><span>Precipitação média anual</span><strong>{mean === null ? "—" : `${fmt(mean, 1)} mm`}</strong><small>Calculada para o período selecionado</small></article>
        <article className="iona-stat"><span>Grade geográfica</span><strong>{overview.dataset.spatial_cells}</strong><small>Estudo completo, resolução de {overview.dataset.resolution_degrees}°</small></article>
        <article className="iona-stat"><span>Tendência regional · 2001–2024</span><strong className="iona-stat__amber">{fmt(overview.regional_trend.sen_slope_mm_per_year_per_year)} mm/ano²</strong><small>{significant ? "Significativa" : "Não significativa"} · p = {fmt(pValue, 3)}</small></article>
      </div>

      <div className="iona-panels">
        <section className="iona-science-panel" id="trends" aria-labelledby="iona-trends-title">
          <div className="iona-science-panel__header"><div><h3 id="iona-trends-title">Série temporal de precipitação</h3><p>Acumulados anuais regionais · NASA GPM</p></div><span>ANNUAL SERIES</span></div>
          <AnnualPrecipitationPlot records={filtered} />
          <details className="iona-science-data"><summary>Consultar tabela de valores anuais</summary>
            <div className="iona-science-data__scroll"><table><thead><tr><th>Ano</th><th>Precipitação (mm)</th></tr></thead><tbody>
              {filtered.map((record) => <tr key={record.year}><td>{record.year}</td><td>{fmt(record.precipitation_mm)}</td></tr>)}
            </tbody></table></div>
          </details>
        </section>
        <section className="iona-science-panel" id="patterns" aria-labelledby="iona-patterns-title">
          <div className="iona-science-panel__header"><div><h3 id="iona-patterns-title">Distribuição espacial</h3><p>Tendências estimadas e clusters locais · estudo completo</p></div><span>SPATIAL GRID</span></div>
          <SpatialGrid features={spatial.features} />
        </section>
      </div>

      <div className="iona-evidence" id="evidence">
        <section className="iona-science-panel" aria-labelledby="iona-evidence-title">
          <div className="iona-science-panel__header"><div><h3 id="iona-evidence-title">Evidências estatísticas</h3><p>Resultados já calculados nos scripts científicos</p></div><span>VALIDAÇÃO</span></div>
          <div className="iona-evidence__row"><span>Sen's slope regional (2001–2024)</span><strong>{fmt(overview.regional_trend.sen_slope_mm_per_year_per_year)} mm/ano²</strong></div>
          <div className="iona-evidence__row"><span>Mann–Kendall regional</span><strong>p = {fmt(pValue, 4)} · {significant ? "significativo" : "não significativo"}</strong></div>
          <div className="iona-evidence__row"><span>Global Moran's I, Sen's slopes</span><strong>{fmt(overview.spatial_autocorrelation.observed_moran_i, 3)} · p = {fmt(overview.spatial_autocorrelation.p_two_sided, 4)}</strong></div>
          <div className="iona-evidence__row"><span>Clusters LISA com FDR (Queen)</span><strong>{overview.robustness.queen.significant_cells_fdr} células</strong></div>
          <div className="iona-evidence__row"><span>Clusters LISA com FDR (Rook)</span><strong>{overview.robustness.rook.significant_cells_fdr} células</strong></div>
          <div className="iona-evidence__row"><span>Tendências temporais significativas (FDR)</span><strong>{overview.spatial_results.conventional_significant_fdr} células</strong></div>
          <p className="iona-evidence__warning">
            {noTemporalFdr ? "Nenhuma das 625 células apresentou tendência temporal significativa após correção FDR nos testes comparados. " : "Consulte a metodologia de correção FDR. "}
            Moran's I e os agrupamentos LISA não demonstram causalidade nem significância temporal.
            Os indicadores espaciais se referem ao estudo completo (2001–2024), independentemente do filtro da série anual.
          </p>
        </section>
        <section className="iona-science-panel" id="methodology" aria-labelledby="iona-method-title">
          <div className="iona-science-panel__header"><div><h3 id="iona-method-title">Metodologia e limitações</h3><p>Transparência e interpretação responsável</p></div><span>METHODS</span></div>
          <p className="iona-method-text">A série anual agrega precipitação mensal com ponderação pelo cosseno da latitude. As tendências são estimadas com Sen's slope e testadas com Mann–Kendall. A correção Benjamini–Hochberg controla falsas descobertas em múltiplos testes espaciais. Moran's I e LISA descrevem estrutura espacial das estimativas.</p>
          <p className="iona-method-text">O recorte é um retângulo piloto do Distrito Federal e entorno, não a fronteira oficial do Cerrado. Alterar anos no gráfico não recalcula tendências ou clusters: isso exigiria executar novamente os procedimentos científicos.</p>
          <p className="iona-method-text">A visualização cartográfica usa centros das células fornecidos pelo GeoJSON; não representa limites administrativos nem inferências causais.</p>
        </section>
      </div>

      <footer className="iona-sources" id="sources">
        <div><strong>Fontes dos dados</strong><p>{quality.source.provider} · {quality.source.product} · {quality.source.version}</p></div>
        <div><strong>Qualidade</strong><p>{quality.dataset.valid_observations.toLocaleString("pt-BR")} observações válidas · {quality.dataset.missing_observations} ausências reportadas</p></div>
        <a href={`https://doi.org/${quality.source.doi}`} target="_blank" rel="noopener noreferrer">DOI do conjunto de dados</a>
      </footer>
    </div>
  );
}

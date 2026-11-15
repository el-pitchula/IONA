
import { useEffect, useState } from "react";

type AnnualRecord = {
  year: number;
  precipitation_mm: number;
};

type AnnualDataset = {
  metadata: {
    period: string;
    unit: string;
  };
  data: AnnualRecord[];
};

export default function PrecipitationChart() {
  const [records, setRecords] = useState<AnnualRecord[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const controller = new AbortController();

    async function loadData() {
      try {
        const response = await fetch(
          `${import.meta.env.BASE_URL}data/annual_series.json`,
          { signal: controller.signal }
        );

        if (!response.ok) {
          throw new Error("Não foi possível carregar os dados.");
        }

        const result: AnnualDataset = await response.json();

        if (!Array.isArray(result.data)) {
          throw new Error("Formato de dados inválido.");
        }

        setRecords(result.data);
      } catch (err) {
        if (controller.signal.aborted) return;
        setError(
          err instanceof Error ? err.message : "Erro desconhecido."
        );
      } finally {
        if (!controller.signal.aborted) setLoading(false);
      }
    }

    loadData();
    return () => controller.abort();
  }, []);

  if (loading) return <p>Carregando dados científicos...</p>;
  if (error) return <p role="alert">{error}</p>;

  return (
    <section aria-labelledby="precipitation-title">
      <h2 id="precipitation-title">
        Precipitação anual | NASA GPM
      </h2>
      <p>
        Série histórica regional de 2001 a 2024,
        expressa em milímetros por ano.
      </p>

      <div style={{ overflowX: "auto" }}>
        <table>
          <thead>
            <tr>
              <th scope="col">Ano</th>
              <th scope="col">Precipitação (mm)</th>
            </tr>
          </thead>
          <tbody>
            {records.map((record) => (
              <tr key={record.year}>
                <td>{record.year}</td>
                <td>
                  {record.precipitation_mm.toLocaleString("pt-BR", {
                    minimumFractionDigits: 2,
                    maximumFractionDigits: 2,
                  })}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

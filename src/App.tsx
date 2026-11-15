import IONASidebar from "./components/IONASidebar";
import OrbitalHero from "./components/OrbitalHero";
import ScientificWorkspace from "./components/ScientificWorkspace";
import "./App.css";
import "./AppShellFinal.css";

function App() {
  return (
    <div className="iona-app-shell">
      <IONASidebar />
      <main className="iona-app-main">
        <header className="iona-topbar">
          <div className="iona-breadcrumb">IONA / <strong>PAINEL DE EXPLORAÇÃO</strong></div>
          <div className="iona-topbar-badges">
            <span className="iona-environment">MVP · ESTUDO PILOTO</span>
            <span className="iona-data-source">NASA GPM IMERG</span>
          </div>
        </header>
        <div id="overview" className="iona-overview"><OrbitalHero /></div>
        <ScientificWorkspace />
      </main>
    </div>
  );
}

export default App;

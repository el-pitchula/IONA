import { useEffect, useState, type MouseEvent } from "react";
import { navigateToIONASection } from "../ionaNavigation";
import "./Sidebar.css";

type Item = { id: string; label: string; icon: string };
const groups: { title: string; items: Item[] }[] = [
  { title: "PLATAFORMA", items: [
    { id: "overview", icon: "◫", label: "Visão geral" },
    { id: "explorer", icon: "◎", label: "Explorador" },
    { id: "trends", icon: "⌁", label: "Laboratório de tendências" },
    { id: "patterns", icon: "▦", label: "Padrões espaciais" },
    { id: "evidence", icon: "✳", label: "Evidências" },
  ] },
  { title: "PESQUISA", items: [
    { id: "methodology", icon: "∑", label: "Metodologia" },
    { id: "sources", icon: "▫", label: "Fontes dos dados" },
  ] },
];
const allowedIds = new Set(groups.flatMap((group) => group.items.map((item) => item.id)));

function currentSection() {
  const id = window.location.hash.replace(/^#/, "");
  return allowedIds.has(id) ? id : "overview";
}

export default function IONASidebar() {
  const [active, setActive] = useState(currentSection);
  const [mobileOpen, setMobileOpen] = useState(false);

  useEffect(() => {
    const sync = () => setActive(currentSection());
    window.addEventListener("hashchange", sync);
    window.addEventListener("popstate", sync);
    window.addEventListener("iona:navigate", sync);
    return () => {
      window.removeEventListener("hashchange", sync);
      window.removeEventListener("popstate", sync);
      window.removeEventListener("iona:navigate", sync);
    };
  }, []);

  useEffect(() => {
    if (!mobileOpen) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setMobileOpen(false);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [mobileOpen]);

  function handleNavigation(event: MouseEvent<HTMLAnchorElement>, id: string) {
    // Native anchor remains a fallback when the asynchronous workspace is loading.
    if (navigateToIONASection(id)) event.preventDefault();
    setActive(id);
    setMobileOpen(false);
  }

  return (
    <>
      <button
        type="button"
        className="iona-mobile-menu"
        aria-controls="iona-navigation"
        aria-expanded={mobileOpen}
        aria-label={mobileOpen ? "Fechar navegação" : "Abrir navegação"}
        onClick={() => setMobileOpen((open) => !open)}
      >☰</button>
      {mobileOpen && <button type="button" className="iona-sidebar-overlay" aria-label="Fechar navegação" onClick={() => setMobileOpen(false)} />}
      <aside id="iona-navigation" className={`iona-sidebar ${mobileOpen ? "is-open" : ""}`} aria-label="Navegação principal">
        <div className="iona-sidebar__brand">
          <div className="iona-sidebar__logo" aria-hidden="true">I</div>
          <div><strong>IONA</strong><small>EARTH INTELLIGENCE</small></div>
          <button type="button" className="iona-nav-close" aria-label="Fechar menu" onClick={() => setMobileOpen(false)}>×</button>
        </div>
        <nav aria-label="Seções do painel">
          {groups.map((group) => <div className="iona-sidebar__group" key={group.title}>
            <p className="iona-sidebar__heading">{group.title}</p>
            {group.items.map((item) => (
              <a
                key={item.id}
                href={`#${item.id}`}
                className={`iona-sidebar__link ${active === item.id ? "is-active" : ""}`}
                aria-current={active === item.id ? "location" : undefined}
                onClick={(event) => handleNavigation(event, item.id)}
              >
                <span className="iona-sidebar__icon" aria-hidden="true">{item.icon}</span>{item.label}
              </a>
            ))}
          </div>)}
        </nav>
        <div className="iona-sidebar__footer">
          <span className="iona-sidebar__status"><span className="iona-sidebar__dot" />MVP CIENTÍFICO EM DESENVOLVIMENTO</span>
          <small>NASA GPM IMERG V07B</small><small>Estudo piloto DF e entorno</small>
        </div>
      </aside>
    </>
  );
}

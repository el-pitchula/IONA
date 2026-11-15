
import { useState } from "react";
import "./Sidebar.css";

const navigation = [
  {
    heading: "PLATAFORMA",
    items: [
      { id: "overview", icon: "◫", label: "Visão geral" },
      { id: "explorer", icon: "◎", label: "Explorador" },
      { id: "trends", icon: "⌁", label: "Laboratório de tendências" },
      { id: "patterns", icon: "▦", label: "Padrões espaciais" },
      { id: "evidence", icon: "✳", label: "Evidências" },
    ],
  },
  {
    heading: "PESQUISA",
    items: [
      { id: "methodology", icon: "∑", label: "Metodologia" },
      { id: "sources", icon: "▫", label: "Fontes dos dados" },
    ],
  },
];

export default function Sidebar() {
  const [active, setActive] = useState("overview");
  const [mobileOpen, setMobileOpen] = useState(false);

  return (
    <>
      <button
        className="iona-mobile-menu"
        type="button"
        aria-label="Abrir menu"
        aria-expanded={mobileOpen}
        onClick={() => setMobileOpen(!mobileOpen)}
      >
        ☰
      </button>

      {mobileOpen && (
        <button
          className="iona-sidebar-overlay"
          aria-label="Fechar menu"
          onClick={() => setMobileOpen(false)}
        />
      )}

      <aside
        className={`iona-sidebar ${mobileOpen ? "is-open" : ""}`}
        aria-label="Navegação principal"
      >
        <div className="iona-sidebar__brand">
          <div className="iona-sidebar__logo">I</div>
          <div>
            <strong>IONA</strong>
            <small>EARTH INTELLIGENCE</small>
          </div>
        </div>

        <nav>
          {navigation.map((group) => (
            <div className="iona-sidebar__group" key={group.heading}>
              <p className="iona-sidebar__heading">{group.heading}</p>

              {group.items.map((item) => (
                <button
                  type="button"
                  key={item.id}
                  className={`iona-sidebar__link ${
                    active === item.id ? "is-active" : ""
                  }`}
                  onClick={() => {
                    setActive(item.id);
                    setMobileOpen(false);
                  }}
                >
                  <span className="iona-sidebar__icon">
                    {item.icon}
                  </span>
                  {item.label}
                </button>
              ))}
            </div>
          ))}
        </nav>

        <div className="iona-sidebar__footer">
          <span className="iona-sidebar__status">
            <span className="iona-sidebar__dot" />
            PLATAFORMA EM DESENVOLVIMENTO
          </span>
          <small>NASA GPM IMERG</small>
          <small>IONA / EARTH INTELLIGENCE</small>
        </div>
      </aside>
    </>
  );
}

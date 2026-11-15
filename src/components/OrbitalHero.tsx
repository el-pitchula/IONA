
import "./OrbitalHero.css";
import { navigateToIONASection } from "../ionaNavigation";

export default function OrbitalHero() {
  return (
    <section className="iona-hero" aria-labelledby="iona-hero-title">
      <div className="iona-hero__content">
        <span className="iona-hero__eyebrow">
          EARTH INTELLIGENCE / PATTERN DISCOVERY
        </span>

        <h1 id="iona-hero-title">
          Descubra o que as
          <span> médias escondem.</span>
        </h1>

        <p>
          Investigue padrões de precipitação,
          contrastes espaciais e mudanças ambientais
          utilizando dados científicos da NASA.
        </p>

        <a
          className="iona-hero__button"
          href="#trends"
          onClick={(event) => { if (navigateToIONASection("trends")) event.preventDefault(); }}
        >
          Explorar tendências
        </a>
      </div>

      <div className="iona-orbital" aria-hidden="true">
        <div className="iona-orbital__glow" />

        <div className="iona-orbital__orbit iona-orbital__orbit--one">
          <span className="iona-orbital__satellite" />
        </div>

        <div className="iona-orbital__orbit iona-orbital__orbit--two">
          <span className="iona-orbital__particle" />
        </div>

        <div className="iona-orbital__orbit iona-orbital__orbit--three">
          <span className="iona-orbital__particle" />
        </div>

        <div className="iona-orbital__planet">
          <div className="iona-orbital__continents" />
          <div className="iona-orbital__atmosphere" />
        </div>
      </div>
    </section>
  );
}

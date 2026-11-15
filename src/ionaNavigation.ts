/** Navegação interna do dashboard. Não depende de biblioteca de rotas. */
export function navigateToIONASection(id: string, updateHistory = true): boolean {
  const target = document.getElementById(id);
  if (!target) return false;

  if (updateHistory && window.location.hash !== `#${id}`) {
    window.history.pushState(null, "", `#${id}`);
  }
  // pushState não dispara hashchange; sincroniza o menu, inclusive com o botão do hero.
  window.dispatchEvent(new Event("iona:navigate"));
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  target.scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "start" });
  if (!target.hasAttribute("tabindex")) target.setAttribute("tabindex", "-1");
  target.focus({ preventScroll: true });
  return true;
}

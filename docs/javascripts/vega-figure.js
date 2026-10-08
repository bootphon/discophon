// Renders a documentation figure (the Vega-Lite spec in window.spec, see scripts/interactive_plots.py) with the colors
// and font of the page that embeds it in an iframe, and renders it again when the page switches between light and
// dark mode. Opened on its own, the figure uses the light colors defined in vega.css.
(function () {
  const VARIABLES = [
    "--md-default-fg-color",
    "--md-default-fg-color--light",
    "--md-default-fg-color--lighter",
    "--md-default-fg-color--lightest",
    "--md-default-bg-color",
    "--md-primary-fg-color",
    "--md-primary-bg-color",
  ];
  const FONT = 'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif';

  function parentBody() {
    try {
      return window.parent !== window ? window.parent.document.body : null;
    } catch {
      return null;
    }
  }

  // Resolve a CSS variable to a color that the canvas understands.
  function color(name) {
    const probe = document.createElement("span");
    probe.style.color = `var(${name})`;
    document.body.append(probe);
    const value = getComputedStyle(probe).color;
    probe.remove();
    return value;
  }

  function render() {
    const body = parentBody();
    const style = body && getComputedStyle(body);
    for (const name of VARIABLES) {
      const value = style && style.getPropertyValue(name).trim();
      if (value) document.documentElement.style.setProperty(name, value);
    }
    // The browser paints an opaque background behind an iframe whose color scheme differs from its parent.
    if (style) document.documentElement.style.colorScheme = style.colorScheme;
    const [fg, light, lighter, lightest] = ["", "--light", "--lighter", "--lightest"].map((s) =>
      color(`--md-default-fg-color${s}`)
    );
    const theme = {
      background: "transparent",
      font: FONT,
      view: { stroke: null },
      title: { color: fg, fontSize: 14, fontWeight: 500 },
      axis: {
        labelColor: light,
        titleColor: light,
        titleFontWeight: 500,
        titleFontSize: 12,
        labelFontSize: 11,
        gridColor: lightest,
        domainColor: lighter,
        tickColor: lighter,
      },
      legend: { labelColor: fg, titleColor: light, titleFontWeight: 500, labelFontSize: 12, titleFontSize: 12 },
    };
    const config = { ...window.spec.config };
    for (const [key, value] of Object.entries(theme)) {
      config[key] = typeof value === "object" && value !== null ? { ...config[key], ...value } : value;
    }
    vegaEmbed("#vis", { ...window.spec, config }, { actions: false, mode: "vega-lite" });
  }

  Promise.all(["400 12px Inter", "500 12px Inter"].map((font) => document.fonts.load(font))).then(render, render);
  const body = parentBody();
  if (body) new MutationObserver(render).observe(body, { attributes: true, attributeFilter: ["data-md-color-scheme"] });
})();

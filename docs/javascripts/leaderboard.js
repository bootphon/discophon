(function () {
  function init(root) {
    // The payload lives in a data attribute rather than a <script type="application/json">
    // because zensical's instant navigation re-executes inline scripts when it swaps the
    // page in, which would try to run the JSON as JavaScript and drop it. Attributes are
    // copied verbatim and never executed, so the data survives instant navigation.
    const data = JSON.parse(root.dataset.dlbPayload);
    const splitSelect = root.querySelector('[data-control="split"]');
    const langSelect = root.querySelector('[data-control="language"]');
    const tbody = root.querySelector("tbody");

    function populateLanguages(split) {
      const current = langSelect.value;
      langSelect.replaceChildren();
      const avg = document.createElement("option");
      avg.value = data.avg_key;
      avg.textContent = "Average over " + split + " languages";
      langSelect.appendChild(avg);
      for (const lang of data.splits[split]) {
        const opt = document.createElement("option");
        opt.value = lang.iso;
        opt.textContent = lang.name;
        langSelect.appendChild(opt);
      }
      const stillThere = Array.from(langSelect.options).some((o) => o.value === current);
      langSelect.value = stillThere ? current : data.avg_key;
    }

    function setCell(td, value, std, showStd) {
      if (value === null || value === undefined) {
        td.textContent = "N/A";
        return;
      }
      td.textContent = value.toFixed(2);
      if (showStd && std !== null && std !== undefined) {
        td.appendChild(document.createTextNode(" "));
        const span = document.createElement("span");
        span.className = "dlb-std";
        span.textContent = "±" + std.toFixed(2);
        td.appendChild(span);
      }
    }

    function render() {
      const split = splitSelect.value;
      const language = langSelect.value;
      const isAvg = language === data.avg_key;
      const rows = data.rows.filter((r) => r.split === split && r.language === language);
      tbody.replaceChildren();
      for (const dur of data.durations) {
        const header = document.createElement("tr");
        header.className = "dlb-group";
        const th = document.createElement("th");
        th.colSpan = 1 + data.metrics.length;
        th.textContent = dur.label;
        header.appendChild(th);
        tbody.appendChild(header);
        for (const category of data.categories) {
          const subheader = document.createElement("tr");
          subheader.className = "dlb-subgroup";
          const subth = document.createElement("th");
          subth.colSpan = 1 + data.metrics.length;
          subth.textContent = category.label;
          subheader.appendChild(subth);
          tbody.appendChild(subheader);
          for (const model of data.models) {
            if (model.category !== category.key) continue;
            // Skip models with no data at all for this split+duration (e.g. a model
            // only evaluated on test languages is hidden in the dev-split view).
            const hasData = data.rows.some(
              (r) => r.model === model.key && r.duration === dur.key && r.split === split
            );
            if (!hasData) continue;
            // row may be undefined when the model was not evaluated on the current
            // language; we still render the row but leave metric cells empty.
            const row = rows.find((r) => r.model === model.key && r.duration === dur.key);
            const tr = document.createElement("tr");
            const name = document.createElement("td");
            const link = document.createElement("a");
            // Baselines link out to their HuggingFace checkpoint; submissions link to an
            // in-page section (#model-...). External links open in a new tab and are left
            // for the browser to follow (the in-page click handler below ignores them).
            link.href = model.href;
            if (model.external) {
              link.target = "_blank";
              link.rel = "noopener";
            }
            link.textContent = model.label;
            name.appendChild(link);
            if (row) {
              name.appendChild(document.createTextNode(" (L" + row.layer + ")"));
            }
            tr.appendChild(name);
            for (const metric of data.metrics) {
              const td = document.createElement("td");
              if (row) {
                const value = row.scores[metric.key];
                setCell(td, value, row.stds[metric.key], isAvg);
                if (value !== null && row.top[metric.key]) td.classList.add("dlb-top");
              }
              tr.appendChild(td);
            }
            tbody.appendChild(tr);
          }
        }
      }
    }

    // In-page model-name links (submissions, #model-...) jump to a section on the page;
    // the href^="#" filter leaves external baseline checkpoint links to the browser. We
    // scroll to the target ourselves rather than letting a hash change happen, because zensical re-fetches
    // and re-renders the page on the first hash change after a hard load (its location$
    // has no seeded pathname yet, so its "pathname unchanged" guard cannot fire) — which
    // would reset this table. preventDefault blocks the native anchor's hash change and
    // stopPropagation blocks zensical's document.body click handler; replaceState updates
    // the address bar without dispatching an event.
    root.addEventListener("click", (event) => {
      const link = event.target.closest('a[href^="#"]');
      if (!link || !root.contains(link)) return;
      const target = document.getElementById(decodeURIComponent(link.getAttribute("href").slice(1)));
      if (!target) return;
      event.preventDefault();
      event.stopPropagation();
      history.replaceState(history.state, "", link.getAttribute("href"));
      target.scrollIntoView({ behavior: "smooth" });
    });

    splitSelect.addEventListener("change", () => {
      populateLanguages(splitSelect.value);
      render();
    });
    langSelect.addEventListener("change", render);

    populateLanguages(splitSelect.value);
    render();
  }

  // zensical's document$ is a ReplaySubject(1): subscribing initialises the table on
  // first load (the current document is replayed, so there is no race and no need to
  // wait for DOMContentLoaded) and again after every instant navigation, when the page
  // content is swapped in. The guard stops a root being initialised twice. Model-name
  // links (#model-...) keep the pathname, so instant navigation scrolls to them
  // without re-fetching the page, leaving the rendered table untouched.
  document$.subscribe(() => {
    document.querySelectorAll("[data-discophon-leaderboard]").forEach((root) => {
      if (root.dataset.dlbReady) return;
      root.dataset.dlbReady = "1";
      init(root);
    });
  });
})();

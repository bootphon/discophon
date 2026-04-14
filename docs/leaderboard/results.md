# Additional results

We provide below the complete results for the baseline models:
every layer, metric, language, and finetuning duration.

## Across layers

<iframe
  id="vegaFrame"
  title="Baseline results across layers"
  style="border: none; width: 100%;"
  src="../../assets/baseline_across_layers_by_split.html"
  onload="
    var f = this;
    var resize = function() { f.style.height = f.contentDocument.body.scrollHeight + 'px'; };
    new ResizeObserver(resize).observe(f.contentDocument.body);
  ">
</iframe>

## Across layers, for a specific language

<iframe
  id="vegaFrame"
  title="Baseline results across layers, for a specific language"
  style="border: none; width: 100%;"
  src="../../assets/baseline_across_layers_by_lang.html"
  onload="
    var f = this;
    var resize = function() { f.style.height = f.contentDocument.body.scrollHeight + 'px'; };
    new ResizeObserver(resize).observe(f.contentDocument.body);
  ">
</iframe>

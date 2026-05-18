# Detailed results

We provide below the complete results for the baseline models:
every layer, metric, language, and finetuning duration.

## Across layers

The chart below shows the scores averaged across dev or test languages, for the four baseline models, every layer,
and finetuning duration:

<iframe
  title="Baseline results across layers"
  style="border: none; width: 100%;"
  src="../assets/baseline_across_layers_by_split.html"
  onload="
    var f = this;
    var resize = function() { f.style.height = f.contentDocument.body.scrollHeight + 'px'; };
    new ResizeObserver(resize).observe(f.contentDocument.body);
  ">
</iframe>

And this one for a specific language:

<iframe
  title="Baseline results across layers, for a specific language"
  style="border: none; width: 100%;"
  src="../assets/baseline_across_layers_by_lang.html"
  onload="
    var f = this;
    var resize = function() { f.style.height = f.contentDocument.body.scrollHeight + 'px'; };
    new ResizeObserver(resize).observe(f.contentDocument.body);
  ">
</iframe>

## Best layer, by finetuning duration

This one only displays the scores for the best layer, averaged across dev or test languages:

<iframe
  title="Baseline results for the best layer, by finetuning duration"
  style="border: none; width: 100%;"
  src="../assets/baseline_best_layer_by_ft_by_split.html"
  onload="
    var f = this;
    var resize = function() { f.style.height = f.contentDocument.body.scrollHeight + 'px'; };
    new ResizeObserver(resize).observe(f.contentDocument.body);
  ">
</iframe>

And this one for a specific language:

<iframe
  title="Baseline results for the best layer, by finetuning duration, for a specific language"
  style="border: none; width: 100%;"
  src="../assets/baseline_best_layer_by_ft_by_lang.html"
  onload="
    var f = this;
    var resize = function() { f.style.height = f.contentDocument.body.scrollHeight + 'px'; };
    new ResizeObserver(resize).observe(f.contentDocument.body);
  ">
</iframe>

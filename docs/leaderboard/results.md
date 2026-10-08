# Detailed results

We provide below the complete results for the baseline models: every layer, metric, language, and finetuning
duration, on the test split. Select a language, or the average over the dev or test languages.
The units and scores are available in the [artifacts dataset](https://huggingface.co/datasets/coml/discophon-artifacts).

## Across layers

<iframe
  title="Baseline results across layers"
  style="border: none; width: 100%;"
  src="../assets/baseline_across_layers.html"
  onload="
    var f = this;
    var resize = function() { f.style.height = f.contentDocument.body.scrollHeight + 'px'; };
    new ResizeObserver(resize).observe(f.contentDocument.body);
  ">
</iframe>

## Best layer, by finetuning duration

The best layer of each model and finetuning duration minimizes the continuous ABX on dev languages.

<iframe
  title="Baseline results for the best layer, by finetuning duration"
  style="border: none; width: 100%;"
  src="../assets/baseline_best_layer.html"
  onload="
    var f = this;
    var resize = function() { f.style.height = f.contentDocument.body.scrollHeight + 'px'; };
    new ResizeObserver(resize).observe(f.contentDocument.body);
  ">
</iframe>

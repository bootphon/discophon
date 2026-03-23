# Benchmark

<figure markdown="span" style="width: 90%">
  ![ok](../assets/streams.svg){ width=100% }<figcaption style="text-align: justify;">
  **Overview of the streams in the evaluation pipeline**. The model maps a waveform to units $\bm{u}$,
  evaluated against the gold phones $\bm{p}$ via $\textnormal{PNMI}(\bm{p}, \bm{u})$.
  The assignment $\bm{a}$ is assessed for recognition with $\textnormal{PER}(\bm{p}, \bm{a})$ and
  segmentation with $R\textnormal{-value}(\bm{p}, \bm{a})$ and $F_1(\bm{p}, \bm{a})$.
  </figcaption>
</figure>

- **Languages under consideration**:
    - dev languages: German, Swahili, Tamil, Thai, Turkish, Ukrainian
    - test languages: Basque, English, French, Japanese, Mandarin Chinese, Wolof
- **Evaluation metrics**:
    - Units quality: PNMI
    - Recognition: Phone Error Rate
    - Segmentation: $F_1$, $R$-value
    - Discriminability (optional): ABX discrete and continuous
- **Tracks**
    - Many-to-one (256 units)
    - One-to-one (number of phonemes + 1 units)

# Benchmark

<figure markdown="span" style="width: 90%">
  ![Streams in the evaluation pipeline](../assets/streams.svg){ width=100% }<figcaption style="text-align: justify;">
  **Overview of the streams in the evaluation pipeline**. The model maps a waveform to units $\bm{u}$,
  evaluated against the gold phones $\bm{p}$ via $\textnormal{PNMI}(\bm{p}, \bm{u})$.
  The assignment $\bm{a}$ is assessed for recognition with $\textnormal{PER}(\bm{p}, \bm{a})$ and
  segmentation with $R\textnormal{-value}(\bm{p}, \bm{a})$ and $F_1(\bm{p}, \bm{a})$.
  </figcaption>
</figure>

The benchmark covers 12 languages chosen to span a wide range of phonemic contrasts, split into dev languages for
tuning and test languages for final evaluation. Systems are given 10 hours of unannotated speech and must produce
discrete units that can be mapped to the language's phoneme inventory, either many-to-one (with 256 units), or one-to-one
(with as many units as phonemes).

- **Languages**:
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

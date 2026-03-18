# Datasets

## Description

DiscoPhon includes 6 languages (German, Swahili, Tamil, Thai, Turkish, Ukrainian)
and 6 test languages (Basque, English, French, Japanese, Mandarin Chinese, Wolof),
selected to span a diverse range of phonemic categories, as shown in [this page](./phonemes.md),
while ensuring sufficient data availability.
All data consists of read speech, either from audiobooks or from sentence-level prompts.
For each language, we provide 10 hours of training data, along with 2 hours dev and test splits
annotated with phone-level automatic transcriptions. Additional 10 minutes and 1 hour train splits
are included to study learning under more constrained conditions.
Speakers do not overlap across splits, and each evaluation set is gender-balanced with 10 male and
10 female speakers and in uniform quantities, when possible.

German, English, French, and Wolof data come from the ZeroSpeech 2017 challenge, with Wolof extended
using the original dataset. German, English, and French are based on LibriVox audiobooks,
from which we selected 10 male and 10 female speakers per train set.
We use the original Kaldi-based alignments for all four languages.
The remaining languages are sourced from Common Voice, which involves a much larger number of
speakers per language since contributors read individual sentences rather than full book chapters.
For these, we use alignments from the VoxCommunis project, produced with the Montreal Forced Aligner,
and simplify some specific phonetic notations by folding them back to a single IPA symbol representing
the underlying constrastive category.

We distribute annotations and splits for all languages, along with audio data for the ZeroSpeech languages.
Audio for CommonVoice languages must be downloaded separately.

See [this page](../docs/prepare.md) to download and pre-process data.

## Statistics

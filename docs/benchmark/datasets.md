# Datasets

## Description

DiscoPhon includes 6 languages (German, Swahili, Tamil, Thai, Turkish, Ukrainian)
and 6 test languages (Basque, English, French, Japanese, Mandarin Chinese, Wolof),
spanning a wide range of phonemic categories, as shown in [this page](./phonemes.md).

All data consists of read speech, either from audiobooks or from sentence-level prompts.
For each language, we provide **10 hours of training data**, along with **2 hours dev and test splits**
annotated with phone-level automatic transcriptions. Additional 10 minutes and 1 hour train splits
are included to study learning under more constrained conditions.
Speakers do not overlap across splits, and each evaluation set is gender-balanced with 10 male and
10 female speakers and in uniform quantities, when possible.

German, English, French, and Wolof data come from the ZeroSpeech 2017 challenge, with Wolof extended
using the original dataset. German, English, and French are based on LibriVox audiobooks,
from which we selected 10 male and 10 female speakers per train set.
The remaining languages are sourced from Common Voice, using alignments from the VoxCommunis project. We simplified
some specific phonetic notations by folding them back to a single IPA symbol representing
the underlying constrastive category.

See [this page](../guide/prepare.md) to download and pre-process data. You will:

- Download the benchmark assets, and the audio files for English, French, German, and Wolof.
- Download Common Voice datasets from their website for the other languages.
- Pre-process Common Voice data.

## Statistics

Coming soon!

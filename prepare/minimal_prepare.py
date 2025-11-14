import csv
import itertools
import shutil
import time
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import panphon
import soundfile as sf
import textgrids
from tqdm import tqdm


ALIGNMENT_FREQ = 100  # in Hz
SAMPLE_RATE = 16_000
SILENCE = "SIL"
_DIACRITICS_TO_FIX = {
    "pa-IN": {"diacritic": "ʰ", "reverse": True},
    "ml": {"diacritic": "ʱ", "reverse": False},
    "mt": {"diacritic": "ˤː", "reverse": False},
    "ja": {"diacritic": "ː", "reverse": False},
}
MAX_SEGMENT_DURATION = 0.5


class MontrealForcedAligner:
    MFA_TO_IPA: dict[str, str] = {"g": "ɡ", "ɚ": "ə˞", "ʡ": "ʔ"}

    @classmethod
    def convert_to_ipa(cls, montreal_phones: list[str] | str) -> list[str]:
        if isinstance(montreal_phones, str):
            montreal_phones = montreal_phones.split(" ")
        return [cls.MFA_TO_IPA.get(phone, phone) for phone in montreal_phones]


def _fix_phone_tier(phones: textgrids.Tier, diacritic: str, reverse: bool = False) -> tuple[list[str], list[float]]:
    segments, durations = [], []
    for k, phn in enumerate(phones):
        if phn.text == "spn":
            return [], []
        elif phn.text == diacritic:
            prev_seg = segments[-1]
            segments[-1] = prev_seg[:-1] + phn.text + prev_seg[-1] if reverse else prev_seg + phn.text
            durations[-1] += phn.xmax - phn.xmin
        else:
            segments.append(SILENCE if phn.text == "" else phn.text)
            durations.append(phn.xmax - phn.xmin)
    return segments, durations


def convert_alignment(
    ft: panphon.FeatureTable, path: Path, diacritic: Optional[str] = None, reverse: bool = False
) -> str:
    if not path.exists():
        return ""

    grid = textgrids.TextGrid(path)

    # adapt the key for Urdu's alignments
    phones_key = "phones" if "phones" in grid else "sentence - phones"
    if diacritic is not None:
        # fix some diacritic
        segments, durations = _fix_phone_tier(grid[phones_key], diacritic, reverse)
        if not segments:
            return ""
    else:
        # no diacritic issue
        segments, durations = [], []
        for phn in grid[phones_key]:
            if phn.text == "spn":
                return ""
            else:
                segments.append(SILENCE if phn.text == "" else phn.text)
                durations.append(phn.xmax - phn.xmin)

    # skip files with extremely long segment(s)
    for seg, dur in zip(segments, durations):
        if seg != SILENCE and dur > MAX_SEGMENT_DURATION:
            return ""

    # translate the non-IPA-compatible MFA segments
    segments = MontrealForcedAligner.convert_to_ipa(segments)

    # ensure that all the segments exist in PanPhon
    unique_segments = set(segments) - {SILENCE}
    for seg in unique_segments:
        if not ft.word_to_vector_list(seg):
            return ""

    repeated_segments = []
    for seg, dur in zip(segments, durations):
        # use `round` instead of `int` because of floating numbers' inexact repr.
        repeated_segments += [seg] * round(dur * ALIGNMENT_FREQ)

    return f"{path.stem}\t{' '.join(repeated_segments)}"


def write_manifest_and_alignment(
    output: Path, language: str, manifest_lines: list[str], textgrid_lines: list[str]
) -> tuple[Path, Path]:
    manifests_dir = output / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    lang_manifest = manifests_dir / f"{language}.tsv"
    with open(lang_manifest, "w") as fp:
        fp.write("\n".join(manifest_lines) + "\n")

    alignments_dir = output / "alignments"
    alignments_dir.mkdir(exist_ok=True)
    lang_alignment = alignments_dir / f"{language}.align"
    with open(lang_alignment, "w") as fp:
        fp.write("\n".join(textgrid_lines) + "\n")
    return lang_manifest, lang_alignment


def clean_manifest_and_alignment(
    ft: panphon.FeatureTable,
    root: Path,
    textgrid_dir: Path,
    output: Path,
    split: Literal["train", "dev", "test"],
    language: str,
    target_size: int,
    min_length: float,
    max_length: float,
    suffix: str = ".TextGrid",
) -> tuple[Path, Path]:
    print(f"Language: {language}")
    lang_dir = root / language
    audio_dir = lang_dir / "clips_wav"
    full_manifest = lang_dir / f"{split}.tsv"
    paths, lengths = [], []
    textgrid_lines = []
    with open(full_manifest, "r", newline="") as fp:
        reader = csv.reader(fp, delimiter="\t", quoting=csv.QUOTE_NONE)
        _ = next(reader)
        for row in reader:
            # assert len(row) == 11, f"Invalid tsv file: {full_manifest}"
            audio_name = audio_dir / row[1]
            man_name = audio_name.with_suffix(".wav")
            length = sf.info(man_name).frames
            if length < min_length or length > max_length:
                continue
            align_line = convert_alignment(
                ft, (textgrid_dir / audio_name.name).with_suffix(suffix), **(_DIACRITICS_TO_FIX.get(language, {}))
            )
            if align_line:
                paths.append(man_name)
                lengths.append(length)
                textgrid_lines.append(align_line)
    lengths = np.array(lengths)

    manifest_lines = [root.as_posix()]
    if target_size > 0:
        print("\tSorting the audio files by length...", end="")
        tick = time.perf_counter()
        indices = np.argsort(-lengths)
        print(f"done in {time.perf_counter() - tick:.2f} s!")
        print(f"\tMinimum length: {lengths[indices[-1]] / SAMPLE_RATE:.2f} s")
        print(f"\tMaximum length: {lengths[indices[0]] / SAMPLE_RATE:.2f} s")

        cum_lengths = np.cumsum(lengths[indices]) / SAMPLE_RATE / 60
        cutoff_idx = np.searchsorted(cum_lengths, target_size, side="right")
        max_idx = min(cutoff_idx + 1, len(indices))
        print(f"\tSize retrieved: {cum_lengths[max_idx - 1]:.2f} min.")
        for idx in indices[:max_idx]:
            manifest_lines.append(f"{paths[idx].relative_to(root).as_posix()}\t{lengths[idx]}")
        textgrid_lines = [textgrid_lines[idx] for idx in indices[:max_idx]]
    else:
        for name, length in zip(paths, lengths):
            manifest_lines.append(f"{name.relative_to(root).as_posix()}\t{length}")

    return write_manifest_and_alignment(output, language, manifest_lines, textgrid_lines)


def create_split(
    root: Path,
    textgrid: Path,
    output: Path,
    split: Literal["train", "dev", "test"],
    language_file: Path,
    target_size: int,
    min_length: float,
    max_length: float,
    align_suffix: str = ".TextGrid",
) -> None:
    min_length *= SAMPLE_RATE
    if max_length < 0:
        max_length = float("inf")
    else:
        max_length *= SAMPLE_RATE
        assert min_length < max_length

    with open(language_file, "r") as fp:
        languages = fp.read().splitlines()
    print(f"Found {len(languages)} language(s) in {language_file}")

    ft = panphon.FeatureTable()
    manifest_files, alignment_files = [], []
    for lang in tqdm(languages, desc="Generating manifest and alignment files"):
        man_file, align_file = clean_manifest_and_alignment(
            ft, root, textgrid / lang, output / split, split, lang, target_size, min_length, max_length, align_suffix
        )
        manifest_files.append(man_file)
        alignment_files.append(align_file)

    with open(output / f"multilingual-{split}.tsv", "w") as wfp:
        wfp.write(f"{root.as_posix()}\n")
        for man_file in tqdm(manifest_files, desc="Dumping the multilingual manifest"):
            with open(man_file, "r") as rfp:
                list(itertools.islice(rfp, 1))
                shutil.copyfileobj(rfp, wfp)

    with open(output / f"multilingual-{split}.align", "w") as wfp:
        for align_file in tqdm(alignment_files, desc="Dumping the multilingual alignment"):
            with open(align_file, "r") as rfp:
                shutil.copyfileobj(rfp, wfp)

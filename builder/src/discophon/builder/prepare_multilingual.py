import csv
import itertools
import pickle
import shutil
import time
import wave
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import panphon
import soundfile as sf
import textgrids
from tqdm import tqdm

from .data import ALIGNMENT_FREQ, FeatureTokenizer, MontrealForcedAligner, PhoneticFeatureDataset, SAMPLE_RATE
from .decoder import FeatureDecoder, SILENCE

_DIACRITICS_TO_FIX = {
    "pa-IN": {"diacritic": "ʰ", "reverse": True},
    "ml": {"diacritic": "ʱ", "reverse": False},
    "mt": {"diacritic": "ˤː", "reverse": False},
    "ja": {"diacritic": "ː", "reverse": False},
    "cv17_ja": {"diacritic": "ː", "reverse": False},
}
MAX_SEGMENT_DURATION = 0.5


def concatenate_wavs(input_wavs: list[Path], output_wav: Path) -> None:
    data = []
    for input_wav in input_wavs:
        with wave.open(input_wav.as_posix(), "rb") as wav:
            data.append((wav.getparams(), wav.readframes(wav.getnframes())))
    with wave.open(output_wav.as_posix(), "wb") as wav:
        wav.setparams(data[0][0])
        for i in range(len(data)):
            wav.writeframes(data[i][1])


def _fix_phone_tier(phones: textgrids.Tier, diacritic: str, reverse: bool = False) -> tuple[list[str], list[float]]:
    segments, durations = [], []
    for k, phn in enumerate(phones):
        if phn.text == "spn":
            breakpoint()
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
                breakpoint()
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


def clean_manifest_and_alignment_lr(
    ft: panphon.FeatureTable,
    root: Path,
    new_root: Path,
    textgrid_dir: Path,
    output: Path,
    split: Literal["train", "dev", "test"],
    language: str,
    length_threshold: float,
    suffix: str = ".TextGrid",
) -> None:
    print(f"Language: {language}")
    (new_root / language).mkdir(parents=True, exist_ok=True)
    audio_dir = root / "clips_wav"
    full_manifest = root / f"{split}.tsv"
    path_length_textgrid_lines = []
    with open(full_manifest, "r", newline="") as fp:
        reader = csv.reader(fp, delimiter="\t", quoting=csv.QUOTE_NONE)
        _ = next(reader)
        for row in reader:
            assert len(row) == 11, f"Invalid tsv file: {full_manifest}"
            audio_name = audio_dir / row[1]
            man_name = audio_name.with_suffix(".wav")
            length = sf.info(man_name).frames
            if length >= length_threshold:
                continue
            align_line = convert_alignment(ft, (textgrid_dir / audio_name.name).with_suffix(suffix))
            if align_line:
                path_length_textgrid_lines.append((man_name, length, align_line))

    if not path_length_textgrid_lines:
        return

    manifest_lines, textgrid_lines = [new_root.as_posix()], []
    path_length_textgrid_lines.sort(key=lambda x: x[1])
    beg, end = 0, len(path_length_textgrid_lines) - 1
    merged_wavs = [path_length_textgrid_lines[end][0]]
    size = path_length_textgrid_lines[end][1]
    alignments = [path_length_textgrid_lines[end][2]]
    while beg < end:
        merged_wavs.append(path_length_textgrid_lines[beg][0])
        size += path_length_textgrid_lines[beg][1]
        alignments.append(path_length_textgrid_lines[beg][2].split("\t", 1)[1])
        beg += 1
        if size >= length_threshold:
            out_path = merged_wavs[0].relative_to(audio_dir)
            if out_path.parent.stem != language:
                out_path = out_path.parent.parent / out_path.stem
            manifest_lines.append(f"{language}/{out_path.as_posix()}\t{size}")
            concatenate_wavs(merged_wavs, new_root / language / out_path)
            textgrid_lines.append(" ".join(alignments))
            end -= 1
            merged_wavs = [path_length_textgrid_lines[end][0]]
            size = path_length_textgrid_lines[end][1]
            alignments = [path_length_textgrid_lines[end][2]]

    write_manifest_and_alignment(output, language, manifest_lines, textgrid_lines)


def enrich_low_resource(
    root: Path,
    textgrid: Path,
    output: Path,
    split: Literal["train", "dev", "test"],
    language_file: Path,
    length_threshold: float,
    align_suffix: str = ".TextGrid",
) -> None:
    length_threshold *= SAMPLE_RATE

    with open(language_file, "r") as fp:
        languages = fp.read().splitlines()
    print(f"Found {len(languages)} language(s) in {language_file}")

    ft = panphon.FeatureTable()
    for lang in tqdm(languages, desc="Generating manifest and alignment files"):
        clean_manifest_and_alignment_lr(
            ft,
            root / lang,
            output / "clips_wav",
            textgrid / lang,
            output / split,
            split,
            lang,
            length_threshold,
            align_suffix,
        )


def fetch_subset(split: Path, subset: str, language_file: Path, target_size: int) -> None:
    with open(language_file, "r") as fp:
        languages = fp.read().splitlines()
    print(f"Found {len(languages)} language(s) in {language_file}")

    root = split.with_name(subset)
    manifests_dir = root / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    alignment_num_lines = []
    for lang in tqdm(languages, desc="Processing manifest files"):
        manifest_lines = []
        manifest = split / "manifests" / f"{lang}.tsv"
        size = 0
        num_lines = 0
        with open(manifest, "r", newline="") as fp:
            reader = csv.reader(fp, delimiter="\t")
            manifest_lines.append(Path(next(reader)[0]).as_posix())
            for row in reader:
                assert len(row) == 2, f"Invalid tsv file: {manifest}"
                manifest_lines.append("\t".join(row))
                num_lines += 1
                size += int(row[1]) / SAMPLE_RATE / 60
                if size >= target_size:
                    break
            alignment_num_lines.append(num_lines)

        with open(manifests_dir / f"{lang}.tsv", "w") as fp:
            fp.write("\n".join(manifest_lines) + "\n")

    alignments_dir = root / "alignments"
    alignments_dir.mkdir(exist_ok=True)
    for lang, num_lines in tqdm(zip(languages, alignment_num_lines), desc="Processing alignment files"):
        alignment_lines = []
        with open(split / "alignments" / f"{lang}.align", "r") as fp:
            for line in fp:
                alignment_lines.append(line)
                if len(alignment_lines) >= num_lines:
                    break

        with open(alignments_dir / f"{lang}.align", "w") as fp:
            fp.write("".join(alignment_lines))


def fetch_and_merge_subsets(split: Path, subset: str, language_file: Path, target_size: int) -> None:
    with open(language_file, "r") as fp:
        languages = fp.read().splitlines()
    print(f"Found {len(languages)} language(s) in {language_file}")

    manifest_lines = []
    alignment_num_lines = []
    for lang in tqdm(languages, desc="Processing manifest files"):
        manifest = split / "manifests" / f"{lang}.tsv"
        size = 0
        num_lines = 0
        with open(manifest, "r", newline="") as fp:
            reader = csv.reader(fp, delimiter="\t")
            root = Path(next(reader)[0])
            if not manifest_lines:
                manifest_lines.append(root.as_posix())
            for row in reader:
                assert len(row) == 2, f"Invalid tsv file: {manifest}"
                manifest_lines.append("\t".join(row))
                num_lines += 1
                size += int(row[1]) / SAMPLE_RATE / 60
                if size >= target_size:
                    break
            alignment_num_lines.append(num_lines)

    with open(split.with_name(f"multilingual-{subset}.tsv"), "w") as fp:
        fp.write("\n".join(manifest_lines) + "\n")

    alignment_lines = []
    for lang, num_lines in tqdm(zip(languages, alignment_num_lines), desc="Processing alignment files"):
        count = 0
        with open(split / "alignments" / f"{lang}.align", "r") as fp:
            for line in fp:
                alignment_lines.append(line)
                count += 1
                if count >= num_lines:
                    break

    with open(split.with_name(f"multilingual-{subset}.align"), "w") as fp:
        fp.write("".join(alignment_lines))


def filter_data(root: Path, utt_accuracy: Path, thresholds: list[int]) -> None:
    manifests_dir = root / "manifests"
    alignments_dir = root / "alignments"

    with open(utt_accuracy, "rb") as fp:
        bin_sets = pickle.load(fp)
    thres_sets = [set() for _ in thresholds]
    idx = len(bin_sets) - 1
    print("Starting the union of threshold indices...", end="")
    while idx >= min(thresholds):
        for i, thres in enumerate(thresholds):
            if idx >= thres:
                thres_sets[i].update(bin_sets[idx])
        idx -= 1
    del bin_sets
    print("Done!")

    feature_decoder = FeatureDecoder(sum_diphthong=False)
    tokenizer = FeatureTokenizer("unk-ign", feature_decoder)
    silence_prediction = True
    dataset = PhoneticFeatureDataset(
        manifests_dir, alignments_dir, tokenizer, separate_files=True, silence_prediction=silence_prediction
    )

    cv_root = Path("/lustre/fsmisc/dataset/CommonVoice/cv-corpus-16.1-2023-12-06")
    for i, thres in enumerate(thresholds):
        man_lines = [cv_root.as_posix()]
        align_lines = []

        for idx in tqdm(thres_sets[i], desc=f"Processing threshold {thres}"):
            file, num_samples = dataset.manifest[idx][1]
            man_lines.append(f"{file.relative_to(cv_root)}\t{num_samples}")
            _align = " ".join(dataset.ipa_phones[file.stem])
            align_lines.append(f"{file.stem}\t{_align}")

        with open(root.parent / f"multilingual-{root.name}_{thres}.tsv", "w") as fp:
            fp.write("\n".join(man_lines) + "\n")
        with open(root.parent / f"multilingual-{root.name}_{thres}.align", "w") as fp:
            fp.write("\n".join(align_lines))
        print(f"Saved the manifest and alignment files for threshold {thres}")

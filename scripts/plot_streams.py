# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "discophon>=0.0.5",
#     "matplotlib>=3.10.8",
#     "numpy>=2.3.5",
#     "praat-parselmouth>=0.4.7",
#     "tqdm>=4.67.3",
# ]
# ///
import argparse
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import parselmouth  # ty: ignore[unresolved-import]
from discophon.evaluate.pnmi import coocurrence_matrix, mapping_many_to_one
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from discophon.data import (
    Phones,
    TextGridEntry,
    Units,
    read_gold_annotations,
    read_submitted_units,
    textgrid_array_from_sequence,
)
from discophon.languages import Language, get_language


def plot_wav(ax: Axes, snd: parselmouth.Sound) -> None:
    ax.plot(snd.xs(), snd.values.T, color=(0.3, 0.3, 0.3), rasterized=True)
    ax.set_xlim(snd.xmin, snd.xmax)
    ax.set_ylabel("amplitude")
    ax.tick_params(axis="y", which="major", labelsize=8)


def plot_spectrogram(ax: Axes, snd: parselmouth.Sound, *, dynamic_range: float = 70) -> None:
    spectrogram = snd.to_spectrogram(window_length=0.03, maximum_frequency=8000)
    sg_db = 10 * np.log10(spectrogram.values)
    ax.pcolormesh(
        spectrogram.x_grid(),
        spectrogram.y_grid(),
        sg_db,
        vmin=sg_db.max() - dynamic_range,
        cmap="gray_r",
        rasterized=True,
    )
    ax.set_ylim(spectrogram.ymin, spectrogram.ymax)
    ax.set_ylabel("frequency [Hz]")
    ax.tick_params(axis="y", which="major", labelsize=8)

    ax2 = ax.twinx()
    pitch = snd.to_pitch()
    pitch_values = pitch.selected_array["frequency"]
    pitch_values[pitch_values == 0] = np.nan
    ax2.plot(pitch.xs(), pitch_values, "o", markersize=5, color="w")
    ax2.plot(pitch.xs(), pitch_values, "o", markersize=2)
    ax2.grid(visible=False)
    ax2.set_ylim(0, pitch.ceiling)
    ax2.set_ylabel("F0 [Hz]")
    ax2.tick_params(axis="y", which="major", labelsize=8)


def draw_entries(
    ax: Axes,
    entries: list[TextGridEntry],
    begin: float,
    end: float,
    *,
    rect_kwargs: dict | None = None,
    text_kwargs: dict | None = None,
) -> None:
    rect_kwargs = rect_kwargs or {}
    text_kwargs = text_kwargs or {}
    for entry in entries:
        if (entry["begin"] < begin and entry["end"] <= begin) or (entry["begin"] >= end and entry["end"] > end):
            continue
        if entry["begin"] < begin and entry["end"] <= end:
            entry_begin, entry_end = begin, entry["end"]
        elif entry["begin"] >= begin and entry["end"] > end:
            entry_begin, entry_end = entry["begin"], end
        else:
            entry_begin, entry_end = entry["begin"], entry["end"]
        rect = plt.Rectangle((entry_begin, 0), entry_end - entry_begin, 1, **rect_kwargs)
        ax.add_patch(rect)
        ax.text((entry_begin + entry_end) / 2, 1 / 2, entry["label"], ha="center", va="center", **text_kwargs)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_yticks([])


def plot_streams_on_axis(
    ax: Sequence[Axes],
    snd: parselmouth.Sound,
    phones: list[TextGridEntry],
    units: list[TextGridEntry],
    predictions: list[TextGridEntry],
) -> None:
    plot_wav(ax[0], snd)
    plot_spectrogram(ax[1], snd)
    draw_entries(ax[2], phones, snd.xmin, snd.xmax, rect_kwargs={"color": "C0", "alpha": 0.4})
    draw_entries(
        ax[3],
        units,
        snd.xmin,
        snd.xmax,
        rect_kwargs={"color": "C1", "alpha": 0.4},
        text_kwargs={"fontsize": 6, "rotation": 90},
    )
    draw_entries(ax[4], predictions, snd.xmin, snd.xmax, rect_kwargs={"color": "C2", "alpha": 0.4})
    ax[0].set_xlim(snd.xmin, snd.xmax)
    ax[2].set_ylabel("gold phones")
    ax[3].set_ylabel("units")
    ax[4].set_ylabel("predictions")
    ax[4].set_xlabel("time [s]")


def infer_from_arguments(path_audio: Path, path_units: Path) -> tuple[Phones, Units, Language]:
    path_audio = path_audio.resolve()
    assert path_audio.parent.stem in {"dev", "test", "all"}, "Audio should be in a split folder"
    assert path_audio.parents[2].stem == "audio", "Audio should be in the 'audio' folder of the dataset."
    language, split, dataset = get_language(path_audio.parents[1].stem), path_audio.parent.stem, path_audio.parents[3]
    if split == "all":
        split = path_units.stem.split("-")[-1]
        assert split in {"dev", "test"}, (
            "When the audio file is in the 'all' folder, the units file should be named 'units-<lang>-<split>.jsonl'"
        )
    units = read_submitted_units(path_units)
    phones = read_gold_annotations(dataset / f"alignment/alignment-{language.iso_639_3}-{split}.txt")
    return phones, units, language


def plot_streams(
    path_audio: Path,
    path_units: Path,
    *,
    n_units: int,
    step_units: int,
    step_phones: int = 10,
    begin_and_end: tuple[float, float] | None = None,
) -> Figure:
    phones, units, language = infer_from_arguments(path_audio, path_units)
    contingency = coocurrence_matrix(
        units,
        phones,
        n_units=n_units,
        n_phonemes=language.n_phonemes,
        step_units=step_units,
        step_phones=step_phones,
    )
    mapping = mapping_many_to_one(contingency)

    name = path_audio.stem
    snd = parselmouth.Sound(str(path_audio))
    if begin_and_end is not None:
        snd = snd.extract_part(*begin_and_end, preserve_times=True)
    this_units = textgrid_array_from_sequence(units[name], step_in_ms=step_units)
    this_phones = textgrid_array_from_sequence(phones[name], step_in_ms=step_phones)
    this_predictions = textgrid_array_from_sequence([mapping[u] for u in units[name]], step_in_ms=step_units)
    figsize_x = 4 * (snd.xmax - snd.xmin)
    fig, ax = plt.subplots(5, 1, figsize=(figsize_x, 6.5), constrained_layout=True, sharex=True)
    plot_streams_on_axis(ax, snd, this_phones, this_units, this_predictions)
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot waveform, spectrogram, gold phones, units, and predicted phones."
    )
    parser.add_argument("audio", type=Path, help="Path to the audio file in the dataset folder.")
    parser.add_argument("units", type=Path, help="Path to the units file")
    parser.add_argument("output", type=Path, help="Path to the output figure")
    parser.add_argument("--n-units", type=int, default=256, help="Number of discrete units. (default: 256)")
    parser.add_argument("--step-units", type=int, default=20, help="Step in ms between units (default: 20)")
    parser.add_argument(
        "--begin-and-end",
        type=float,
        nargs=2,
        metavar=("BEGIN", "END"),
        help="Begin and end times in seconds",
    )
    args = parser.parse_args()

    plot_streams(
        args.audio,
        args.units,
        n_units=args.n_units,
        step_units=args.step_units,
        begin_and_end=args.begin_and_end,
    ).savefig(args.output)
    plt.close()

import itertools
from collections.abc import Iterable
from typing import TypedDict

import matplotlib.pyplot as plt
import numpy as np
import parselmouth
from matplotlib.axes import Axes

from discophon.data import read_gold_annotations, read_submitted_units
from discophon.evaluate.pnmi import contingency_table, mapping_many_to_one
from discophon.languages import get_language, load_tipa

TIPA = load_tipa()
TIPA["SIL"] = "SIL"


class Entry(TypedDict):
    begin: float
    end: float
    label: str


def seq_to_entries(seq: Iterable[str | int], *, step: float) -> list[Entry]:
    labels, counts = zip(
        *[(key, len(list(group))) for key, group in itertools.groupby(seq)], strict=True
    )
    ends = np.cumsum(counts)
    starts = np.concatenate(([0], ends[:-1]))
    return [
        {"begin": starts[i] * step, "end": ends[i] * step, "label": str(labels[i])}
        for i in range(len(labels))
    ]


def draw_stream(
    ax: Axes,
    entries: list[Entry],
    begin: float,
    end: float,
    *,
    rect_kw: dict | None = None,
    text_kw: dict | None = None,
) -> None:
    if rect_kw is None:
        rect_kw = {}
    if text_kw is None:
        text_kw = {}
    for entry in entries:
        if (entry["begin"] < begin and entry["end"] <= begin) or (
            entry["begin"] >= end and entry["end"] > end
        ):
            continue
        if entry["begin"] < begin and entry["end"] <= end:
            entry_begin, entry_end = begin, entry["end"]
        elif entry["begin"] >= begin and entry["end"] > end:
            entry_begin, entry_end = entry["begin"], end
        else:
            entry_begin, entry_end = entry["begin"], entry["end"]
        ax.add_patch(
            plt.Rectangle((entry_begin, 0), entry_end - entry_begin, 1, **rect_kw)
        )
        ax.text(
            (entry_begin + entry_end) / 2,
            1 / 2,
            entry["label"] if entry["label"].isdigit() else TIPA[entry["label"]],
            ha="center",
            va="center",
            **text_kw,
        )
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_yticks([])


if __name__ == "__main__":
    plt.style.use("./paper.mplstyle")
    name, start, end = "0188-135249-0001", 0.37, 1.535
    snd = parselmouth.Sound(f"./data/{name}.wav").extract_part(
        from_time=start, to_time=end, preserve_times=True
    )
    all_units = read_submitted_units("./data/units-eng-test.jsonl")
    all_phones = read_gold_annotations("./data/alignment-eng-test.txt")
    contingency = contingency_table(
        all_units,
        all_phones,
        n_units=256,
        n_phonemes=get_language("eng").n_phonemes,
        step_units=20,
        step_phones=10,
    )
    mapping = mapping_many_to_one(contingency)
    units = seq_to_entries(all_units[name], step=0.02)
    phones = seq_to_entries(all_phones[name], step=0.01)
    preds = seq_to_entries([mapping[u] for u in all_units[name]], step=0.02)

    fig, ax = plt.subplots(
        4,
        1,
        figsize=(3.1, 1.2),
        layout="constrained",
        sharex=True,
        height_ratios=[0.75, 1, 1, 1],
    )
    ax[0].plot(snd.xs(), snd.values.T, color=(0.3, 0.3, 0.3), rasterized=True)
    ax[0].axis("off")
    ax[0].set_xlim(snd.xmin, snd.xmax)
    ax[0].set_ylabel("amplitude")
    ax[0].set_ylabel("")
    ax[0].tick_params(axis="y", which="major", labelsize=8)
    draw_stream(
        ax[1],
        phones,
        snd.xmin,
        snd.xmax,
        rect_kw={"color": "C0", "alpha": 0.3},
        text_kw={"fontsize": 8},
    )
    draw_stream(
        ax[2],
        units,
        snd.xmin,
        snd.xmax,
        rect_kw={"color": "C1", "alpha": 0.3},
        text_kw={"fontsize": 4, "rotation": 90},
    )
    draw_stream(
        ax[3],
        preds,
        snd.xmin,
        snd.xmax,
        rect_kw={"color": "C2", "alpha": 0.3},
        text_kw={"fontsize": 8},
    )
    ax[1].set_ylabel(r"$\bm{p}$", rotation=0, va="center", labelpad=7, fontsize=9)
    ax[2].set_ylabel(r"$\bm{u}$", rotation=0, va="center", labelpad=7, fontsize=9)
    ax[3].set_ylabel(r"$\bm{a}$", rotation=0, va="center", labelpad=7, fontsize=9)
    ax[1].tick_params(axis="x", length=0)
    ax[2].tick_params(axis="x", length=0)
    ax[3].set_xticks([0.5, 0.7, 0.9, 1.1, 1.3, 1.5])
    ax[3].set_xticklabels(
        [r"$0.5$s", r"$0.7$s", r"$0.9$s", r"$1.1$s", r"$1.3$s", r"$1.5$s"]
    )
    for x in [patch.get_x() for patch in ax[3].patches[1:]]:
        ax[2].axvline(
            x=x,
            color="k",
            linestyle="dotted",
            linewidth=0.5,
            ymin=-0.28,
            ymax=0,
            clip_on=False,
        )
    fig.get_layout_engine().set(w_pad=0, h_pad=0.03, hspace=0, wspace=0)
    plt.savefig("figures/streams.pdf", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close()

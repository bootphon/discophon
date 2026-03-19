# %%
import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Literal, TypedDict

import matplotlib.pyplot as plt
import numpy as np
from discophon.languages import (
    Language,
    dev_languages,
    load_sonority,
    load_tipa,
    test_languages,
)
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle


class Vowel(TypedDict):
    height: Literal[
        "close", "near-close", "close-mid", "mid", "open-mid", "near-open", "open"
    ]
    backness: Literal["front", "central", "back"]
    roundedness: Literal["rounded", "unrounded"]


VOWELS = {
    "i": ["i", "iː"],
    "y": ["y", "yː"],
    "ɯ": ["ɯ", "ɯː"],
    "u": ["u", "uː"],
    "ɪ": ["ɪ"],
    "ʏ": ["ʏ"],
    "ʊ": ["ʊ"],
    "e": ["e", "eː"],
    "ø": ["ø", "øː"],
    "ɤ": ["ɤ", "ɤː"],
    "o": ["o", "oː"],
    "ə": ["ə"],
    "ɛ": ["ɛ", "ɛː", "ɛ̃"],
    "œ": ["œ", "œ̃"],
    "ɜ": ["ɜ", "ɝ"],
    "ʌ": ["ʌ"],
    "ɔ": ["ɔ", "ɔː", "ɔ̃"],
    "æ": ["æ"],
    "a": ["a", "aː"],
    "ɑ": ["ɑ", "ɑː", "ɑ̃"],
}

VOWELS_COORDS = {
    "i": Vowel(height="close", backness="front", roundedness="unrounded"),
    "y": Vowel(height="close", backness="front", roundedness="rounded"),
    "ɯ": Vowel(height="close", backness="back", roundedness="unrounded"),
    "u": Vowel(height="close", backness="back", roundedness="rounded"),
    "ɪ": Vowel(height="near-close", backness="front", roundedness="unrounded"),
    "ʏ": Vowel(height="near-close", backness="front", roundedness="rounded"),
    "ʊ": Vowel(height="near-close", backness="back", roundedness="rounded"),
    "e": Vowel(height="close-mid", backness="front", roundedness="unrounded"),
    "ø": Vowel(height="close-mid", backness="front", roundedness="rounded"),
    "ɤ": Vowel(height="close-mid", backness="back", roundedness="unrounded"),
    "o": Vowel(height="close-mid", backness="back", roundedness="rounded"),
    "ə": Vowel(height="mid", backness="central", roundedness="unrounded"),
    "ɛ": Vowel(height="open-mid", backness="front", roundedness="unrounded"),
    "œ": Vowel(height="open-mid", backness="front", roundedness="rounded"),
    "ɜ": Vowel(height="open-mid", backness="central", roundedness="unrounded"),
    "ʌ": Vowel(height="open-mid", backness="back", roundedness="unrounded"),
    "ɔ": Vowel(height="open-mid", backness="back", roundedness="rounded"),
    "æ": Vowel(height="near-open", backness="front", roundedness="unrounded"),
    "a": Vowel(height="open", backness="front", roundedness="unrounded"),
    "ɑ": Vowel(height="open", backness="back", roundedness="unrounded"),
}


CONSONANTS = {
    "b": ["b", "ᵐb"],
    "c": ["c"],
    "ç": ["ç"],
    "d": ["d", "dʲ", "ⁿd"],
    "d͡z": ["d͡z"],
    "d͡ʑ": ["d͡ʑ"],
    "d͡ʒ": ["d͡ʒ"],
    "f": ["f"],
    "h": ["h"],
    "j": ["j"],
    "k": ["k", "kʰ", "kː"],
    "l": ["l", "lʲ"],
    "m": ["m", "mː"],
    "n": ["n", "nʲ", "nː"],
    "p": ["p", "pʰ", "pː"],
    "q": ["q"],
    "r": ["r", "rʲ"],
    "s": ["s", "sʲ", "sː", "s̺", "s̻"],
    "t": ["t", "tʰ", "tʲ", "tː", "t̪"],
    "t͡s": ["t͡s", "t͡sʰ", "t͡sʲ", "t͡s̺", "t͡s̻"],
    "t͡ɕ": ["t͡ɕ", "t͡ɕʰ"],
    "t͡ʃ": ["t͡ʃ"],
    "v": ["v"],
    "w": ["w"],
    "x": ["x"],
    "z": ["z", "zʲ"],
    "ð": ["ð"],
    "ŋ": ["ŋ"],
    "ɕ": ["ɕ", "ɕː"],
    "ɟ": ["ɟ", "ᶮɟ"],
    "ɡ": ["ɡ", "ᵑɡ"],
    "ɣ": ["ɣ"],
    "ɥ": ["ɥ"],
    "ɦ": ["ɦ"],
    "ɭ": ["ɭ"],
    "ɲ": ["ɲ"],
    "ɳ": ["ɳ"],
    "ɴ": ["ɴ"],
    "ɸ": ["ɸ"],
    "ɹ": ["ɹ", "ɻ", "ɹ̩", "ɻ̩"],
    "ɾ": ["ɾ"],
    "ʁ": ["ʁ"],
    "ʂ": ["ʂ"],
    "ʃ": ["ʃ"],
    "ʈ": ["ʈ"],
    "ʈ͡ʂ": ["ʈ͡ʂ", "ʈ͡ʂʰ"],
    "ʋ": ["ʋ"],
    "ʑ": ["ʑ"],
    "ʒ": ["ʒ"],
    "ʔ": ["ʔ"],
    "θ": ["θ"],
}
CONSONANTS_COORDINATES = json.loads(
    Path("./data/coordinates.json").read_text(encoding="utf-8")
)
for a, b in CONSONANTS_COORDINATES.items():
    if b <= 3:
        CONSONANTS_COORDINATES[a] = b
    elif b <= 19:
        CONSONANTS_COORDINATES[a] = b - 2
    else:
        CONSONANTS_COORDINATES[a] = b - 4
FONTSIZE_TEXT = 9
FONTSIZE_PHONEMES = 8
FONTSIZE_COUNT = 6
ORDER = [
    "Fricative",
    "Affricate",
    "Plosive",
    "Nasal",
    "Vibrant",
    "Approximant",
    "Monophthong",
    "Diphthong",
]
THRESHOLD = 7


def count(
    languages: Iterable[Language], reference: dict[str, list | tuple]
) -> Counter[str]:
    return Counter(
        [
            ipa
            for language in languages
            for ipa, forms in reference.items()
            if any(form in language.phonemes for form in forms)
        ]
    )


def plot_counts(fig: Figure, colors: list) -> None:
    ax = fig.subplots(nrows=2, ncols=1, sharey=True, gridspec_kw={"hspace": 0.12})
    for k, langs in enumerate([dev_languages(), test_languages()]):
        counts = {
            lang: Counter([load_sonority()[p] for p in lang.phonemes]) for lang in langs
        }
        total = sorted(
            {name: sum(c.values()) for name, c in counts.items()}.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        x = [lang.iso_639_3 for lang, _ in total]
        bottom = np.zeros(len(counts))
        for i, s in enumerate(ORDER):
            values = np.asarray([counts[lang][s.lower()] for lang, _ in total])
            ax[k].bar(
                x=x,
                height=values,
                bottom=bottom,
                label=s,
                color=colors[i],
                hatch="////" if s in {"Monophthong", "Diphthong"} else None,
            )
            bottom += values
        for i, b in enumerate(bottom):
            ax[k].text(
                i, b, f"{int(b)}", ha="center", va="bottom", fontsize=FONTSIZE_COUNT
            )
        ax[k].spines[["top", "right"]].set_visible(False)
        ax[k].set_yticks([0, 20, 40])
    ax[1].legend(
        bbox_to_anchor=(0.5, 0.5),
        loc="center",
        ncols=3,
        fontsize=FONTSIZE_COUNT,
        bbox_transform=fig.transSubfigure,
    )
    ax[0].set_title("dev", fontsize=FONTSIZE_TEXT)
    ax[1].set_title("test", fontsize=FONTSIZE_TEXT)


class Trapezoid:
    def __init__(self, max_x: float, *, y_bottom: float = 1, y_top: float = 0) -> None:
        self.max_x = max_x
        self.y_bottom = y_bottom
        self.y_top = y_top
        self.y_center = y_bottom / 3
        self.y_lower = 2 * y_bottom / 3
        self.x0_top = 0
        self.x1_top = 0.5 * max_x
        self.x0_bottom = 0.4 * max_x
        self.x1_bottom = 0.7 * max_x
        self.N = 2

    def left_diagonal(self, y: np.ndarray | float) -> np.ndarray | float:
        return (y - self.y_top) / (self.y_bottom - self.y_top) * (
            self.x0_bottom - self.x0_top
        ) + self.x0_top

    def center_diagonal(self, y: np.ndarray | float) -> np.ndarray | float:
        return (y - self.y_top) / (self.y_bottom - self.y_top) * (
            self.x1_bottom - self.x1_top
        ) + self.x1_top

    def coordinates(self, vowel: Vowel) -> tuple[float, float]:
        match vowel["height"]:
            case "close":
                y = self.y_top
            case "near-close":
                y = (self.y_top + self.y_center) / 2
            case "close-mid":
                y = self.y_center
            case "mid":
                y = (self.y_center + self.y_lower) / 2
            case "open-mid":
                y = self.y_lower
            case "near-open":
                y = (self.y_lower + self.y_bottom) / 2
            case "open":
                y = self.y_bottom
        match vowel["backness"]:
            case "back":
                x = self.max_x
            case "central":
                x = self.center_diagonal(y)
            case "front":
                x = self.left_diagonal(y)
        return x, y

    def plot(self, ax: Axes, **kwargs) -> None:
        y = np.linspace(self.y_top, self.y_bottom, self.N)
        ax.plot(np.full_like(y, self.max_x), y, "k-", **kwargs)
        ax.plot(self.left_diagonal(y), y, "k-", **kwargs)
        ax.plot(self.center_diagonal(y), y, "k-", **kwargs)
        for min_x, height in [
            (self.x0_top, self.y_top),
            (self.left_diagonal(self.y_center), self.y_center),
            (self.left_diagonal(self.y_lower), self.y_lower),
            (self.x0_bottom, self.y_bottom),
        ]:
            x = np.linspace(min_x, self.max_x, self.N)
            y = np.full_like(x, height)
            ax.plot(x, y, "k-", **kwargs)


def plot_vowels(ax: Axes, colors: list, *, width: float, height: float) -> None:
    trapezoid = Trapezoid(1.35)
    shift_exceptions = {"ɪ": 1, "ʏ": 1, "ʊ": -1}
    center_exceptions = {"ə", "ɹ̩"}
    dev_vowels = count(dev_languages(), VOWELS)
    test_vowels = count(test_languages(), VOWELS)
    trapezoid.plot(
        ax, lw=0.5, solid_capstyle="round", solid_joinstyle="round", zorder=1
    )
    for ipa, vowel in VOWELS_COORDS.items():
        x, y = trapezoid.coordinates(vowel)
        if ipa in shift_exceptions:
            x += shift_exceptions[ipa] * 0.25
        if ipa not in center_exceptions:
            x += (-1 if vowel["roundedness"] == "unrounded" else 1) * width / 2
        d, t = dev_vowels[ipa], test_vowels[ipa]
        ax.annotate(
            load_tipa()[ipa],
            (x, y),
            xytext=(0, 0),
            textcoords="offset points",
            ha="center",
            va="center",
            zorder=3,
            fontsize=FONTSIZE_PHONEMES,
            color="#f0f0f0" if d + t > THRESHOLD else "black",
        )
        rect = Rectangle(
            (x - width / 2, y - height / 2),
            width=width,
            height=height,
            color=colors[d + t - 1],
            zorder=1,
            lw=None,
            ec=None,
        )
        ax.add_patch(rect)

    ax.set_xlim(-width, trapezoid.max_x + 0.3)
    ax.set_ylim(-height / 2, 1 + height / 2)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.axis("off")


def plot_consonants(ax: Axes, colors: list, *, width: float, height: float) -> None:
    dev_consonants = count(dev_languages(), CONSONANTS)
    test_consonants = count(test_languages(), CONSONANTS)
    all_consonants = dev_consonants + test_consonants
    for i, m in enumerate(
        ["fricative", "affricate", "plosive", "vibrant", "nasal", "approximant"]
    ):
        ipas = [ipa for ipa in all_consonants if load_sonority()[ipa] == m]
        ipas = sorted(ipas, key=lambda ipa: CONSONANTS_COORDINATES[ipa])
        for ipa in ipas:
            x, y = CONSONANTS_COORDINATES[ipa] * width, i * height
            d, t = dev_consonants[ipa], test_consonants[ipa]
            ax.annotate(
                load_tipa()[ipa],
                (x, y),
                xytext=(0, 0),
                textcoords="offset points",
                ha="center",
                va="center",
                zorder=3,
                fontsize=FONTSIZE_PHONEMES,
                color="#f0f0f0" if d + t > THRESHOLD else "black",
            )
            rect = Rectangle(
                (x - width / 2, y - height / 2),
                width=width,
                height=height,
                color=colors[d + t - 1],
                zorder=1,
                ec=None,
                lw=None,
            )
            ax.add_patch(rect)
        ax.text(
            -0.8,
            i * height,
            m.capitalize(),
            fontsize=FONTSIZE_TEXT,
            ha="left",
            va="center",
        )

    bottom = 5 * height
    top = -0.5 * height
    right = max(CONSONANTS_COORDINATES[ipa] for ipa in all_consonants) * width
    toprule, bottomrule, rightedge, leftedge = (
        top - 0.35,
        bottom + 0.2,
        right + 0.18,
        -0.85,
    )
    vlines = [3.5, 12.5, 17.5]
    for xx in vlines:
        ax.vlines(
            xx * width,
            top,
            bottomrule,
            colors="k",
            alpha=0.75,
            lw=0.5,
            zorder=1,
            linestyle="dotted",
        )
    for xx, place in zip(
        [
            vlines[0] / 2,
            (vlines[0] + vlines[1]) / 2,
            (vlines[1] + vlines[2]) / 2,
            (rightedge / width + vlines[2]) / 2,
        ],
        ["Labial", "Coronal", "Dorsal", "Glottal"],
        strict=True,
    ):
        ax.text(
            xx * width,
            top - 0.15,
            place,
            fontsize=FONTSIZE_TEXT,
            ha="center",
            va="center",
        )
    ax.hlines(
        [toprule, top, bottomrule],
        leftedge,
        rightedge,
        colors="k",
        lw=[1, 0.625, 1],
        zorder=4,
    )
    ax.axis("off")
    ax.set_xlim(leftedge, rightedge)
    ax.set_ylim(toprule - 0.1, bottomrule + 0.1)
    ax.invert_yaxis()


def plot_cbar(ax: Axes, colors: list) -> None:
    ax.axis("off")
    cax = ax.inset_axes([0, 0.5, 0.8, 0.1])
    ticks = (np.arange(len(colors)) + 0.5) / len(colors)
    cbar = Colorbar(
        ax=cax,
        cmap=ListedColormap(colors),
        orientation="horizontal",
        ticks=ticks,
        location="bottom",
    )
    cbar.ax.tick_params(labelsize=FONTSIZE_PHONEMES)
    cbar.set_ticklabels([str(i) for i in range(1, 13)])
    cax.set_title("Number of languages with this phoneme", fontsize=FONTSIZE_TEXT)


colors = [plt.colormaps["magma"]((i + 0.5) / len(ORDER)) for i in range(len(ORDER))]
colors_rect = [plt.colormaps["Greys"]((i * 1 / 2 + 2) / 12) for i in range(12)]

plt.style.use("./paper.mplstyle")
fig = plt.figure(figsize=(6.69, 2.2), layout="constrained")
left_right = fig.subfigures(1, 2, wspace=0, width_ratios=[1, 0.6])
top_bottom = left_right[0].subfigures(2, 1, hspace=0.0, height_ratios=[1, 0.85])
vowel_and_empty = top_bottom[1].subfigures(
    1, 2, hspace=0.0, wspace=0, width_ratios=[1, 1.2]
)
plot_consonants(top_bottom[0].add_subplot(), colors_rect, width=0.2, height=0.4)
plot_vowels(vowel_and_empty[0].add_subplot(), colors_rect, width=0.12, height=0.167)
plot_cbar(vowel_and_empty[1].add_subplot(), colors_rect)
plot_counts(left_right[1], colors)
fig.text(0.02, 0.05, r"\textbf{A}", fontsize=14, va="top")
fig.text(0.62, 0.05, r"\textbf{B}", fontsize=14, va="top")
plt.savefig("figures/phonemes.pdf", bbox_inches="tight", pad_inches=0.02)
plt.show()

# %%
colors
# %%
ListedColormap(colors_rect)
# %%
len([i / len(colors_rect) for i in range(len(colors_rect))])
# %%

# %%
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from discophon.data import STEP_PHONES, read_gold_annotations, read_submitted_units
from discophon.evaluate.per import deduplicate
from discophon.evaluate.pnmi import align_units_and_phones, compute_pnmi_and_predict
from discophon.languages import (
    Language,
    all_languages,
    dev_languages,
    load_sonority,
    test_languages,
)
from jiwer import Compose, WordOutput, collect_error_counts, process_words
from matplotlib.axes import Axes
from matplotlib.colors import to_rgba_array

CLASS_ORDER = [
    "SIL",
    "fricative",
    "affricate",
    "plosive",
    "vibrant",
    "nasal",
    "approximant",
    "monophthong",
    "diphthong",
]


def load_data(
    units_path: str,
    phones_path: str,
    language: Language,
    *,
    n_units: int = 256,
    step_units: int = 20,
) -> tuple[list[list[str]], list[list[str]], Counter[str]]:
    units = read_submitted_units(units_path)
    phones = read_gold_annotations(phones_path)
    _, preds = compute_pnmi_and_predict(
        units,
        phones,
        n_units=n_units,
        n_phonemes=language.n_phonemes,
        step_units=step_units,
        step_phones=STEP_PHONES,
    )
    units_and_phones = align_units_and_phones(
        units, phones, step_units=step_units, step_phones=STEP_PHONES
    ).values()
    counts = Counter([p for seq in units_and_phones for p in seq["phones"]])

    reference, hypothesis = zip(
        *[(deduplicate(phones[name]), deduplicate(preds[name])) for name in preds],
        strict=True,
    )
    return list(reference), list(hypothesis), counts


def clean_substitutions(
    errors: defaultdict[tuple[str, str], int],
) -> defaultdict[tuple[str, str], int]:
    new_errors = defaultdict(int)
    for (ref, hyp), count in errors.items():
        for r, h in zip(ref.split(), hyp.split(), strict=True):
            new_errors[r, h] += count
    return new_errors


def clean_insertions_or_deletions(
    errors: defaultdict[str, int],
) -> defaultdict[str, int]:
    new_errors = defaultdict(int)
    for key, count in errors.items():
        for k in key.split():
            new_errors[k] += count
    return new_errors


def error_counts(
    *, reference: list[list[str]], hypothesis: list[list[str]]
) -> tuple[
    defaultdict[tuple[str, str], int],
    defaultdict[str, int],
    defaultdict[str, int],
    WordOutput,
]:
    word_output = process_words(
        reference=reference,
        hypothesis=hypothesis,
        reference_transform=Compose([]),
        hypothesis_transform=Compose([]),
    )
    substitutions, insertions, deletions = collect_error_counts(word_output)
    return (
        clean_substitutions(substitutions),
        clean_insertions_or_deletions(insertions),
        clean_insertions_or_deletions(deletions),
        word_output,
    )


def get_heatmap_array(model: str) -> np.ndarray:
    sonority = load_sonority() | {"SIL": "SIL"}
    s_by_class = defaultdict(int)
    for lang in all_languages():
        for (a, b), count in results[model][lang.iso_639_3]["substitutions"].items():
            s_by_class[(sonority[a], sonority[b])] += (
                count if sonority[a] != sonority[b] else 0
            )
    heatmap = np.zeros((len(CLASS_ORDER), len(CLASS_ORDER)), dtype=np.float64)
    for i, c1 in enumerate(CLASS_ORDER):
        for j, c2 in enumerate(CLASS_ORDER):
            heatmap[i, j] = s_by_class[(c1, c2)]  # C[i] replaced by C[j]
    heatmap /= heatmap.sum(axis=1, keepdims=True)
    # heatmap = heatmap + heatmap.T - np.diag(heatmap.diagonal())
    # n = heatmap.sum(axis=1)
    # norm = np.sqrt(np.outer(n, n))
    # heatmap = heatmap / norm
    return heatmap


data, results, counts = [], defaultdict(dict), {}
for model, layer in [("spidr-vp20", 5), ("spidr-vp20-ft-10h", 5)]:
    for lang in all_languages():
        reference, hypothesis, this_counts = load_data(
            f"./data/units/{model}/{layer}/units-{lang.iso_639_3}-test.jsonl",
            f"./data/alignment-{lang.iso_639_3}-test.txt",
            language=lang,
        )
        counts[lang.iso_639_3] = this_counts
        s, i, d, o = error_counts(reference=reference, hypothesis=hypothesis)
        data.append(
            {
                "model": model,
                "language": lang.iso_639_3,
                "hits": o.hits,
                "substitutions": o.substitutions,
                "insertions": o.insertions,
                "deletions": o.deletions,
            }
        )
        results[model][lang.iso_639_3] = {
            "substitutions": s,
            "insertions": i,
            "deletions": d,
            "word_output": o,
        }
df = (
    pl.DataFrame(data)
    .with_columns(total=pl.col("hits") + pl.col("substitutions") + pl.col("deletions"))
    .with_columns(
        hits=100 * pl.col("hits") / pl.col("total"),
        substitutions=100 * pl.col("substitutions") / pl.col("total"),
        deletions=100 * pl.col("deletions") / pl.col("total"),
        insertions=100 * pl.col("insertions") / pl.col("total"),
        per=100
        * (pl.col("substitutions") + pl.col("deletions") + pl.col("insertions"))
        / pl.col("total"),
    )
)

order = {
    lang: k
    for k, lang in enumerate(
        df.filter(pl.col("model") == "spidr-vp20")
        .sort("per", descending=True)
        .select("language")
        .to_series()
        .to_list()
    )
}

df = (
    df.with_columns(order=pl.col("language").replace_strict(order))
    .sort("order", "model")
    .drop("order")
)

# %%

plt.style.use("./paper.mplstyle")

colors = ["#CCCCCC", "#777777", "#222222"]
width, step_lang = 0.5, 0.6
step_model = 0.52
x = np.linspace(0, step_lang * 12, num=12)
x[::2] = x[1::2]
x[1::2] += width + 0.05
xticks = (x[::2] + x[1::2]) / 2

fig = plt.figure(figsize=(6.5, 2.2), constrained_layout=True)
subfigs = fig.subfigures(ncols=2, width_ratios=[1, 1], wspace=0.05)
ax = subfigs[0].subplots(nrows=2, sharey=True, gridspec_kw={"hspace": 0.3})
for i, select in enumerate(
    [[n.iso_639_3 for n in langs] for langs in (dev_languages(), test_languages())]
):
    subdf = df.filter(pl.col("language").is_in(select))
    bottom = np.zeros(len(select) * 2)
    for j, col in enumerate(["insertions", "substitutions", "deletions"]):
        ax[i].bar(
            x,
            subdf[col],
            bottom=bottom,
            label=col.capitalize() if i == 0 else None,
            width=width,
            facecolor=colors[j],
            edgecolor="black",
            linewidth=0.1,
            hatch=["", "///"] * 6,
        )
        ax[i].set_xticks(xticks)
        ax[i].set_xticklabels(subdf["language"][::2])
        bottom += subdf[col].to_numpy()
    ax[i].set_ylabel(r"PER (\%)", fontsize=9)
    ax[i].set_yticks([0, 25, 50, 75, 100])
    ax[i].tick_params(axis="both", labelsize=9)
    ax[i].spines[["top", "right"]].set_visible(False)

handles, labels = ax[0].get_legend_handles_labels()
handles = [
    handles[0],
    plt.Rectangle((0, 0), 1, 1, fill=False, lw=0.5),
    handles[1],
    plt.Rectangle((0, 0), 1, 1, fill=False, hatch="/" * 5, lw=0.5),
    handles[2],
]
labels = [labels[0], "Zero-shot", labels[1], "Finetuned on 10h", labels[2]]
ax[0].set_title("Recognition error types by model and language", fontsize=9)
subfigs[0].legend(
    handles=handles,
    labels=labels,
    fontsize=6,
    loc="center",
    bbox_to_anchor=(0.57, 0.45),
    ncols=3,
)


def relative_luminance(color):
    rgb = to_rgba_array(color)[:, :3]
    rgb = np.where(rgb <= 0.03928, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    lum = rgb.dot([0.2126, 0.7152, 0.0722])
    try:
        return lum.item()
    except ValueError:
        return lum


def plot_heatmap(ax: Axes, heatmap: np.ndarray) -> None:
    heatmap[np.diag_indices_from(heatmap)] = np.nan
    mesh = ax.pcolormesh(heatmap, cmap="magma_r")
    labels = [c.capitalize() if c != "SIL" else "SIL" for c in CLASS_ORDER]
    ax.set_xticks(
        np.arange(len(CLASS_ORDER)) + 0.5,
        labels=labels,
        rotation=30,
        fontsize=6,
        va="top",
    )
    ax.set_yticks(np.arange(len(CLASS_ORDER)) + 0.5, labels=labels, fontsize=6)
    height, width = heatmap.shape

    ax.set(xlim=(0, width), ylim=(0, height))
    ax.invert_yaxis()
    ax.spines[["top", "left", "right", "bottom"]].set_visible(False)
    mesh.update_scalarmappable()
    xpos, ypos = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
    for x, y, m, color, val in zip(
        xpos.flat,
        ypos.flat,
        mesh.get_array().flat,
        mesh.get_facecolors(),
        heatmap.flat,
    ):
        lum = relative_luminance(color)
        text_color = ".15" if lum > 0.408 else "w"
        ax.text(
            x,
            y,
            rf"{val:.1f}",
            ha="center",
            va="center",
            color=text_color,
            fontsize=6,
        )
    return mesh


ax2 = subfigs[1].subplots()
mesh = plot_heatmap(ax2, get_heatmap_array("spidr-vp20-ft-10h") * 100)
ax2.set_ylabel("Ground truth")
ax2.set_xlabel("Prediction", labelpad=0)
cbar = subfigs[1].colorbar(mesh, ax=ax2)
cbar.ax.tick_params(labelsize=6)
ax2.set_title(r"Substitution distribution for Finetuned on 10h (in \%)", fontsize=9)

fig.text(0.02, 0.05, r"\textbf{A}", fontsize=14, va="top")
fig.text(0.62, 0.05, r"\textbf{B}", fontsize=14, va="top")
fig.get_layout_engine().set(wspace=0.0, hspace=0.0, w_pad=0.0, h_pad=0.0)
plt.savefig("figures/details.pdf", bbox_inches="tight", pad_inches=0.02)
plt.show()

# %%


# %%

df.filter(pl.col("model") == "spidr-vp20-ft-10h").sort("per")
# %%
x = {}
for model, language in results.items():
    total_s, total_i, total_d = 0, 0, 0
    for lang, res in language.items():
        total_s += res["word_output"].substitutions
        total_i += res["word_output"].insertions
        total_d += res["word_output"].deletions
    x[model] = {
        "substitutions": total_s,
        "insertions": total_i,
        "deletions": total_d,
        "total": total_s + total_i + total_d,
    }
# %%
x
# %%
for m, res in x.items():
    print(
        m,
        f"{res['substitutions'] / x[m]['total']:.0%}",
        f"{res['insertions'] / x[m]['total']:.0%}",
        f"{res['deletions'] / x[m]['total']:.0%}",
    )

# %%
res
# %%

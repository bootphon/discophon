import logging
from uuid import uuid4

import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.patches import Rectangle

from discophon.core.languages import load_tipa

logger = logging.getLogger()


def test_tipa_mapping() -> None:
    mapping = load_tipa()
    rc("text", usetex=True)
    rc("text.latex", preamble=r"\usepackage{tipa}")
    n_items = len(mapping)
    cols = 4
    rows = (n_items + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, 3 * rows))
    fig.suptitle("Unicode to TIPA Verification", fontsize=16, y=0.995)
    axes_flat = axes.flatten()
    for idx, (unicode, tipa) in enumerate(mapping.items()):
        ax = axes_flat[idx]
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        rect = Rectangle((0.05, 0.1), 0.9, 0.8, linewidth=2, edgecolor="black", facecolor="lightgray", alpha=0.3)
        ax.add_patch(rect)
        ax.text(0.5, 0.75, unicode, ha="center", va="center", fontsize=30, usetex=False)
        ax.text(0.5, 0.45, tipa, ha="center", va="center", fontsize=9, family="monospace", color="blue", usetex=False)
        ax.text(0.5, 0.3, tipa, ha="center", va="center", fontsize=30, color="red", usetex=True)
    for idx in range(n_items, len(axes_flat)):
        axes_flat[idx].axis("off")
    output = f"tipa-mapping-{uuid4().hex[:6]}.pdf"
    logger.info("Saving TIPA mapping verification plot to %s", output)
    plt.tight_layout()
    plt.savefig(output)
    plt.close(fig)

from pathlib import Path

import polars as pl
import textgrids


def _read_single_textgrid(path: str | Path) -> dict[str, pl.DataFrame]:
    grid = textgrids.TextGrid(path)
    tiers = {}
    for name, tier in grid.items():
        if tier.is_point_tier:
            tiers[name] = pl.DataFrame([{"text": p.text or "SIL", "pos": p.xpos} for p in tier])
        else:
            tiers[name] = pl.DataFrame([{"text": p.text or "SIL", "start": p.xmin, "end": p.xmax} for p in tier])
        tiers[name] = tiers[name].with_columns(fileid=pl.lit(Path(path).stem))
    return tiers


def read_textgrid(path: str | Path) -> dict[str, pl.DataFrame]:
    if Path(path).is_file():
        return _read_single_textgrid(path)
    if Path(path).is_dir():
        textgrids = [_read_single_textgrid(p) for p in Path(path).glob("*.TextGrid")]
        return {name: pl.concat(textgrid[name] for textgrid in textgrids).sort("fileid") for name in textgrids[0]}
    raise ValueError(path)

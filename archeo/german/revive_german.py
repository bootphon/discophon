# %%
from pathlib import Path

import polars as pl

long_files = {
    "M1": "M16_N_B16_C24_ge",
    "M2": "M17_N_B16_C26_ge",
    "M3": "M18_N_B16_C39_ge",
    "M4": "M19_N_B17_C1_ge",
    "M5": "M20_N_B5_C23_ge",
    "M6": "M21_N_B22_C14_ge",
    "M7": "M22_N_B22_C2_ge",
    "M8": "M23_N_B23_C9_ge",
    "M9": "M24_N_B26_C4_ge",
    "M10": "M25_N_B27_C_ge",
    "F1": "F16_N_B19_C1_ge",
    "F2": "F17_N_B20_C2_ge",
    "F3": "F18_N_B21_C_ge",
    "F4": "F19_N_B24_C1_ge",
    "F5": "F20_N_B25_C3_ge",
    "F6": "F21_N_B28_C_ge",
    "F7": "F22_N_B29_C_ge",
    "F8": "F23_N_B30_C6_ge",
    "F9": "F24_N_B31_C_ge",
    "F10": "F25_N_B34_C_ge",
}
wavs = Path("/store/projects/zerospeech/archive/2017/raw_data/LANG2/German_corpus_challenge/wav_by_utt")
segments = pl.concat(
    [
        pl.read_csv(
            wavs / f"{f}.seg.txt",
            separator=" ",
            new_columns=["segment", "file", "onset", "offset"],
            schema_overrides={"onset": pl.String, "offset": pl.String},
        )
        for f in long_files.values()
    ]
)

align = (
    pl.read_csv(
        "/store/projects/zerospeech/archive/2017/archives/LANG2/output_corpus_creation_german/models/alignment_phone/alignment.txt",
        has_header=False,
        separator=" ",
        new_columns=["segment", "onset", "offset", "proba", "phone"],
        schema_overrides={"onset": pl.String, "offset": pl.String},
    )
    .drop("proba")
    # .filter(pl.col("segment").is_in(segments["segment"].implode()))
)


# %%
(segments["offset"] - segments["onset"]).sum() / 3600£
# %%
segments
# %%
align
# %%
segments
# %%
align
# %%
segments.filter(pl.col("segment").str.starts_with("M05_R_B5"))
# %%

# %%
import json
from pathlib import Path

import polars as pl
import soundfile as sf
from discophon.core.builder import build_items, remove_last_entry_from_item
from discophon.prepare import resample
from tqdm import tqdm

phonology = json.loads(Path("./builder/src/discophon/builder/rules/rules.json").read_text(encoding="utf-8"))["jpn"]
FORBIDDEN = {"spn", "ː", "ʔ", "tː͡s"}
ROOT = Path("/store/projects/phoneme_discovery/benchmark")
COMMONVOICE = Path("/store/data/raw_data/commonvoice/cv-corpus-22.0-2025-06-20/ja")
MANEL = Path("/store/projects/phoneme_discovery_benchmark/splits_improved/ja_improved/22")

cv = (
    pl.read_csv(COMMONVOICE / "validated.tsv", separator="\t", quote_char=None)
    .select("client_id", "path")
    .with_columns(pl.col("path").str.strip_suffix(".mp3").str.split("/").list.get(-1).alias("#file"))
    .rename({"client_id": "speaker"})
)

for split in ["dev", "test"]:
    df = pl.read_csv(
        MANEL / f"alignment/ja-{split}.align",
        separator=" ",
        schema_overrides={"onset": pl.String, "offset": pl.String},
    ).with_columns(pl.col("#phone").replace_strict(phonology | {"SIL": "SIL"}))
    files_with_forbidden = df.filter(pl.col("#phone").is_in(FORBIDDEN))["#file"].unique().to_list()
    df = df.filter(~pl.col("#file").is_in(files_with_forbidden))
    df.write_csv(ROOT / f"alignment/alignment-jpn-{split}.txt", separator=" ")

    phoneme, triphone = build_items(df.join(cv, on="#file", how="left"))
    phoneme = remove_last_entry_from_item(phoneme)
    triphone = remove_last_entry_from_item(triphone)
    phoneme.write_csv(ROOT / f"item/phoneme-jpn-{split}.item", separator=" ")
    triphone.write_csv(ROOT / f"item/triphone-jpn-{split}.item", separator=" ")

    manifest = []
    entries = (
        phoneme.join(cv, on="#file", how="left")[["#file", "speaker", "path"]]
        .unique()
        .sort("speaker", "#file")
        .iter_rows()
    )
    for file, speaker, path in tqdm(list(entries)):
        original = Path(COMMONVOICE / f"clips/{path}")
        if not (target := Path(ROOT / f"audio/jpn/all/{file}.wav")).is_file():
            resample(original, target, output_sample_rate=16_000)
        if not (symlink := Path(ROOT / f"audio/jpn/{split}/{file}.wav")).is_symlink():
            symlink.symlink_to(f"../all/{file}.wav")
        manifest.append({"fileid": file, "speaker": speaker, "num_samples": sf.info(target).frames})
    pl.DataFrame(manifest).write_csv(ROOT / f"manifest/manifest-jpn-{split}.csv")
# %%

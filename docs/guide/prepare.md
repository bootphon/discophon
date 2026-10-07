# Data preparation

To download the benchmark data, install the `discophon` package with the `prepare` extra:

```bash
pip install "discophon[prepare]"
```

Let's call `$DATA` the directory where you want to install the benchmark data.

## Download benchmark assets

The following command

```bash
python -m discophon.prepare download $DATA
```

will download:

- Manifests, alignments and item files
- Audio data for English, French, German, and Wolof [^1]
- Symlinks to audio files for each split (train-10h, train-1h, train-10min, dev, test)

If you prefer, you can manually download the data from: https://cognitive-ml.fr/downloads/phoneme-discovery/discophon_data.tar.gz.

[^1]: Audio data for the other languages is from Common Voice and cannot be redistributed. See the following section.

## Download and process Common Voice data

The audio for the other languages comes from [Common Voice Scripted Speech](https://datacollective.mozillafoundation.org/organization/cmfh0j9o10006ns07jq45h7xk), distributed by Mozilla Data Collective:

- Dev languages: *Swahili* (22 GB), *Tamil* (9 GB), *Thai* (9 GB), *Turkish* (3 GB), *Ukrainian* (3 GB)
- Test languages: *Basque* (15 GB), *Chinese (China)* (23 GB), *Japanese* (15 GB)

Before downloading:

1. Create an API key in your Mozilla Data Collective account and export it as `MDC_API_KEY`.
2. On the Mozilla Data Collective website, read and accept the terms of the latest *Common Voice Scripted Speech*
   release for each of these languages. Each new Common Voice release is a new dataset, so you have to accept
   its terms again.

Then run:

```bash
export MDC_API_KEY=...
python -m discophon.prepare commonvoice $DATA
```

For each language, this downloads the latest release to `$DATA/raw`, verifies its checksum, and converts the clips
listed in the manifests to 16 kHz WAV files in `$DATA/audio/{code}/all`, directly from the archive. The archive is
deleted afterwards. The directories `$DATA/audio/{code}/{split}` already contain symlinks to those files.

The command can be resumed if interrupted: the download restarts where it stopped, and the WAV files already written
are skipped. Languages that are fully prepared are skipped without downloading anything.

You can also prepare only some languages, for example to run them in parallel:

```bash
python -m discophon.prepare commonvoice $DATA swa tam
```

If a clip listed in the manifests is missing from the latest release, the command converts all the other clips,
and then fails with the list of missing clips. In that case, the dataset cannot be rebuilt from the current
Common Voice release: please [open an issue](https://github.com/bootphon/discophon/issues).

## Check the Common Voice releases

To check that the latest Common Voice releases still contain all the clips listed in the manifests, without
writing anything to disk, run:

```bash
python -m discophon.prepare commonvoice $DATA --check-only
```

Each release is streamed and only the names of its files are read. This still downloads the whole archives, but
stops early once all the clips of a language are found. The command exits with an error and lists the missing clips
if any language is incomplete. It only needs the manifests from the asset download above.

# Data preparation

You need the `sox` binary available in your `$PATH` for pre-processing audio files.

Let's say you want to install the benchmark data and assets in a directory `$DATA`.

## Download Common Voice data

You first need to download audio data from CommonVoice. You can use their API if you don't want to download large
files on your local computer.

Download everything in `$DATA/raw`.

Dev languages:

- [Common Voice Scripted Speech 23.0 - German](https://datacollective.mozillafoundation.org/datasets/cmflnuzw5p0q7ydlq4k8skhqi) (34.41 GB)
- [Common Voice Scripted Speech 23.0 - Swahili](https://datacollective.mozillafoundation.org/datasets/cmflnuzw7mjcay14kmowc4y96) (21.23 GB)
- [Common Voice Scripted Speech 23.0 - Tamil](https://datacollective.mozillafoundation.org/datasets/cmflnuzw73r9g1avrbu6bwkfx) (8.56 GB)
- [Common Voice Scripted Speech 23.0 - Thai](https://datacollective.mozillafoundation.org/datasets/cmflnuzw7fwn4fc969r5owufz) (8.35 GB)
- [Common Voice Scripted Speech 23.0 - Turkish](https://datacollective.mozillafoundation.org/datasets/cmflnuzw71qkz8x3kil3tgjvk) (2.73 GB)
- [Common Voice Scripted Speech 23.0 - Ukrainian](https://datacollective.mozillafoundation.org/datasets/cmflnuzw7ijdv5oe9u7ky0zrc) (2.55 GB)

Test languages:

- [Common Voice Scripted Speech 23.0 - Basque](https://datacollective.mozillafoundation.org/datasets/cmflnuzw5qoauo49kpf8y1gzp) (14.58 GB)
- [Common Voice Scripted Speech 23.0 - Chinese (China)](https://datacollective.mozillafoundation.org/datasets/cmflnuzw8fvgv2vdgt6f52qvh) (21.26 GB)
- [Common Voice Scripted Speech 23.0 - English](https://datacollective.mozillafoundation.org/datasets/cmflnuzw52mzok78yz6woemc1) (86.83 GB)
- [Common Voice Scripted Speech 23.0 - French](https://datacollective.mozillafoundation.org/datasets/cmflnuzw5ahjms0zbrcl0vg4e) (27.87 GB)
- [Common Voice Scripted Speech 23.0 - Japanese](https://datacollective.mozillafoundation.org/datasets/cmflnuzw5lv4n3cd25tbavjb9) (11.80 GB)
- Wolof data comes from a different source, and will be downloaded with the other assets in the following section.

Extract each archive, with `tar --strip-components=1 -xvf ...`.

For example, let's say your archive is named `mcv-scripted-uk-v23.0.tar.gz`.
Extract it with `tar --strip-components=1 -xvf mcv-scripted-uk-v23.0.tar.gz`, and move the output directory to
`$DATA/raw`.

You can delete the archives afterwards. You should have the following structure:

```bash
❯ tree -L 2 $DATA
$DATA
└── raw
    ├── de
    ├── en
    ├── eu
    ├── fr
    ├── ja
    ├── sw
    ├── ta
    ├── th
    ├── tr
    ├── uk
    └── zh-CN
```

## Download benchmark assets

Now download the benchmark assets with the following command:

```bash
python -m discophon.prepare download $DATA
```

This will download:
- Symlinks to audio files for each split in each language
- Manifests
- Alignments and item files

## Preprocess selected audio files

Now resample audio files and convert them to WAV with the command:

```bash
for code in swa tam tha tur ukr cmn eus jpn; do
    python -m discophon.prepare audio $DATA $code
done
```

This will create directories `$DATA/audio/cmn/all`, `$DATA/audio/deu/all`, `$DATA/audio/eng/all`, etc. with
resampled audio files. The directories corresponding to each split contain symlinks to those files.

You should parallelize this loop if you can. If you are in a SLURM cluster, you should also parallelize each dataset
processing across tasks or array jobs. The `discophon.prepare` package will automatically handle the distribution of
files to process across jobs.

You can delete the `$DATA/raw` folder afterwards.

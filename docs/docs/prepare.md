# Data preparation

To download the benchmark data, you need:

- The `discophon` package installed.
- The `sox` binary available in your `$PATH` to pre-process audio files.

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

[^1]: Audio data for the other languages is from CommonVoice and cannot be redistributed. See the following section.

## Download and process Common Voice data

Download the following datasets from [Common Voice Scripted](https://datacollective.mozillafoundation.org/organization/cmfh0j9o10006ns07jq45h7xk):

- Dev languages: *Swahili* (22 GB), *Tamil* (9 GB), *Thai* (9 GB), *Turkish* (3 GB), *Ukrainian* (3 GB)
- Test languages: *Basque* (15 GB), *Chinese (China)* (12 GB), *Japanese* (22 GB)

Since only the latest version is distributed, we cannot provide direct download links.
You can use their API or [their python package](https://github.com/Mozilla-Data-Collective/datacollective-python).

Extract each archive, with something like `tar --strip-components=1 -xvfz ...`, and move the output to `$DATA/raw`.
You can delete the archives afterwards. You should have the following structure:

```bash
❯ tree -L 2 $DATA
$DATA
└── raw
    ├── eu
    ├── ja
    ├── sw
    ├── ta
    ├── th
    ├── tr
    ├── uk
    └── zh-CN
```

Now resample audio files and convert them to WAV with the command:

```bash
for code in swa tam tha tur ukr cmn eus jpn; do
    python -m discophon.prepare audio $DATA $code
done
```

This will create directories `$DATA/audio/cmn/all`, `$DATA/audio/deu/all`, `$DATA/audio/eng/all`, etc., with
resampled audio files. The directories corresponding to each split contain symlinks to those files.

You should parallelize this loop if you can to speed things up. If you are in a SLURM cluster, you should also parallelize each dataset
processing across tasks or array jobs. The `discophon.prepare` package will automatically distribute the files to process to each job.

You can delete the `$DATA/raw` folder afterwards.

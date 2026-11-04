# Baselines

## Usage

The baselines and pretraining datasets are available on Hugging Face.

Baselines:

- SpidR MMS-ulab: https://huggingface.co/coml/spidr-mmsulab
- SpidR VP-20: https://huggingface.co/coml/spidr-vp20
- HuBERT MMS-ulab: https://huggingface.co/coml/hubert-base-mmsulab
- HuBERT VP-20: https://huggingface.co/coml/hubert-base-vp20

Datasets:

- MMS-ulab segmented dataset: https://huggingface.co/datasets/coml/mmsulab
- VP-20 dataset: https://huggingface.co/datasets/coml/vp20 (to be added soon)

### Models

First install the [`spidr`](https://github.com/facebookresearch/spidr) and
[`minimal_hubert`](https://github.com/mxmpl/minimal_hubert) libraries,
or directly:

```bash
pip install "discophon[baselines]"
```

#### SpidR checkpoints

```python
import joblib
from spidr.models import SpidR
from torch.hub import load_state_dict_from_url
from torch.nn import functional as F
from torchcodec.decoders import WavDecoder

state_dict = load_state_dict_from_url("https://huggingface.co/coml/spidr-vp20/resolve/main/final.pt")
model = SpidR().eval()
model.load_state_dict(state_dict)
wav = WavDecoder("/path/to/file.wav").get_all_samples().data  # (1, num_samples), 16 kHz mono
wav = F.layer_norm(wav, wav.shape)  # Normalized like in the extraction functions of discophon

# Training loss
mask_indices = ...  # Set up the indices of the masked frames
loss, _ = model(wav, mask_indices=mask_indices)

# Continuous representations
codebook_predictions = model.get_codebooks(wav)  # Log-probs from prediction heads
hidden_states = model.get_intermediate_outputs(wav)  # Hidden Transformer states

# Discrete units
layer = 6  # Target layer

# From prediction heads
units_from_heads = codebook_predictions[layer - 1].argmax(-1)

# From intermediate representations, using K-means
kmeans = joblib.load("/path/to/kmeans.joblib")
units_from_interm = kmeans.predict(hidden_states[layer - 1].squeeze(0).detach().numpy())
```

#### HuBERT checkpoints

- With Transformers (check out [their documentation](https://huggingface.co/docs/transformers/model_doc/hubert)
    for details):

    ```python
    from transformers import HubertModel

    model = HubertModel.from_pretrained("coml/hubert-base-vp20")
    ```


- With [`minimal_hubert`](https://github.com/mxmpl/minimal_hubert):

    ```python
    from minimal_hubert import HuBERT, HuBERTPretrain
    from torch.nn import functional as F
    from torchcodec.decoders import WavDecoder

    wav = WavDecoder("/path/to/file.wav").get_all_samples().data  # (1, num_samples), 16 kHz mono
    wav = F.layer_norm(wav, wav.shape)  # Normalized like in the extraction functions of discophon

    model = HuBERT.from_pretrained("coml/hubert-base-vp20").eval()
    model_from_pretraining = HuBERTPretrain.from_pretrained(
        "https://huggingface.co/coml/hubert-base-vp20/resolve/main/it2.pt"
    )

    # Training loss
    labels = ...  # Target cluster of each frame
    mask = ...  # Set up your boolean mask
    loss, _ = model_from_pretraining(wav, labels, mask=mask, attention_mask=None)

    # Intermediate Transformer representations (same convention as in fairseq)
    # Use this method if you want to get discrete units using K-means
    # that were trained in this project or in projects that used fairseq.
    feats = model.get_intermediate_outputs(wav)

    # Same as HF transformers, s3prl and torchaudio:
    # representations are taken at the end of the Transformer layer block
    # instead of just before the residual.
    feats_after_residual = model.get_intermediate_outputs(wav, before_residual=False)
    ```

### Datasets

We redistribute both pretraining datasets on HuggingFace Hub. You can access them directly if you have `datasets`
installed:

```python
from datasets import load_dataset

mmsulab = load_dataset("coml/mmsulab")
vp20 = load_dataset("coml/vp20")  # To be added soon
```

Check out their README for more details on their structure and how they were built.

## Replication

### SpidR pretraining

1.  Create the following TOML config file to `cfg.toml`:

    ```toml
    [data]
    manifest = "./manifests/train_manifest.jsonl"

    [validation]
    val.manifest = "./manifests/val_manifest.jsonl"

    [run]
    workdir = "./workdir"
    wandb_mode = "offline"# (1)!
    wandb_project = "discophon"
    wandb_name = "spidr-pretraining"
    model_type = "spidr"

    [run.slurm_validation]
    nodes = 1
    gpus_per_node = 1
    qos = "qos_gpu-dev"
    time = 60
    cpus_per_task = 10
    constraint = "v100-32g"
    ```

    1. > Set to `online` if your cluster has internet access and you want to log to Weights & Biases

    Adapt the paths and SLURM parameters to your setup.

2.  Launch pretraining with:

    ```bash
    python -m spidr ./cfg.toml \
        --nodes 4 \
        --gpus-per-node 4 \
        --cpus-per-task 24 \
        --time 1200 \
        --constraint h100 \
        --dump ./dump
    ```

    Again, adapt the SLURM parameters to your setup. This specific command will launch one job on 4 nodes with 4 H100 GPUs each, for 20 hours.
    The `--dump` argument will specify the directory where to dump the submitit output.

3.  You're done! Checkpoints will be available in `./workdir/discophon/spidr-pretraining`.

### HuBERT pretraining

Check out
[`minimal_hubert`'s README](https://github.com/mxmpl/minimal_hubert/blob/main/README.md#pretraining-hubert-step-by-step)
for easy pretraining. It involves multiple steps, but the pretraining part is very similar to SpidR's.

### Finetuning

Use the CLI utility:

```console
❯ python -m discophon.baselines --help
usage: discophon.baselines [-h] [--n-clusters N_CLUSTERS] [--layer LAYER]
                           {hubert,spidr} name project workdir checkpoint manifest

Baseline finetuning of HuBERT or SpidR

positional arguments:
  {hubert,spidr}        Model architecture
  name                  Run name
  project               Run project
  workdir               Working directory for checkpoints and Wandb logs
  checkpoint            Path to pretrained checkpoint
  manifest              Manifest file for finetuning

options:
  -h, --help            show this help message and exit
  --n-clusters N_CLUSTERS
                        Number of target clusters for HuBERT finetuning
  --layer LAYER         Target layer for HuBERT finetuning used to train the K-means
```

or use the [`finetune_spidr`][discophon.baselines.finetune_spidr]
and [`finetune_hubert`][discophon.baselines.finetune_hubert] functions.

The finetuned checkpoints of the four baselines, on each language with 10 minutes, 1 hour and 10 hours of data, are
in [coml/discophon-finetuned-baselines](https://huggingface.co/coml/discophon-finetuned-baselines). They were
obtained with the following recipe.

1. **Finetuning.** The manifest is one of the training manifests of the dataset, such as
   `/path/to/discophon_data/manifest/manifest-deu-train-1h.csv`. Both models are finetuned with the same optimizer
   and schedule, and the checkpoints `step_*.pt` are saved in `{workdir}/{project}/{name}/`.
   SpidR keeps its pretraining objective. HuBERT needs new targets: before finetuning, a K-means with
   `--n-clusters` clusters is fitted on the features of the pretrained model at `--layer`, on the finetuning data,
   and saved next to the checkpoints as `km{n_clusters}-targets-l{layer}.joblib`. The baselines use 256 clusters,
   at layer 11 for `hubert-base-vp20` and layer 12 for `hubert-base-mmsulab`.
2. **Checkpoint selection.**
   [`validate_all_spidr_checkpoints`][discophon.baselines.validate_all_spidr_checkpoints] and
   [`validate_all_hubert_checkpoints`][discophon.baselines.validate_all_hubert_checkpoints] compute the validation
   loss of every checkpoint on the dev split of the finetuning language (such as
   `/path/to/discophon_data/manifest/manifest-deu-dev.csv`), with the same masks and crops, and link the best one to
   `best.pt`. For HuBERT, the
   validation targets come from the same targets K-means, on the features of the pretrained model.
3. **Discrete units.** SpidR units come directly from the prediction heads of `best.pt`. HuBERT units come from a new
   K-means with 256 clusters for each layer, fitted on the features of `best.pt` on the finetuning data. These
   K-means are in the [artifacts dataset](https://huggingface.co/datasets/coml/discophon-artifacts), next to the
   units, as `{key}/ft-{language}-{duration}/many_to_one/{layer}/km256-ft-{language}-{duration}-l{layer}.joblib`.
   Then extract the units as [below](#discrete-units).

### Discrete units

We provide utilities to easily extract discrete units with the functions
[`extract_spidr_discrete_units`][discophon.baselines.extract_spidr_discrete_units]
and [`extract_hubert_discrete_units`][discophon.baselines.extract_hubert_discrete_units]:

```python
import joblib
from huggingface_hub import hf_hub_download
from discophon.baselines import extract_hubert_discrete_units, extract_spidr_discrete_units

language = "German"  # Either name or ISO 639-3 code
split = "test"  # Or "dev", "train-10min", "train-1h", "train-10h"

# HuBERT units, only layer 11, with the K-means of the artifacts dataset
path_kmeans = hf_hub_download(
    "coml/discophon-artifacts",
    "hubert-vp20-it2/zero-shot/many_to_one/11/km256-vp20-l11.joblib",
    repo_type="dataset",
)
kmeans = joblib.load(path_kmeans)
extract_hubert_discrete_units(
    "/path/to/discophon_data",
    path_units="/path/to/scores/hubert-base-vp20",
    language=language,
    split=split,
    pretrained_model_name_or_path="coml/hubert-base-vp20",
    kmeans_by_layer={11: kmeans},
    layers=11,
)

# SpidR units, all layers
extract_spidr_discrete_units(
    "/path/to/discophon_data",
    path_units="/path/to/scores/spidr-vp20",
    language=language,
    split=split,
    checkpoint="/path/to/spidr-vp20/final.pt",
    layers=None,  # Default value
)
```

The units of each layer are written to `{path_units}/{layer}/units-{code}-{split}.jsonl`, the layout expected by
[the benchmark](evaluate.md#high-level). Continuous features are extracted with
[`extract_spidr_continuous_features`][discophon.baselines.extract_spidr_continuous_features]
and [`extract_hubert_continuous_features`][discophon.baselines.extract_hubert_continuous_features].
The extraction can be resumed if interrupted: the files already extracted are skipped.

Or use the CLI, which extracts all languages on the dev and test splits by default:

```console
❯ python -m discophon.baselines.extract hubert units /path/to/discophon_data /path/to/units coml/hubert-base-vp20 \
    --kmeans 11=/path/to/discophon-artifacts/hubert-vp20-it2/zero-shot/many_to_one/11/km256-vp20-l11.joblib
❯ python -m discophon.baselines.extract spidr features /path/to/discophon_data /path/to/features \
    /path/to/spidr-vp20/final.pt --layers 6 --batch-size 16
```

HuBERT is processed one utterance at a time: padding changes the statistics of its GroupNorm, so batches would
give different features. SpidR gives the same results with batches, up to numerical precision.

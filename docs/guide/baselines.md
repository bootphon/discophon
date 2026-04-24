# Baselines

## Usage

The baselines and pretraining datasets are available on Hugging Face.

Baselines[^1]:

- SpidR MMS-ulab: https://huggingface.co/coml/spidr-mmsulab
- SpidR VP-20: https://huggingface.co/coml/spidr-vp20
- HuBERT MMS-ulab: https://huggingface.co/coml/hubert-base-mmsulab
- HuBERT VP-20: https://huggingface.co/coml/hubert-base-vp20

[^1]: For HuBERT, we also provide checkpoints compatible with Transformers or torchaudio.

Datasets:

- MMS-ulab segmented dataset: https://huggingface.co/datasets/coml/mmsulab
- VP-20 dataset: https://huggingface.co/datasets/coml/vp20

### Models

First install the [`spidr`](https://github.com/facebookresearch/spidr) and
[`minimal_hubert`](https://github.com/mxmpl/minimal_hubert) libraries,
or directly:

```bash
pip install discophon[baselines]
```

#### SpidR checkpoints

```python
import joblib
from spidr import SpidR
from torchcodec.decoders import AudioDecoder

model = SpidR.from_pretrained("coml/spidr-vp20")
wav = AudioDecoder("/path/to/file.wav").get_all_samples().data

# Training loss
mask = ...  # Set up your boolean mask
loss, _ = model(wav, mask=mask)

# Continuous representations
codebook_predictions = model.get_codebooks(wav)  # Log-probs from prediction heads
hidden_states = model.get_intermediate_outputs(wav)  # Hidden Transformer states

# Discrete units
layer = 6  # Target layer

# From prediction heads
units_from_heads = codebook_predictions[layer - 1].argmax(-1)

# From intermediate representations, using K-means
kmeans = joblib.load("/path/to/kmeans.joblib")
units_from_interm = kmeans.predict(hidden_states[layer - 1])
```

#### HuBERT checkpoints

- With Transformers (check out [their documentation](https://huggingface.co/docs/transformers/model_doc/hubert)
    for details):

    ```python
    from transformers import HubertModel

    model = HubertModel.from_pretrained("coml/hubert-vp20")
    ```


- With [`minimal_hubert`](https://github.com/mxmpl/minimal_hubert):

    ```python
    from minimal_hubert import HuBERT, HuBERTPretrain

    model = HuBERT.from_pretrained("coml/hubert-vp20")
    model_from_pretraining = HuBERTPretrain.from_pretrained(
        "https://huggingface.co/coml/hubert-base-vp20/resolve/main/it2.pt"
    )

    # Training loss
    loss, _ = model_from_pretraining(wav, mask=mask)

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
vp20 = load_dataset("coml/vp20")
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

    1. > Set to `online` if you cluster has internet access and you want to log to Weights & Biases

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
usage: python -m discophon.baselines [-h] [--n-clusters N_CLUSTERS] [--layer LAYER]
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
and [`finetune_hubert`][discophon.baselines.finetune_spidr] functions.

### Discrete units

Coming soon!

# Baselines

## Usage

The pretrained baselines and datasets are available on Hugging Face.

Models[^1]:

- SpidR MMS-ulab: https://huggingface.co/coml/spidr-mmsulab
- SpidR VP-20: https://huggingface.co/coml/spidr-vp20
- HuBERT MMS-ulab: https://huggingface.co/coml/hubert-mmsulab
- HuBERT VP-20: https://huggingface.co/coml/hubert-vp20

[^1]: For HuBERT, we also provide checkpoints compatible with Transformers or torchaudio.

Datasets:

- MMS-ulab pre-segmented dataset: https://huggingface.co/coml/mmsulab
- VP-20 dataset: https://huggingface.co/coml/vp20

### Models

First install

```bash
pip install discophon[baselines]
```

This will install the `spidr` and [`minimal_hubert`](https://github.com/mxmpl/minimal_hubert) libraries.

### Datasets

## Replication

### Pretraining

#### SpidR

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

#### HuBERT

### Finetuning

Work-in-progress!

### Discrete units

Work-in-progress!

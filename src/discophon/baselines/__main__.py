import argparse
from pathlib import Path

from discophon.baselines.hubert import finetune_hubert
from discophon.baselines.spidr import finetune_spidr

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline finetuning of HuBERT or SpidR")
    parser.add_argument("architecture", type=str, choices=["hubert", "spidr"], help="Model architecture")
    parser.add_argument("name", type=str, help="Run name")
    parser.add_argument("project", type=str, help="Run project")
    parser.add_argument("workdir", type=Path, help="Working directory for checkpoints and Wandb logs")
    parser.add_argument("checkpoint", type=Path, help="Path to pretrained checkpoint")
    parser.add_argument("manifest", type=str, help="Manifest file for finetuning")
    parser.add_argument("--n-clusters", type=int, help="Number of target clusters for HuBERT finetuning")
    parser.add_argument("--layer", type=int, help="Target layer for HuBERT finetuning used to train the K-means")
    args = parser.parse_args()

    if args.architecture == "hubert":
        if args.n_clusters is None or args.layer is None:
            parser.error("When finetuning HuBERT, you have to specify both `--n-clusters` and `--layer`.")
        finetune_hubert(
            args.name,
            args.project,
            args.workdir,
            args.checkpoint,
            args.manifest,
            n_clusters=args.n_clusters,
            target_layer=args.layer,
        )
    else:
        finetune_spidr(args.name, args.project, args.workdir, args.checkpoint, args.manifest)

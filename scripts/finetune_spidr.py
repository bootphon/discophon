"""Training loop."""

import math
from contextlib import ExitStack
from pathlib import Path

import torch
import wandb
from spidr.checkpoint import Checkpointer
from spidr.config import DataConfig, MaskingConfig
from spidr.data import build_dataloader
from spidr.environment import set_seed, setup_environment, setup_pytorch
from spidr.models import build_model
from spidr.tools import AverageMeters, profiler_context
from torch import GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW, Optimizer, lr_scheduler
from tqdm import tqdm


def tristage_scheduler(
    opt: Optimizer,
    *,
    warmup_steps: int,
    hold_steps: int,
    decay_steps: int,
    init_lr_scale: float = 1e-2,
    final_lr_scale: float = 1e-2,
) -> lr_scheduler.SequentialLR:
    warmup = lr_scheduler.LinearLR(opt, start_factor=init_lr_scale, total_iters=warmup_steps)
    hold = lr_scheduler.LinearLR(opt, start_factor=1.0, total_iters=hold_steps)
    decay = lr_scheduler.LambdaLR(opt, lambda step: math.exp(math.log(final_lr_scale) * step / decay_steps))
    return lr_scheduler.SequentialLR(opt, [warmup, hold, decay], [warmup_steps, hold_steps + warmup_steps])


def spidr_ft_data_config(manifest: str) -> DataConfig:
    return DataConfig(
        manifest,
        min_sample_size=0,
        max_sample_size=320_000,
        max_batch_length=3_800_000,
        num_buckets=5,
        num_workers=10,
        prefetch_factor=2,
        bucket_method="percentile",
    )


def finetune_spidr(name: str, workdir: Path, checkpoint: Path, manifest: str) -> None:
    max_steps, seed = 20_000, 0
    with ExitStack() as stack:
        set_seed(seed)
        setup_pytorch(use_deterministic=False)
        setup_environment()
        rundir = workdir / "discophon-spidr-ft" / name
        rundir.mkdir(parents=True, exist_ok=True)
        wandb.init(project="discophon-spidr-ft", name=name, mode="offline", dir=workdir)
        stack.callback(wandb.finish)
        device = torch.device("cuda")
        model = build_model(model_type="spidr", checkpoint=checkpoint).to(device).train()
        optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01, betas=(0.9, 0.95), eps=1e-6, fused=True)
        scaler = GradScaler("cuda")
        scheduler = tristage_scheduler(optimizer, warmup_steps=600, hold_steps=9_400, decay_steps=10_000)
        loader = build_dataloader(spidr_ft_data_config(manifest), MaskingConfig())
        ckpt = Checkpointer(rundir, 1_000)
        ckpt.init_state(model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler)
        step, epoch = int(ckpt.step), int(ckpt.epoch)
        stack.callback(lambda: ckpt.save(step, epoch))
        meters = AverageMeters(["loss", "grad_norm", "batch_size", "target_ppl", "pred_ppl"], device=device)
        profiler = stack.enter_context(profiler_context(rundir / "trace.html"))
        pbar = stack.enter_context(tqdm(total=max_steps, initial=step))
        if torch.cuda.get_device_capability() >= (8, 0):
            model.compile(dynamic=True)
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
        while step < max_steps:
            epoch += 1
            loader.batch_sampler.set_epoch(epoch)
            for waveforms, attn_mask, mask in loader:
                if step >= max_steps:
                    break
                with torch.autocast("cuda", dtype):
                    loss, outputs = model(
                        waveforms.to(device),
                        mask=mask.to(device),
                        attention_mask=attn_mask.to(device),
                    )
                loss = loss.mean()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = clip_grad_norm_(model.parameters(), 10.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                lr = scheduler.get_last_lr()[0]
                scheduler.step()
                step += 1
                meters.update(
                    loss=loss.detach(),
                    batch_size=waveforms.size(0),
                    grad_norm=grad_norm,
                    target_ppl=outputs["target_ppl"],
                    pred_ppl=outputs["pred_ppl"],
                )
                pbar.update()
                if step % 200 == 0:
                    infos = meters.pop() | {"lr": lr, "step": step, "epoch": epoch}
                    wandb.log(infos)
                    pbar.set_postfix(loss=infos["loss"], target_ppl=infos["target_ppl"], pred_ppl=infos["pred_ppl"])
                ckpt.save(step, epoch)
                profiler.step()
        ckpt.save_final(step, epoch)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("name", type=str)
    parser.add_argument("workdir", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("manifest", type=str)
    args = parser.parse_args()
    finetune_spidr(args.name, args.workdir, args.checkpoint, args.manifest)

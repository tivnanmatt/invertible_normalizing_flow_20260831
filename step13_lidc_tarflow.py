#!/usr/bin/env python
"""step13_lidc_tarflow.py -- Unconditional TarFlow priors on LIDC CT slices, one per resolution.

The step-9 recipe (official apple/ml-tarflow model, uniform dequantization, AdamW,
cosine schedule with a one-epoch warm-up, official bits/dim, early stopping on
the validation bpd, sample grids, memorisation check) applied to the step-12
LIDC slices at 32, 64, 128 and 256 px. One config per resolution
(configs/step13_lidc_tarflow_{32,64,128,256}.yml) chooses the patch size and the
transformer width/depth; the code is the same.

Differences from step 9, all driven by the data size and the resolution:
  * the slices come from the step-12 memmap caches (num_workers = 0, the
    container's /dev/shm is 64 MB); an "epoch" may be a fresh random subset of
    train.images_per_epoch images (0 = the whole training split), so that the
    validation bpd, checkpoints and sample grids are spaced by a fixed number of
    images at every resolution;
  * TF32 matmuls during training if `tf32` is set (the 4090 is 4-8x faster);
    every reported bits/dim is evaluated with TF32 OFF (exact fp32);
  * the memorisation check compares against a random subset of the training
    split (nn_train_subset images) instead of the whole split;
  * a fixed validation subset (train.val_subset, 0 = all) keeps the per-epoch
    evaluation short at high resolution.

Outputs (output_root/): as step 9 -- data/{metrics.csv, model.pth, model_best.pth,
  ckpt.pth, result.json, test_bpd_per_image_*.npy, summary.tex, provenance.json},
  figures/{real_grid.png, samples_epochNNN.png, final_samples.png, curves.png,
  nearest_neighbours.png, test_worst_image_*.png}.

Usage:
  python step13_lidc_tarflow.py --config configs/step13_lidc_tarflow_32.yml            # train (auto-resumes)
  python step13_lidc_tarflow.py --config configs/step13_lidc_tarflow_32.yml --collect  # figures/fragments only
Library: load_prior(cfg, device, which) as in step 9; build_bundle(cfg).
"""

import argparse
import csv
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision as tv
import yaml
from torch.utils.data import DataLoader, Sampler, Subset

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "step13_lidc_tarflow_32.yml"

import step2_tarflow as s2  # noqa: E402
import step9_chestmnist_tarflow as s9  # noqa: E402
import step12_lidc_dataset as s12  # noqa: E402

build_model, load_prior, per_image_bpd, bpd_stats, sample = (
    s9.build_model, s9.load_prior, s9.per_image_bpd, s9.bpd_stats, s9.sample)


def load_config(path=DEFAULT_CONFIG):
    with open(path) as f:
        return yaml.safe_load(f)


def resolve(p):
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def build_bundle(cfg):
    return s12.build_lidc(cfg["res"], s12.load_config(resolve(cfg["step12_config"])))


class EpochSubsetSampler(Sampler):
    """A fresh random subset of `n` indices per epoch (all of them, shuffled, if n = 0);
    the epoch counter is set by the training loop so a resumed run repeats nothing."""

    def __init__(self, size, n, seed):
        self.size, self.n, self.seed, self.epoch = size, n or size, seed, 0

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed + 1000 * self.epoch)
        return iter(torch.randperm(self.size, generator=g)[:self.n].tolist())

    def __len__(self):
        return min(self.n, self.size)


def make_loaders(cfg, device):
    bundle = build_bundle(cfg)
    tr = cfg["train"]
    kw = dict(batch_size=tr["batch"], num_workers=0, pin_memory=device.type == "cuda")
    g = torch.Generator().manual_seed(cfg["seed"])

    def subset(ds, n):  # 0 = whole split; a fixed seeded subset otherwise
        if n and n < len(ds):
            return Subset(ds, torch.randperm(len(ds), generator=g)[:n].tolist())
        return ds

    train_ds = s2.TarFlowInput(bundle.train, hflip=cfg.get("hflip", False))
    sampler = EpochSubsetSampler(len(train_ds), tr.get("images_per_epoch", 0), cfg["seed"])
    train = DataLoader(train_ds, sampler=sampler, drop_last=True, **kw)
    val = DataLoader(subset(s2.TarFlowInput(bundle.val), tr.get("val_subset", 0)), shuffle=False, **kw)
    test = DataLoader(subset(s2.TarFlowInput(bundle.test), tr.get("test_subset", 0)), shuffle=False, **kw)
    return train, val, test, bundle, sampler


def set_tf32(on):
    torch.backends.cuda.matmul.allow_tf32 = bool(on)
    torch.backends.cudnn.allow_tf32 = bool(on)
    torch.set_float32_matmul_precision("high" if on else "highest")


def latent_shape(cfg):
    return (cfg["img_size"] // cfg["patch_size"]) ** 2, cfg["channel_size"] * cfg["patch_size"] ** 2


def forward_checkpointed(model, x):
    """Model.forward (patchify, blocks in order, summed log-determinants) with the
    activations of each block recomputed in the backward pass: the stored memory is one
    block instead of all of them, for ~30% more compute. Used when train.block_checkpoint
    is set (T = 256 tokens and 48 layers at batch 64 do not fit in 24 GB otherwise).
    Mathematically identical to model(x) -- the same modules run in the same order."""
    from torch.utils.checkpoint import checkpoint
    x = model.patchify(x)
    logdets = torch.zeros((), device=x.device)
    for block in model.blocks:
        x, logdet = checkpoint(block, x, None, use_reentrant=False)
        logdets = logdets + logdet
    return x, logdets


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------

def train(cfg, device, out_data, out_figs):
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    tr = cfg["train"]
    tag = f"[step13:{cfg['res']}]"
    train_loader, val_loader, test_loader, bundle, sampler = make_loaders(cfg, device)
    model = build_model(cfg, device)
    n_params = sum(p.numel() for p in model.parameters())
    T, D = latent_shape(cfg)
    print(f"{tag} {n_params/1e6:.1f}M params, T={T} tokens x D={D}; train {len(train_loader.dataset)} "
          f"({len(sampler)} per epoch) val {len(val_loader.dataset)} test {len(test_loader.dataset)}; tf32={cfg.get('tf32', False)}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.95), lr=tr["lr"], weight_decay=tr["weight_decay"])
    steps_per_epoch = len(train_loader)
    total_steps, warmup = tr["epochs"] * steps_per_epoch, steps_per_epoch

    def lr_at(step):  # official utils.CosineLRSchedule shape
        min_lr, max_lr = 1e-6, tr["lr"]
        if step <= warmup:
            return min_lr + step / warmup * (max_lr - min_lr)
        t = (step - warmup) / max(1, total_steps - warmup)
        return min_lr + 0.5 * (1 + math.cos(math.pi * t)) * (max_lr - min_lr)

    n_show = tr["sample_nrow"] * tr["sample_rows"]
    fixed_noise = torch.randn(n_show, T, D, generator=torch.Generator().manual_seed(cfg["seed"] + 1)).to(device)
    real = torch.stack([train_loader.dataset[i][0] for i in range(n_show)])
    tv.utils.save_image((real + 1) / 2, out_figs / "real_grid.png", nrow=tr["sample_nrow"])

    # The validation statistic that selects the checkpoint and drives the early
    # stop: "mean" is the official bits/dim; "robust" is the mean over the
    # validation images below 10 bpd (the outlier accounting of step 9), because
    # a single validation slice on which the autoregressive flow blows up moves
    # the mean of 4000 by more than an epoch of progress and the plain rule then
    # stops a run that is still improving (the 64-px run, epochs 12-17).
    val_stat = tr.get("val_stat", "mean")
    columns = ["epoch", "loss", "mse_z", "logdet", "lr", "grad_norm", "val_bpd", "val_bpd_robust", "val_n_fail",
               "epoch_seconds", "sample_seconds", "n_nonfinite"]
    metrics_path, ckpt_path = out_data / "metrics.csv", out_data / "ckpt.pth"
    start_epoch, step, val_bpds, best, patience_ref = 0, 0, [], (float("inf"), 0), 0
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch, step, val_bpds = ckpt["epoch"], ckpt["step"], ckpt["val_bpds"]
        best = tuple(ckpt["best"])
        if ckpt.get("val_stat", "mean") != val_stat:
            # the selection statistic changed at this resume: the stored best score is kept as
            # the value to beat, but the patience is counted from here, not from its epoch
            patience_ref = start_epoch
        print(f"{tag} resumed at epoch {start_epoch} (best val bpd {best[0]:.4f} @ {best[1]}, "
              f"selection statistic '{val_stat}'" + (f", changed from '{ckpt.get('val_stat', 'mean')}': patience "
              f"counted from epoch {start_epoch}" if patience_ref else "") + ")", flush=True)
        with open(metrics_path, newline="") as f:
            rows = list(csv.reader(f))
        if rows and rows[0] != columns:      # metrics written before the robust columns existed: pad them
            old = rows[0]
            rows = [columns] + [[r[old.index(c)] if c in old else "" for c in columns] for r in rows[1:]]
            with open(metrics_path, "w", newline="") as f:
                csv.writer(f).writerows(rows)
    else:
        with open(metrics_path, "w", newline="") as f:
            csv.writer(f).writerow(columns)

    def early_stop(epochs_done):
        patience = tr.get("early_stop_patience")
        return bool(patience) and bool(val_bpds) and epochs_done - max(best[1], patience_ref) >= patience

    def validate():
        """(official mean, robust mean, number of images above 10 bpd) on the validation subset."""
        set_tf32(False)
        st = bpd_stats(per_image_bpd(model, val_loader, device, seed=cfg["seed"])[0])
        set_tf32(cfg.get("tf32", False))
        return st["mean"], st["mean_excl_outliers"], st["n_over_thresh"]

    def sample_grid(name):
        set_tf32(False)
        t0 = time.time()
        s = sample(model, fixed_noise, batch=tr.get("sample_batch", 64))
        n_bad = int((~torch.isfinite(s)).flatten(1).any(1).sum())
        tv.utils.save_image((s.clamp(-1, 1).nan_to_num() + 1) / 2, out_figs / name, nrow=tr["sample_nrow"])
        set_tf32(cfg.get("tf32", False))
        return n_bad, time.time() - t0

    t_start = time.time()
    stopped_early = early_stop(start_epoch)
    if stopped_early:
        print(f"{tag} resumed run already meets the early-stop rule: skipping training", flush=True)
    set_tf32(cfg.get("tf32", False))
    for epoch in range(start_epoch, tr["epochs"]):
        if stopped_early:
            break
        sampler.epoch = epoch
        t0 = time.time()
        sums = {"loss": 0.0, "mse": 0.0, "ld": 0.0, "gn": 0.0}
        nb, n_nonfinite = 0, 0
        for x, _ in train_loader:
            x = s2.apply_noise(x.to(device, non_blocking=True), cfg)
            step += 1
            for pg in optimizer.param_groups:
                pg["lr"] = lr_at(step)
            optimizer.zero_grad(set_to_none=True)
            if tr.get("block_checkpoint", False):
                z, logdets = forward_checkpointed(model, x)
            else:
                z, _, logdets = model(x)
            loss = model.get_loss(z, logdets)
            if not torch.isfinite(loss):
                n_nonfinite += 1
                continue
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), tr["grad_clip"])
            optimizer.step()
            sums["loss"] += loss.item()
            sums["mse"] += 0.5 * (z.detach() ** 2).mean().item()
            sums["ld"] += logdets.detach().mean().item()
            sums["gn"] += float(gn)
            nb += 1
            if nb % tr.get("log_every", 200) == 0:
                peak = f" peak {torch.cuda.max_memory_allocated()/2**30:.1f} GB" if device.type == "cuda" else ""
                print(f"{tag}   epoch {epoch+1} step {nb}/{steps_per_epoch} loss {sums['loss']/nb:.4f} "
                      f"gn {sums['gn']/nb:.2f} lr {lr_at(step):.2e} {(time.time()-t0)/nb:.3f} s/step{peak}", flush=True)
        train_seconds = time.time() - t0

        val_bpd = val_robust = val_fail = ""
        if (epoch + 1) % tr["eval_every"] == 0 or epoch + 1 == tr["epochs"]:
            val_bpd, val_robust, val_fail = validate()
            val_bpds.append((epoch + 1, val_bpd, val_robust, val_fail))
            score = val_robust if val_stat == "robust" else val_bpd
            if score < best[0]:
                best = (score, epoch + 1)
                torch.save(model.state_dict(), out_data / "model_best.pth")
        stopped_early = early_stop(epoch + 1)
        n_bad, sample_seconds = 0, 0.0
        if (epoch + 1) % tr["sample_every"] == 0 or epoch + 1 == tr["epochs"] or stopped_early:
            n_bad, sample_seconds = sample_grid(f"samples_epoch{epoch+1:03d}.png")
        dt = time.time() - t0
        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, sums["loss"] / max(nb, 1), sums["mse"] / max(nb, 1), sums["ld"] / max(nb, 1),
                                    lr_at(step), sums["gn"] / max(nb, 1), val_bpd, val_robust, val_fail,
                                    round(train_seconds, 1), round(sample_seconds, 1), n_nonfinite])
        msg = (f"{tag} epoch {epoch+1}/{tr['epochs']} loss {sums['loss']/max(nb,1):.4f} gn {sums['gn']/max(nb,1):.2f} "
               f"({train_seconds:.0f}s train, {dt:.0f}s total)")
        if val_bpd != "":
            msg += (f" val_bpd {val_bpd:.4f} (robust {val_robust:.4f}, {val_fail} above 10) "
                    f"(best {best[0]:.4f} @ {best[1]})")
        if sample_seconds:
            msg += f" samples {sample_seconds:.0f}s" + (f" NONFINITE {n_bad}/{n_show}" if n_bad else "")
        if n_nonfinite:
            msg += f" NONFINITE-LOSS {n_nonfinite}"
        print(msg, flush=True)
        if stopped_early:
            print(f"{tag} early stop at epoch {epoch+1}: no val improvement for {tr['early_stop_patience']} epochs "
                  f"(best {best[0]:.4f} @ {best[1]})", flush=True)
        torch.save(model.state_dict(), out_data / "model.pth")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch + 1, "step": step,
                    "val_bpds": val_bpds, "best": list(best), "val_stat": val_stat}, ckpt_path)
    train_minutes = (time.time() - t_start) / 60

    # ---- final evaluation (exact fp32) ---------------------------------
    set_tf32(False)
    epochs_run = val_bpds[-1][0] if val_bpds else tr["epochs"]
    result = dict(n_params=n_params, config=cfg, train_minutes=round(train_minutes, 1), epochs_run=epochs_run,
                  stopped_early=bool(stopped_early), val_bpd_curve=val_bpds, best_val_bpd=best[0], best_epoch=best[1],
                  val_stat=val_stat,
                  n_train=len(train_loader.dataset), images_per_epoch=len(sampler), n_val=len(val_loader.dataset),
                  n_test=len(test_loader.dataset), latent=dict(T=T, D=D))
    for which in ["final", "best"]:
        if which == "best":
            model.load_state_dict(torch.load(out_data / "model_best.pth", map_location=device))
        bpds, worst = per_image_bpd(model, test_loader, device, seed=cfg["seed"])
        st = bpd_stats(bpds)
        result[f"test_bpd_{which}"] = st["mean"]
        result[f"test_bpd_{which}_stats"] = dict(st, worst_max_abs_z_per_block=worst[2])
        np.save(out_data / f"test_bpd_per_image_{which}.npy", bpds)
        tv.utils.save_image((worst[1] + 1) / 2, out_figs / f"test_worst_image_{which}.png")
        print(f"{tag} TEST bpd ({which}{'' if which == 'final' else f', best-val epoch {best[1]}'}): mean {st['mean']:.4g}, "
              f"median {st['median']:.4f}, max {st['max']:.4g} (image {st['argmax']}), {st['n_over_thresh']} images > "
              f"{st['outlier_bpd']} bpd, mean without them {st['mean_excl_outliers']:.4f}; worst image max|z| per block "
              + " ".join(f"{v:.1e}" for v in worst[2]), flush=True)
    bpds, _ = per_image_bpd(model, val_loader, device, seed=cfg["seed"])
    result["val_bpd_best_stats"] = bpd_stats(bpds)

    torch.manual_seed(cfg["seed"] + 2)
    noise = torch.randn_like(fixed_noise)
    t0 = time.time()
    s = sample(model, noise, batch=tr.get("sample_batch", 64))
    result["final_sample_seconds"] = time.time() - t0
    result["final_samples_nonfinite"] = int((~torch.isfinite(s)).flatten(1).any(1).sum())
    result["final_samples_frac_outside_range"] = float(((s.abs() > 1).float().mean()))
    s = s.clamp(-1, 1).nan_to_num()
    tv.utils.save_image((s + 1) / 2, out_figs / "final_samples.png", nrow=tr["sample_nrow"])
    result.update(nearest_neighbour_check(s[:cfg["nn_check_samples"]], bundle, val_loader, device, out_figs,
                                          cfg.get("nn_train_subset", 20000), cfg["seed"]))
    with open(out_data / "result.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


@torch.no_grad()
def nearest_neighbour_check(samples, bundle, val_loader, device, out_figs, n_train, seed):
    """L2 nearest image among a random subset of the training split, for the samples
    and for held-out val images (chunked; the subset stays on the CPU as uint8)."""
    train_ds = s2.TarFlowInput(bundle.train)
    idx = torch.randperm(len(train_ds), generator=torch.Generator().manual_seed(seed + 3))[:n_train].tolist()
    train_u8 = torch.stack([((train_ds[i][0] + 1) * 127.5).round().to(torch.uint8) for i in idx]).flatten(1)  # (n, D)

    def nn_dist(q):
        q = q.to(device).flatten(1)
        best_d = torch.full((q.size(0),), float("inf"), device=device)
        best_i = torch.zeros(q.size(0), dtype=torch.long, device=device)
        for i in range(0, train_u8.size(0), 2048):
            ref = train_u8[i:i + 2048].to(device).float() / 127.5 - 1
            d = torch.cdist(q, ref)
            m, j = d.min(1)
            upd = m < best_d
            best_d, best_i = torch.where(upd, m, best_d), torch.where(upd, j + i, best_i)
        return best_d / math.sqrt(q.size(1)), best_i.cpu()   # RMS distance in [-1, 1] units

    d_s, i_s = nn_dist(samples)
    val_x = []
    for x, _ in val_loader:
        val_x.append(x)
        if sum(v.size(0) for v in val_x) >= 256:
            break
    d_v, _ = nn_dist(torch.cat(val_x)[:256])
    n = samples.size(0)
    fig, axes = plt.subplots(2, n, figsize=(1.2 * n, 2.7))
    for k in range(n):
        axes[0, k].imshow(samples[k, 0].cpu(), cmap="gray", vmin=-1, vmax=1)
        axes[1, k].imshow(train_ds[idx[int(i_s[k])]][0][0], cmap="gray", vmin=-1, vmax=1)
        axes[1, k].set_title(f"{d_s[k]:.3f}", fontsize=7)
        for a in axes[:, k]:
            a.axis("off")
    fig.suptitle(f"samples (top) and nearest of {len(idx):,} training images (bottom, RMS L2); "
                 f"val images' nearest-train RMS L2: {d_v.mean():.3f} +- {d_v.std():.3f}", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_figs / "nearest_neighbours.png", dpi=150)
    plt.close(fig)
    return dict(nn_rms_samples=d_s.tolist(), nn_rms_samples_mean=float(d_s.mean()), nn_rms_val_mean=float(d_v.mean()),
                nn_rms_val_std=float(d_v.std()), nn_rms_val_min=float(d_v.min()), nn_train_subset=len(idx))


# --------------------------------------------------------------------------
# figures + LaTeX fragments
# --------------------------------------------------------------------------

@torch.no_grad()
def val_outliers(cfg, device, out_data, out_figs, which="best", thresh=10.0):
    """Which validation slices blow up under a checkpoint, and how: per-image bpd on the
    training run's validation subset; for every slice above `thresh` bpd the image, its
    bpd, the max |z| after each block and the max |z| after the FIRST block per token
    (where along the raster the blow-up starts). Writes data/val_outliers_<which>.json
    and figures/val_outliers_<which>.png. Does not touch the training artefacts."""
    set_tf32(False)
    _, val_loader, _, _, _ = make_loaders(cfg, device)
    model, path = load_prior(cfg, device, which)
    rows, images, idx0 = [], [], 0
    devs = [device.index] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devs):
        torch.manual_seed(cfg["seed"])
        for x, _ in val_loader:
            x = s2.apply_noise(x.to(device), {"noise_type": "uniform"})
            z, outputs, logdets = model(x)
            n_dims = float(np.prod(x.shape[1:]))
            b = (-s2.gaussian_log_prob(z) / n_dims - logdets) / math.log(2)
            for i in torch.nonzero(b > thresh).flatten().tolist():
                first_tok = int(outputs[0][i].abs().amax(-1).argmax())
                rows.append(dict(index=idx0 + i, bpd=float(b[i]), max_abs_z_per_block=[float(o[i].abs().max()) for o in outputs],
                                 first_block_worst_token=first_tok,
                                 first_block_worst_token_rc=[first_tok // (cfg["img_size"] // cfg["patch_size"]),
                                                             first_tok % (cfg["img_size"] // cfg["patch_size"])],
                                 pixel_min=float(x[i].min()), pixel_max=float(x[i].max())))
                images.append(x[i].cpu())
            idx0 += x.size(0)
    out = dict(checkpoint=str(path.relative_to(REPO_ROOT)), which=which, threshold_bpd=thresh, n_val=idx0,
               n_outliers=len(rows), outliers=rows)
    with open(out_data / f"val_outliers_{which}.json", "w") as f:
        json.dump(out, f, indent=2)
    if images:
        n = len(images)
        fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 3.9), squeeze=False)
        for ax, im, r in zip(axes[0], images, rows):
            ax.imshow(im[0], cmap="gray", vmin=-1, vmax=1)
            rc = r["first_block_worst_token_rc"]
            ps = cfg["patch_size"]
            ax.add_patch(plt.Rectangle((rc[1] * ps - 0.5, rc[0] * ps - 0.5), ps, ps, fill=False, color="r", lw=1))
            mz = r["max_abs_z_per_block"]
            ax.set_title(f"val slice {r['index']}: {r['bpd']:.3g} bpd\nmax|z| after each block: "
                         + " ".join(f"{v:.0e}" for v in mz[:4]) + "\n" + " ".join(f"{v:.0e}" for v in mz[4:]), fontsize=7)
            ax.axis("off")
        fig.suptitle(f"LIDC {cfg['res']} px, {which} checkpoint: validation slices above {thresh:g} bpd ({n} of {idx0})\n"
                     f"red square: the token with the largest |z| after block 1", fontsize=8)
        fig.tight_layout(rect=(0, 0, 1, 0.9))
        fig.savefig(out_figs / f"val_outliers_{which}.png", dpi=130)
        plt.close(fig)
    print(f"[step13:{cfg['res']}] val outliers ({which}): {len(rows)} of {idx0} above {thresh:g} bpd: "
          + "; ".join(f"#{r['index']} {r['bpd']:.3g} bpd, max|z| " + " ".join(f"{v:.0e}" for v in r["max_abs_z_per_block"])
                      for r in rows), flush=True)
    return out


def val_sentence(res):
    """The checkpoint-selection statistic and the validation blow-ups, for the summary."""
    curve = res.get("val_bpd_curve", [])
    # (epoch, mean) entries predate the per-image validation; a mean above 10 bpd is a blow-up
    failed = lambda v: (v[3] > 0) if len(v) > 3 else (v[1] > 10)
    n_fail_epochs = sum(1 for v in curve if failed(v))
    stat = ("mean over the validation images below 10 bpd" if res.get("val_stat") == "robust"
            else "official mean")
    s = f"Best val bpd {res['best_val_bpd']:.4f} (epoch {res['best_epoch']}; {stat}"
    if n_fail_epochs:
        worst = max((v[1], v[0]) for v in curve if failed(v))
        s += (f"; on {n_fail_epochs} of the {len(curve)} evaluated epochs a few validation slices exceeded 10 bpd, "
              f"the plain mean reaching {worst[0]:.3g} at epoch {worst[1]}")
    return s + "); "


def collect(cfg, out_data, out_figs):
    rows = list(csv.DictReader(open(out_data / "metrics.csv"))) if (out_data / "metrics.csv").exists() else []
    res = json.load(open(out_data / "result.json")) if (out_data / "result.json").exists() else None
    if rows:
        ep = [int(r["epoch"]) for r in rows]
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
        axes[0].plot(ep, [float(r["loss"]) for r in rows])
        axes[0].set(xlabel="epoch", ylabel="train loss (nats/dim)", title="training loss")
        bp = [(int(r["epoch"]), float(r["val_bpd"])) for r in rows if r["val_bpd"]]
        rb = [(int(r["epoch"]), float(r["val_bpd_robust"])) for r in rows if r.get("val_bpd_robust")]
        nf = [(int(r["epoch"]), int(r["val_n_fail"])) for r in rows if r.get("val_n_fail")]
        if bp:
            axes[1].plot(*zip(*bp), marker="o", ms=3, label="mean (official)")
            if rb:
                axes[1].plot(*zip(*rb), marker="s", ms=3, label="mean of images < 10 bpd")
                axes[1].legend(fontsize=7)
            for e, n in nf:                      # epochs on which some validation slices blew up
                if n:
                    axes[1].annotate(f"{n}", (e, min(b for _, b in bp) - 0.01), fontsize=6, ha="center", color="C3")
            axes[1].set(xlabel="epoch", ylabel="val bits/dim", title="validation bpd (red: slices above 10 bpd)")
            lo = min(b for _, b in bp + rb)
            axes[1].set_ylim(lo - 0.02, lo + 0.3)
        axes[2].plot(ep, [float(r["grad_norm"]) for r in rows])
        axes[2].set(xlabel="epoch", ylabel="mean grad norm (pre-clip)", title="gradient norm", yscale="log")
        fig.suptitle(f"LIDC {cfg['res']} px", fontsize=9)
        fig.tight_layout()
        fig.savefig(out_figs / "curves.png", dpi=130)
        plt.close(fig)
    lines = []
    if res:
        m, tr = res["config"]["model"], res["config"]["train"]
        minutes = sum(float(r["epoch_seconds"]) + float(r.get("sample_seconds", 0) or 0) for r in rows) / 60 if rows else res["train_minutes"]
        epochs_run = res.get("epochs_run", tr["epochs"])
        per_epoch = (f"{res['images_per_epoch']:,} random training images per epoch"
                     if res.get("images_per_epoch", res["n_train"]) < res["n_train"] else "full epochs")
        schedule = (f"{epochs_run} of a nominal {tr['epochs']}-epoch cosine schedule ({per_epoch}; stopped by the validation "
                    f"rule: no improvement for {tr['early_stop_patience']} epochs)" if res.get("stopped_early")
                    else f"{epochs_run} epochs ({per_epoch})")
        lines.append(
            f"LIDC {cfg['res']}$\\times${cfg['res']} ({384 / cfg['res']:.1f} mm/px), {res['n_train']:,} training slices; TarFlow "
            f"[patch {res['config']['patch_size']}: $T={res['latent']['T']}$ tokens of dimension {res['latent']['D']}; "
            f"{m['channels']} ch, {m['blocks']} blocks, {m['layers_per_block']} layers] = {res['n_params']/1e6:.1f}M parameters, "
            f"batch {tr['batch']}, lr {tr['lr']:g}{', TF32' if res['config'].get('tf32') else ''}{', per-block activation checkpointing' if tr.get('block_checkpoint') else ''}, {schedule}, {minutes:.0f} min. "
            + val_sentence(res) + s9.test_bpd_sentence(res)
            + f"Final samples ({res['config']['train']['sample_nrow'] * res['config']['train']['sample_rows']}, "
            f"{res['final_sample_seconds']:.3g} s with the official sequential reverse): {res['final_samples_nonfinite']} non-finite, "
            f"{100*res['final_samples_frac_outside_range']:.2f}\\% of pixels outside $[-1,1]$. "
            f"Nearest-training-image RMS distance (among {res['nn_train_subset']:,} training slices): samples {res['nn_rms_samples_mean']:.3f} vs. "
            f"val slices {res['nn_rms_val_mean']:.3f} $\\pm$ {res['nn_rms_val_std']:.3f} (min {res['nn_rms_val_min']:.3f}).")
    (out_data / "summary.tex").write_text("\n".join(lines) + "\n")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--val-outliers", action="store_true",
                    help="only probe the validation slices above 10 bpd under the best and final checkpoints")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    out_root = REPO_ROOT / cfg["output_root"]
    out_data, out_figs = out_root / "data", out_root / "figures"
    out_data.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if args.val_outliers:
        for which in ["best", "final"]:
            val_outliers(cfg, device, out_data, out_figs, which)
        return 0
    if not args.collect:
        train(cfg, device, out_data, out_figs)
    collect(cfg, out_data, out_figs)
    prov = dict(step="step13_lidc_tarflow", command=" ".join(["python"] + (argv or sys.argv)), config=cfg,
                tarflow_repo="https://github.com/apple/ml-tarflow", tarflow_commit=s2.tarflow_commit(cfg),
                dataset="TCIA LIDC-IDRI CT slices via step 12", timestamp_utc=datetime.now(timezone.utc).isoformat(),
                versions=dict(python=sys.version.split()[0], torch=torch.__version__, torchvision=tv.__version__))
    with open(out_data / "provenance.json", "w") as f:
        json.dump(prov, f, indent=2)
    print(f"step13 ({cfg['res']}) done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""step9_chestmnist_tarflow.py -- Unconditional TarFlow prior on ChestMNIST 64x64.

Trains the OFFICIAL TarFlow model (apple/ml-tarflow, imported unmodified) as a
likelihood model on 64 x 64 chest radiographs (MedMNIST ChestMNIST, size-64
release), following the official ImageNet64 likelihood recipe: uniform
dequantization, fp32, AdamW(0.9, 0.95, wd 1e-4), cosine schedule with a
one-epoch warm-up, official bits/dim. The width and depth are the paper's
64-resolution setting (768 channels, 8 blocks, 8 layers); the patch size is 8
instead of 2 because that is what one GPU affords (see the config header).

This model is the Bayesian PRIOR used by step 10. Its only job here is to be a
good density p_theta(x) with a stable sampler, and both are measured:

  * val bpd every epoch, test bpd at the end (final and best-val checkpoint);
  * sample grids from fixed noise throughout training;
  * a memorisation check: nearest training image (L2) of each final sample,
    compared with the nearest-training-image distance of held-out val images.

Outputs (outputs/step9_chestmnist_tarflow/):
  data/     metrics.csv, model.pth (final), model_best.pth (best val bpd),
            ckpt.pth (resume), result.json, provenance.json, summary.tex
  figures/  samples_epochNNN.png, real_grid.png, curves.png,
            nearest_neighbours.png, final_samples.png

Usage:
  python step9_chestmnist_tarflow.py                 # train (auto-resumes)
  python step9_chestmnist_tarflow.py --collect       # figures/fragments only
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
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "step9_chestmnist_tarflow.yml"

import invlib  # noqa: E402
import step1_datasets  # noqa: E402
import step2_tarflow as s2  # noqa: E402


def load_config(path=DEFAULT_CONFIG):
    with open(path) as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def build_bundle(cfg):
    """ChestMNIST at the configured size through the step-1 registry."""
    cfg1 = step1_datasets.load_config()
    name = cfg["dataset"]
    flag = name[len("medmnist_"):]
    root = Path(cfg1["data_root"]) / "medmnist"
    return step1_datasets.build_medmnist_one(flag, cfg["medmnist_size"], name, cfg1, root)


def make_loaders(cfg, device):
    bundle = build_bundle(cfg)
    tr = cfg["train"]
    kw = dict(batch_size=tr["batch"], num_workers=4, pin_memory=device.type == "cuda")
    g = torch.Generator().manual_seed(cfg["seed"])

    def subset(ds, n):  # 0 = whole split (the default); subsets are for smoke tests
        if n and n < len(ds):
            return Subset(ds, torch.randperm(len(ds), generator=g)[:n].tolist())
        return ds

    train = DataLoader(subset(s2.TarFlowInput(bundle.train, hflip=cfg.get("hflip", False)),
                              tr.get("train_subset", 0)),
                       shuffle=True, drop_last=True, generator=g, **kw)
    val = DataLoader(subset(s2.TarFlowInput(bundle.val), tr.get("val_subset", 0)), shuffle=False, **kw)
    test = DataLoader(subset(s2.TarFlowInput(bundle.test), tr.get("test_subset", 0)), shuffle=False, **kw)
    return train, val, test, bundle


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------

def build_model(cfg, device):
    tf = s2.import_tarflow(cfg)
    m = cfg["model"]
    model = tf.Model(in_channels=cfg["channel_size"], img_size=cfg["img_size"],
                     patch_size=cfg["patch_size"], channels=m["channels"],
                     num_blocks=m["blocks"], layers_per_block=m["layers_per_block"],
                     nvp=True, num_classes=0)
    if cfg.get("scale_bound", 0):
        invlib.bound_log_scale(model, cfg["scale_bound"])
    return model.to(device)


def load_prior(cfg, device, which="best"):
    """The trained prior for later steps: model + the checkpoint's metadata."""
    out_data = REPO_ROOT / cfg["output_root"] / "data"
    model = build_model(cfg, device)
    path = out_data / ("model_best.pth" if which == "best" else "model.pth")
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model, path


def log_prob(model, x):
    """log p(x) in nats per image (official base + logdet accounting).

    The official forward returns the per-example MEAN log-scale over tokens
    and dims, so it is rescaled by the dimension count.
    """
    z, _, logdet_mean = model(x)
    n_dims = float(np.prod(x.shape[1:]))
    log_pz = -0.5 * (z ** 2 + math.log(2 * math.pi)).flatten(1).sum(-1)
    return log_pz + logdet_mean * n_dims


@torch.no_grad()
def per_image_bpd(model, loader, device, seed=0):
    """Official bits/dim (uniform dequantization, k = 128) of every image in the
    loader, with a fixed noise seed. Also returns, for the worst image, the max
    |z| after each block (the signature of an autoregressive blow-up)."""
    model.eval()
    bpds, worst = [], (-1.0, None, None)
    devs = [device.index] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devs):
        torch.manual_seed(seed)
        for x, _ in loader:
            x = s2.apply_noise(x.to(device), {"noise_type": "uniform"})
            z, outputs, logdets = model(x)
            n_dims = float(np.prod(x.shape[1:]))
            b = (-s2.gaussian_log_prob(z) / n_dims - logdets) / math.log(2)
            i = int(b.argmax())
            if b[i].item() > worst[0]:
                worst = (b[i].item(), x[i].detach().cpu(),
                         [float(o[i].abs().max()) for o in outputs])
            bpds.append(b.detach().cpu())
    model.train()
    return torch.cat(bpds).numpy(), worst


def bpd_stats(bpds, outlier_bpd=10.0):
    """Mean (the official number), plus statistics that are robust to the rare
    image on which an autoregressive flow blows up (bpd >> outlier_bpd)."""
    bad = bpds > outlier_bpd
    return dict(mean=float(bpds.mean()), median=float(np.median(bpds)), max=float(bpds.max()),
                argmax=int(bpds.argmax()), n_over_thresh=int(bad.sum()), outlier_bpd=outlier_bpd,
                mean_excl_outliers=float(bpds[~bad].mean()) if (~bad).any() else float("nan"))


def evaluate_bpd(model, loader, device, seed=0):
    """Mean official bits/dim over the loader."""
    return float(per_image_bpd(model, loader, device, seed)[0].mean())


@torch.no_grad()
def sample(model, noise, batch=64):
    model.eval()
    out = [model.reverse(noise[i:i + batch]) for i in range(0, noise.size(0), batch)]
    model.train()
    return torch.cat(out)


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------

def train(cfg, device, out_data, out_figs):
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    tr = cfg["train"]
    train_loader, val_loader, test_loader, bundle = make_loaders(cfg, device)
    model = build_model(cfg, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[step9] {n_params/1e6:.1f}M params; train {len(train_loader.dataset)} "
          f"val {len(val_loader.dataset)} test {len(test_loader.dataset)}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.95), lr=tr["lr"],
                                  weight_decay=tr["weight_decay"])
    steps_per_epoch = len(train_loader)
    total_steps, warmup = tr["epochs"] * steps_per_epoch, steps_per_epoch

    def lr_at(step):  # official utils.CosineLRSchedule shape
        min_lr, max_lr = 1e-6, tr["lr"]
        if step <= warmup:
            return min_lr + step / warmup * (max_lr - min_lr)
        t = (step - warmup) / max(1, total_steps - warmup)
        return min_lr + 0.5 * (1 + math.cos(math.pi * t)) * (max_lr - min_lr)

    n_show = tr["sample_nrow"] * tr["sample_rows"]
    fixed_noise = torch.randn(n_show, (cfg["img_size"] // cfg["patch_size"]) ** 2,
                              cfg["channel_size"] * cfg["patch_size"] ** 2,
                              generator=torch.Generator().manual_seed(cfg["seed"] + 1)).to(device)

    # a grid of real training images for side-by-side comparison
    real = torch.stack([train_loader.dataset[i][0] for i in range(n_show)])
    tv.utils.save_image((real + 1) / 2, out_figs / "real_grid.png", nrow=tr["sample_nrow"])

    metrics_path, ckpt_path = out_data / "metrics.csv", out_data / "ckpt.pth"
    start_epoch, step, val_bpds, best = 0, 0, [], (float("inf"), 0)
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch, step, val_bpds = ckpt["epoch"], ckpt["step"], ckpt["val_bpds"]
        best = tuple(ckpt["best"])
        print(f"[step9] resumed at epoch {start_epoch} (best val bpd {best[0]:.4f} @ {best[1]})", flush=True)
    else:
        with open(metrics_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "loss", "mse_z", "logdet", "lr", "grad_norm",
                                    "val_bpd", "epoch_seconds", "n_nonfinite"])

    # early stopping on the validation bpd. The lr schedule is NOT re-planned: it
    # stays the cosine decay over the nominal `epochs`, so a stopped run is a
    # truncation of the nominal one, not a shorter schedule.
    def early_stop(epochs_done):
        patience = tr.get("early_stop_patience")
        return bool(patience) and bool(val_bpds) and epochs_done - best[1] >= patience

    t_start = time.time()
    stopped_early = early_stop(start_epoch)
    if stopped_early:
        print(f"[step9] resumed run already meets the early-stop rule (best {best[0]:.4f} "
              f"@ {best[1]}, {start_epoch} epochs done): skipping training", flush=True)
    for epoch in range(start_epoch, tr["epochs"]):
        if stopped_early:
            break
        t0 = time.time()
        sums = {"loss": 0.0, "mse": 0.0, "ld": 0.0, "gn": 0.0}
        nb, n_nonfinite = 0, 0
        for x, _ in train_loader:
            x = s2.apply_noise(x.to(device, non_blocking=True), cfg)
            step += 1
            for pg in optimizer.param_groups:
                pg["lr"] = lr_at(step)
            optimizer.zero_grad(set_to_none=True)
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

        val_bpd = ""
        if (epoch + 1) % tr["eval_every"] == 0 or epoch + 1 == tr["epochs"]:
            val_bpd = evaluate_bpd(model, val_loader, device, seed=cfg["seed"])
            val_bpds.append((epoch + 1, val_bpd))
            if val_bpd < best[0]:
                best = (val_bpd, epoch + 1)
                torch.save(model.state_dict(), out_data / "model_best.pth")
        dt = time.time() - t0
        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, sums["loss"] / max(nb, 1), sums["mse"] / max(nb, 1),
                                    sums["ld"] / max(nb, 1), lr_at(step), sums["gn"] / max(nb, 1),
                                    val_bpd, round(dt, 1), n_nonfinite])
        msg = (f"[step9] epoch {epoch+1}/{tr['epochs']} loss {sums['loss']/max(nb,1):.4f} "
               f"gn {sums['gn']/max(nb,1):.2f} ({dt:.0f}s)")
        if val_bpd != "":
            msg += f" val_bpd {val_bpd:.4f} (best {best[0]:.4f} @ {best[1]})"
        if n_nonfinite:
            msg += f" NONFINITE {n_nonfinite}"
        print(msg, flush=True)

        stopped_early = early_stop(epoch + 1)
        if stopped_early:
            print(f"[step9] early stop at epoch {epoch+1}: no val improvement for "
                  f"{tr['early_stop_patience']} epochs (best {best[0]:.4f} @ {best[1]})", flush=True)

        if (epoch + 1) % tr["sample_every"] == 0 or epoch + 1 == tr["epochs"] or stopped_early:
            s = sample(model, fixed_noise)
            n_bad = int((~torch.isfinite(s)).flatten(1).any(1).sum())
            tv.utils.save_image((s.clamp(-1, 1).nan_to_num() + 1) / 2,
                                out_figs / f"samples_epoch{epoch+1:03d}.png", nrow=tr["sample_nrow"])
            if n_bad:
                print(f"[step9]   {n_bad}/{n_show} non-finite samples at epoch {epoch+1}", flush=True)
        torch.save(model.state_dict(), out_data / "model.pth")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "epoch": epoch + 1, "step": step, "val_bpds": val_bpds, "best": list(best)},
                   ckpt_path)
    train_minutes = (time.time() - t_start) / 60

    # ---- final evaluation ----------------------------------------------
    epochs_run = val_bpds[-1][0] if val_bpds else tr["epochs"]
    result = dict(n_params=n_params, config=cfg, train_minutes=round(train_minutes, 1),
                  epochs_run=epochs_run, stopped_early=bool(stopped_early),
                  val_bpd_curve=val_bpds, best_val_bpd=best[0], best_epoch=best[1],
                  n_train=len(train_loader.dataset), n_val=len(val_loader.dataset),
                  n_test=len(test_loader.dataset))
    # Per-image test bpd for the final and the best-val checkpoint. The mean is
    # the official number, but an autoregressive flow can blow up on a single
    # image (|z| growing by up to exp(scale_bound) per block), and one such image
    # makes the mean meaningless, so the median, the count of such images and
    # the mean without them are recorded too, along with the worst image.
    for tag in ["final", "best"]:
        if tag == "best":
            model.load_state_dict(torch.load(out_data / "model_best.pth", map_location=device))
        bpds, worst = per_image_bpd(model, test_loader, device, seed=cfg["seed"])
        st = bpd_stats(bpds)
        result[f"test_bpd_{tag}"] = st["mean"]
        result[f"test_bpd_{tag}_stats"] = dict(st, worst_max_abs_z_per_block=worst[2])
        np.save(out_data / f"test_bpd_per_image_{tag}.npy", bpds)
        tv.utils.save_image((worst[1] + 1) / 2, out_figs / f"test_worst_image_{tag}.png")
        print(f"[step9] TEST bpd ({tag}{'' if tag == 'final' else f', best-val epoch {best[1]}'}): "
              f"mean {st['mean']:.4g}, median {st['median']:.4f}, max {st['max']:.4g} (image {st['argmax']}), "
              f"{st['n_over_thresh']} images > {st['outlier_bpd']} bpd, mean without them "
              f"{st['mean_excl_outliers']:.4f}; worst image max|z| per block "
              + " ".join(f"{v:.1e}" for v in worst[2]), flush=True)
    for split, loader in [("val", val_loader)]:
        bpds, _ = per_image_bpd(model, loader, device, seed=cfg["seed"])
        result[f"{split}_bpd_best_stats"] = bpd_stats(bpds)

    # final samples from the best-val model + memorisation check
    torch.manual_seed(cfg["seed"] + 2)
    noise = torch.randn_like(fixed_noise)
    s = sample(model, noise)
    result["final_samples_nonfinite"] = int((~torch.isfinite(s)).flatten(1).any(1).sum())
    result["final_samples_frac_outside_range"] = float(((s.abs() > 1).float().mean()))
    s = s.clamp(-1, 1).nan_to_num()
    tv.utils.save_image((s + 1) / 2, out_figs / "final_samples.png", nrow=tr["sample_nrow"])
    result.update(nearest_neighbour_check(s[:cfg["nn_check_samples"]], bundle, val_loader, device, out_figs))

    with open(out_data / "result.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


@torch.no_grad()
def nearest_neighbour_check(samples, bundle, val_loader, device, out_figs):
    """L2 nearest training image of each sample vs. the same for val images."""
    train_ds = s2.TarFlowInput(bundle.train)
    loader = DataLoader(train_ds, batch_size=512, num_workers=0)   # workers would exhaust the container's /dev/shm
    train_x = torch.cat([x for x, _ in loader]).to(device).flatten(1)     # (N, D) in [-1,1]

    def nn_dist(q):
        q = q.to(device).flatten(1)
        best_d, best_i = torch.full((q.size(0),), float("inf"), device=device), torch.zeros(q.size(0), dtype=torch.long, device=device)
        for i in range(0, train_x.size(0), 8192):
            d = torch.cdist(q, train_x[i:i + 8192])
            m, j = d.min(1)
            upd = m < best_d
            best_d, best_i = torch.where(upd, m, best_d), torch.where(upd, j + i, best_i)
        return best_d / math.sqrt(q.size(1)), best_i                     # RMS distance

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
        axes[1, k].imshow(train_x[i_s[k]].view(samples.shape[1:])[0].cpu(), cmap="gray", vmin=-1, vmax=1)
        axes[1, k].set_title(f"{d_s[k]:.3f}", fontsize=7)
        for a in axes[:, k]:
            a.axis("off")
    axes[0, 0].set_ylabel("sample")
    fig.suptitle(f"samples (top) and nearest training image (bottom, RMS L2); "
                 f"val images' nearest-train RMS L2: {d_v.mean():.3f} +- {d_v.std():.3f}", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_figs / "nearest_neighbours.png", dpi=150)
    plt.close(fig)
    return dict(nn_rms_samples=d_s.tolist(), nn_rms_samples_mean=float(d_s.mean()),
                nn_rms_val_mean=float(d_v.mean()), nn_rms_val_std=float(d_v.std()),
                nn_rms_val_min=float(d_v.min()))


# --------------------------------------------------------------------------
# figures + LaTeX fragments
# --------------------------------------------------------------------------

def _tex(s):
    return (str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")
            .replace("#", r"\#"))


def sci(v):
    """LaTeX math for a large number: 3.6\\times 10^{24}."""
    e = int(math.floor(math.log10(abs(v)))) if v else 0
    return f"{v / 10 ** e:.1f}\\times 10^{{{e}}}" if abs(e) >= 3 else f"{v:.3g}"


def test_bpd_sentence(res):
    """Test bpd for the summary: the plain mean if no image blew up, otherwise
    the mean without the blown-up images, stated as such, plus the raw mean."""
    sb, sf = res.get("test_bpd_best_stats"), res.get("test_bpd_final_stats")
    if sb is None or sf is None:      # result.json from before the per-image stats existed
        return (f"test bpd {res['test_bpd_best']:.4f} (best-val checkpoint) / "
                f"{res['test_bpd_final']:.4f} (final). ")
    if sb["n_over_thresh"] == 0 and sf["n_over_thresh"] == 0:
        return (f"test bpd {sb['mean']:.4f} (best-val checkpoint; median {sb['median']:.4f}) / "
                f"{sf['mean']:.4f} (final). ")
    return (f"test bpd {sb['mean_excl_outliers']:.4f} (best-val checkpoint; median {sb['median']:.4f}) / "
            f"{sf['mean_excl_outliers']:.4f} (final) over the {res['n_test'] - sb['n_over_thresh']:,} test "
            f"images below {sb['outlier_bpd']:.0f} bits/dim; the plain test mean is "
            f"${sci(sb['mean'])}$ / ${sci(sf['mean'])}$ because {sb['n_over_thresh']} test image"
            f"{'s' if sb['n_over_thresh'] != 1 else ''} (index {sb['argmax']}) makes the flow blow up "
            f"(bpd ${sci(sb['max'])}$; max $|z|$ after each block: "
            + ", ".join(f"${sci(v)}$" for v in sb["worst_max_abs_z_per_block"]) + "). ")


def collect(cfg, out_data, out_figs):
    rows = list(csv.DictReader(open(out_data / "metrics.csv"))) if (out_data / "metrics.csv").exists() else []
    res = json.load(open(out_data / "result.json")) if (out_data / "result.json").exists() else None
    if rows:
        ep = [int(r["epoch"]) for r in rows]
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
        axes[0].plot(ep, [float(r["loss"]) for r in rows])
        axes[0].set(xlabel="epoch", ylabel="train loss (nats/dim)", title="training loss")
        bp = [(int(r["epoch"]), float(r["val_bpd"])) for r in rows if r["val_bpd"]]
        if bp:
            axes[1].plot(*zip(*bp), marker="o", ms=3)
            axes[1].set(xlabel="epoch", ylabel="val bits/dim", title="validation bpd")
            lo = min(b for _, b in bp)
            axes[1].set_ylim(lo - 0.02, lo + 0.3)
        axes[2].plot(ep, [float(r["grad_norm"]) for r in rows])
        axes[2].set(xlabel="epoch", ylabel="mean grad norm (pre-clip)", title="gradient norm", yscale="log")
        fig.tight_layout()
        fig.savefig(out_figs / "curves.png", dpi=130)
        plt.close(fig)

    lines = []
    if res:
        m, tr = res["config"]["model"], res["config"]["train"]
        minutes = sum(float(r["epoch_seconds"]) for r in rows) / 60 if rows else res["train_minutes"]
        epochs_run = res.get("epochs_run", tr["epochs"])
        schedule = (f"{epochs_run} of a nominal {tr['epochs']}-epoch cosine schedule (stopped by the "
                    f"validation rule: no improvement for {tr['early_stop_patience']} epochs)"
                    if res.get("stopped_early") else f"{epochs_run} epochs")
        lines.append(
            f"ChestMNIST 64$\\times$64, {res['n_train']:,} training images; TarFlow "
            f"[patch {res['config']['patch_size']}, {m['channels']} ch, {m['blocks']} blocks, "
            f"{m['layers_per_block']} layers] = {res['n_params']/1e6:.1f}M parameters, "
            f"{schedule}, {minutes:.0f} min. "
            f"Best val bpd {res['best_val_bpd']:.4f} (epoch {res['best_epoch']}); "
            + test_bpd_sentence(res) +
            f"Final samples: {res['final_samples_nonfinite']}/64 non-finite, "
            f"{100*res['final_samples_frac_outside_range']:.2f}\\% of pixels outside $[-1,1]$. "
            f"Nearest-training-image RMS distance: samples {res['nn_rms_samples_mean']:.3f} vs. "
            f"val images {res['nn_rms_val_mean']:.3f} $\\pm$ {res['nn_rms_val_std']:.3f} "
            f"(min {res['nn_rms_val_min']:.3f}).")
    (out_data / "summary.tex").write_text("\n".join(lines) + "\n")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--collect", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    out_root = REPO_ROOT / cfg["output_root"]
    out_data, out_figs = out_root / "data", out_root / "figures"
    out_data.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if not args.collect:
        train(cfg, device, out_data, out_figs)
    collect(cfg, out_data, out_figs)
    prov = dict(step="step9_chestmnist_tarflow", command=" ".join(["python"] + (argv or sys.argv)),
                config=cfg, tarflow_repo="https://github.com/apple/ml-tarflow",
                tarflow_commit=s2.tarflow_commit(cfg),
                dataset="MedMNIST v2 ChestMNIST, size-64 release (Yang et al. 2023)",
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                versions=dict(python=sys.version.split()[0], torch=torch.__version__,
                              torchvision=tv.__version__))
    with open(out_data / "provenance.json", "w") as f:
        json.dump(prov, f, indent=2)
    print("step9 done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

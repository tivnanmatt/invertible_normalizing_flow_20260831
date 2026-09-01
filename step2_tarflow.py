#!/usr/bin/env python
"""step2_tarflow.py -- Train and test TarFlow (Transformer Autoregressive Flow).

TarFlow is the architecture from "Normalizing Flows are Capable Generative
Models" (Zhai et al., Apple, arXiv:2412.06329). We use the OFFICIAL model code
unmodified, imported from a clone of github.com/apple/ml-tarflow (path set in
the config; the clone lives outside this repo and its commit hash is recorded
in provenance.json). The training loop, noise schemes, optimizer settings and
the bits/dim computation faithfully follow the official train.py /
evaluate_bpd.py / train_local.ipynb.

Runs are defined in configs/step2_tarflow.yml. The default config has two:

  mnist_toy   exact replication of Apple's released MNIST toy experiment
              (train_local.ipynb): class-conditional, 28x28, [4-128-4-4],
              gaussian noise 0.1, bs 256, lr 2e-3, 100 epochs. Reproduces
              their conditional sample grid and Bayes-rule classifier eval.
  cifar10_uniform
              unconditional CIFAR-10 likelihood run in the paper's
              likelihood configuration style (uniform dequantization, fp32,
              patch 2), scaled to a single RTX 4090. Reports val bpd during
              training and final test bpd with the official formula.

Datasets/splits come from step1_datasets.build_dataset (seeded step-1 splits).

Outputs (outputs/step2_tarflow/):
  data/     per-run metrics.csv, result JSON, summary + LaTeX fragments
  figures/  loss and bpd curves, sample grids

Usage:
  python step2_tarflow.py                          # all runs, sequentially
  python step2_tarflow.py --runs mnist_toy         # a single run
  python step2_tarflow.py --collect                # rebuild summary/fragments
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
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "step2_tarflow.yml"

import step1_datasets  # noqa: E402  (dataset registry of step 1)


def load_config(path=DEFAULT_CONFIG):
    with open(path) as f:
        return yaml.safe_load(f)


def import_tarflow(cfg):
    """Import the official transformer_flow module from the ml-tarflow clone."""
    p = str(Path(cfg["tarflow_repo"]).resolve())
    if p not in sys.path:
        sys.path.insert(0, p)
    import transformer_flow  # noqa: F401

    return transformer_flow


def tarflow_commit(cfg):
    head = Path(cfg["tarflow_repo"]) / ".git"
    try:
        ref = (head / "HEAD").read_text().strip()
        if ref.startswith("ref:"):
            return (head / ref.split(" ", 1)[1]).read_text().strip()
        return ref
    except OSError:
        return "unknown"


class TarFlowInput(Dataset):
    """Adapts a step-1 dataset ([0,1] tensors) to TarFlow's [-1,1] inputs,
    with optional random horizontal flip (train-time, as in official train.py)."""

    def __init__(self, base, hflip=False):
        self.base = base
        self.hflip = hflip

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        item = self.base[i]
        x, y = item if isinstance(item, (list, tuple)) else (item, 0)
        x = x * 2 - 1
        if self.hflip and torch.rand(()) < 0.5:
            x = x.flip(-1)
        if isinstance(y, torch.Tensor):  # medmnist-style (1,) labels
            y = int(y.reshape(-1)[0])
        return x, y


def apply_noise(x, rc):
    """Official noise schemes; x in [-1,1]. Returns noised x."""
    if rc["noise_type"] == "gaussian":
        return x + rc["noise_std"] * torch.randn_like(x)
    if rc["noise_type"] == "uniform":  # uniform dequantization of 8-bit data
        x_int = (x + 1) * (255 / 2)
        x = (x_int + torch.rand_like(x_int)) / 256
        return x * 2 - 1
    raise ValueError(rc["noise_type"])


def gaussian_log_prob(z, k=128):
    """Official evaluate_bpd.py prior term (k=128 accounts for [-1,1] scaling
    of 8-bit bins)."""
    log_p = -0.5 * (z**2 + np.log(2 * np.pi))
    return log_p.flatten(1).sum(-1) - np.log(k) * np.prod(z.size()[1:])


@torch.no_grad()
def evaluate_bpd(model, loader, device, conditional):
    """Test/val bits-per-dim with uniform dequantization (official formula,
    generalized to C channels). Per-example logdets recomputed from z chain is
    not needed: model returns per-example mean logdet."""
    model.eval()
    total, count = 0.0, 0
    for x, y in loader:
        x = x.to(device)
        x_int = (x + 1) * (255 / 2)
        x = (x_int + torch.rand_like(x_int)) / 256
        x = x * 2 - 1
        y = y.to(device) if conditional else None
        z, _, logdets = model(x, y)
        n_dims = float(np.prod(x.shape[1:]))
        nll = -gaussian_log_prob(z) / n_dims - logdets
        total += (nll / math.log(2)).sum().item()
        count += x.size(0)
    model.train()
    return total / count


@torch.no_grad()
def evaluate_bayes_accuracy(model, loader, device, num_classes, noise_std):
    """Bayes-rule classifier eval from the official train_local.ipynb:
    argmin_y  0.5 E[z_y^2] - logdet_y  (uniform class prior)."""
    model.eval()
    num_correct, num_examples = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x = x + noise_std * torch.randn_like(x)
        xr = x.repeat(num_classes, 1, 1, 1)
        y_ = torch.arange(num_classes, device=device).view(-1, 1).repeat(1, y.size(0)).flatten()
        z, _, logdets = model(xr, y_)
        losses = 0.5 * z.pow(2).mean(dim=[1, 2]) - logdets
        pred = losses.reshape(num_classes, y.size(0)).argmin(dim=0)
        num_correct += (pred == y).sum().item()
        num_examples += y.size(0)
    model.train()
    return num_correct / num_examples


@torch.no_grad()
def save_samples(model, rc, fixed_noise, fixed_y, path, device, use_amp):
    model.eval()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
        samples = model.reverse(fixed_noise.to(device),
                                None if fixed_y is None else fixed_y.to(device),
                                guidance=rc.get("cfg", 0))
    tv.utils.save_image(samples.float(), path, normalize=True, nrow=rc["sample_nrow"])
    model.train()


def make_loaders(rc, seed):
    cfg1 = step1_datasets.load_config()
    bundle = step1_datasets.build_dataset(rc["dataset"], cfg1)
    conditional = rc.get("conditional", False)
    kw = dict(batch_size=rc["batch_size"], num_workers=4, pin_memory=True)
    g = torch.Generator()
    g.manual_seed(seed)
    train = DataLoader(TarFlowInput(bundle.train, hflip=rc.get("hflip", False)),
                       shuffle=True, drop_last=True, generator=g, **kw)
    val = DataLoader(TarFlowInput(bundle.val), shuffle=False, **kw)
    test = DataLoader(TarFlowInput(bundle.test), shuffle=False, **kw)
    n_classes = bundle.meta.get("n_classes") if conditional else 0
    return train, val, test, (n_classes or 0)


def run_one(name, rc, cfg, out_data, out_figs, device):
    transformer_flow = import_tarflow(cfg)
    seed = cfg["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader, val_loader, test_loader, num_classes = make_loaders(rc, seed)
    model = transformer_flow.Model(
        in_channels=rc["channel_size"],
        img_size=rc["img_size"],
        patch_size=rc["patch_size"],
        channels=rc["channels"],
        num_blocks=rc["blocks"],
        layers_per_block=rc["layers_per_block"],
        nvp=True,
        num_classes=num_classes,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{name}] {n_params/1e6:.1f}M params, {num_classes} classes, device {device}")

    # official recipe: AdamW(0.9, 0.95, wd=1e-4), cosine LR with 1-epoch warmup
    optimizer = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.95),
                                  lr=rc["lr"], weight_decay=1e-4)
    steps_per_epoch = len(train_loader)
    total_steps = rc["epochs"] * steps_per_epoch
    warmup = steps_per_epoch

    def lr_at(step):  # same shape as official utils.CosineLRSchedule
        min_lr, max_lr = 1e-6, rc["lr"]
        if step <= warmup:
            return min_lr + step / warmup * (max_lr - min_lr)
        t = (step - warmup) / (total_steps - warmup)
        return min_lr + 0.5 * (1 + math.cos(math.pi * t)) * (max_lr - min_lr)

    # official train.py: amp only for gaussian noise; uniform (likelihood) runs fp32
    use_amp = rc["noise_type"] == "gaussian"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    n_show = rc["sample_nrow"] * rc["sample_rows"]
    fixed_noise = torch.randn(n_show, (rc["img_size"] // rc["patch_size"]) ** 2,
                              rc["channel_size"] * rc["patch_size"] ** 2)
    if num_classes:
        fixed_y = (torch.arange(rc["sample_rows"]) % num_classes).view(-1, 1)
        fixed_y = fixed_y.repeat(1, rc["sample_nrow"]).flatten()
    else:
        fixed_y = None

    metrics_path = out_data / f"{name}_metrics.csv"
    ckpt_path = out_data / f"{name}_ckpt.pth"

    # resume from a full checkpoint if one exists (crash protection)
    start_epoch, step, val_bpds = 0, 0, []
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch, step = ckpt["epoch"], ckpt["step"]
        val_bpds = ckpt.get("val_bpds", [])
        print(f"[{name}] resumed from checkpoint at epoch {start_epoch}")
    if start_epoch == 0:
        with open(metrics_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "loss", "mse_z", "logdet", "lr",
                                    "val_bpd", "epoch_seconds"])

    t_start = time.time()
    for epoch in range(start_epoch, rc["epochs"]):
        t0 = time.time()
        sums = {"loss": 0.0, "mse": 0.0, "ld": 0.0}
        nb = 0
        for x, y in train_loader:
            x = apply_noise(x.to(device, non_blocking=True), rc)
            y = y.to(device) if num_classes else None
            step += 1
            for pg in optimizer.param_groups:
                pg["lr"] = lr_at(step)
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                z, outputs, logdets = model(x, y)
                loss = model.get_loss(z, logdets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            sums["loss"] += loss.item()
            sums["mse"] += 0.5 * (z.detach() ** 2).mean().item()
            sums["ld"] += logdets.detach().mean().item()
            nb += 1

        val_bpd = ""
        if rc.get("eval_bpd") and ((epoch + 1) % rc["eval_freq"] == 0 or epoch + 1 == rc["epochs"]):
            val_bpd = evaluate_bpd(model, val_loader, device, num_classes > 0)
            val_bpds.append((epoch + 1, val_bpd))
        dt = time.time() - t0
        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, sums["loss"] / nb, sums["mse"] / nb,
                                    sums["ld"] / nb, lr_at(step),
                                    val_bpd, round(dt, 1)])
        msg = f"[{name}] epoch {epoch+1}/{rc['epochs']} loss {sums['loss']/nb:.4f} ({dt:.0f}s)"
        if val_bpd != "":
            msg += f" val_bpd {val_bpd:.4f}"
        print(msg, flush=True)

        if (epoch + 1) % rc["sample_freq"] == 0 or epoch + 1 == rc["epochs"]:
            save_samples(model, rc, fixed_noise, fixed_y,
                         out_figs / f"{name}_samples_epoch{epoch+1:03d}.png", device, use_amp)
        torch.save(model.state_dict(), out_data / f"{name}_model.pth")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "epoch": epoch + 1, "step": step, "val_bpds": val_bpds}, ckpt_path)

    result = {
        "name": name, "config": rc, "n_params": n_params,
        "num_classes": num_classes,
        "train_minutes": round((time.time() - t_start) / 60, 1),
        "final_train_loss": sums["loss"] / nb,
    }
    if rc.get("eval_bpd"):
        result["val_bpd_curve"] = val_bpds
        result["test_bpd"] = evaluate_bpd(model, test_loader, device, num_classes > 0)
        print(f"[{name}] TEST BPD: {result['test_bpd']:.4f}")
    if rc.get("classifier_eval") and num_classes:
        result["test_bayes_accuracy"] = evaluate_bayes_accuracy(
            model, test_loader, device, num_classes, rc["noise_std"])
        print(f"[{name}] Bayes classifier accuracy: {100*result['test_bayes_accuracy']:.2f}%")

    with open(out_data / f"{name}_result.json", "w") as f:
        json.dump(result, f, indent=2)
    plot_curves(name, metrics_path, out_figs)
    return result


def plot_curves(name, metrics_path, out_figs):
    rows = list(csv.DictReader(open(metrics_path)))
    ep = [int(r["epoch"]) for r in rows]
    loss = [float(r["loss"]) for r in rows]
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.plot(ep, loss)
    ax.set_xlabel("epoch")
    ax.set_ylabel("training loss (0.5 E[z^2] - E[logdet])")
    ax.set_title(f"{name}: training loss")
    fig.tight_layout()
    fig.savefig(out_figs / f"{name}_loss.png", dpi=120)
    plt.close(fig)

    bpd_pts = [(int(r["epoch"]), float(r["val_bpd"])) for r in rows if r["val_bpd"]]
    if bpd_pts:
        fig, ax = plt.subplots(figsize=(5.5, 3.4))
        ax.plot(*zip(*bpd_pts), marker="o")
        ax.set_xlabel("epoch")
        ax.set_ylabel("validation bits/dim")
        ax.set_title(f"{name}: val bpd (uniform dequantization)")
        fig.tight_layout()
        fig.savefig(out_figs / f"{name}_bpd.png", dpi=120)
        plt.close(fig)


# --------------------------------------------------------------------------
# summary + LaTeX fragments (consumed by paper/main.tex)
# --------------------------------------------------------------------------

def _tex(s):
    s = (str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")
         .replace("#", r"\#").replace("{", r"\{").replace("}", r"\}"))
    return s.replace("<", r"\textless{}").replace(">", r"\textgreater{}")


def collect(cfg, out_data, out_figs):
    results = []
    for name in cfg["runs"]:
        p = out_data / f"{name}_result.json"
        if p.exists():
            results.append(json.load(open(p)))
    lines = [
        r"\begin{tabular}{llllrrl}",
        r"\hline",
        r"run & dataset & model [p-c-b-l] & noise & params & epochs & result \\",
        r"\hline",
    ]
    for r in results:
        rc = r["config"]
        model_s = f"{rc['patch_size']}-{rc['channels']}-{rc['blocks']}-{rc['layers_per_block']}"
        noise_s = rc["noise_type"] + (f"({rc['noise_std']})" if rc["noise_type"] == "gaussian" else "")
        outs = []
        if "test_bpd" in r:
            outs.append(f"test bpd {r['test_bpd']:.3f}")
        if "test_bayes_accuracy" in r:
            outs.append(f"Bayes acc {100*r['test_bayes_accuracy']:.2f}\\%")
        if "status" in r:  # partial/interrupted runs
            outs.append(_tex(r["status"].split(":")[0]))
            if r.get("val_bpd_curve"):
                ep, bpd = r["val_bpd_curve"][-1]
                outs.append(f"val bpd {bpd:.3f} @ep{ep}")
        lines.append(
            f"{_tex(r['name'])} & {_tex(rc['dataset'])} & {model_s} & {_tex(noise_s)} & "
            f"{r['n_params']/1e6:.1f}M & {rc['epochs']} & {', '.join(outs) or '--'} \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    (out_data / "summary_table_step2.tex").write_text("\n".join(lines) + "\n")

    figs = []
    for r in results:
        name = r["name"]
        for suffix, width in [("loss", 0.55), ("bpd", 0.55)]:
            f = out_figs / f"{name}_{suffix}.png"
            if f.exists():
                figs += [
                    r"\begin{figure}[htbp]", r"\centering",
                    rf"\includegraphics[width={width}\textwidth]{{outputs/step2_tarflow/figures/{f.name}}}",
                    rf"\caption{{\textbf{{{_tex(name)}}} ({suffix}). "
                    rf"File: \texttt{{{_tex('outputs/step2_tarflow/figures/' + f.name)}}}, "
                    rf"produced by \texttt{{step2\_tarflow.py}}.}}",
                    r"\end{figure}", "",
                ]
        sample_figs = sorted(out_figs.glob(f"{name}_samples_epoch*.png"))
        for f in [sample_figs[0], sample_figs[-1]] if len(sample_figs) > 1 else sample_figs:
            figs += [
                r"\begin{figure}[htbp]", r"\centering",
                rf"\includegraphics[width=0.7\textwidth]{{outputs/step2_tarflow/figures/{f.name}}}",
                rf"\caption{{\textbf{{{_tex(name)}}}: samples at {_tex(f.stem.split('epoch')[-1])} epochs "
                rf"(reverse pass from Gaussian noise, no denoising step). "
                rf"File: \texttt{{{_tex('outputs/step2_tarflow/figures/' + f.name)}}}.}}",
                r"\end{figure}", "",
            ]
        figs.append(r"\clearpage")
    (out_data / "figures_step2.tex").write_text("\n".join(figs) + "\n")
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--runs", nargs="*", default=None,
                    help="subset of runs (default: all in config)")
    ap.add_argument("--device", default=None, help="override run device (GPU0-only policy: cuda:0)")
    ap.add_argument("--collect", action="store_true",
                    help="only rebuild summary/latex fragments from existing results")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    out_root = REPO_ROOT / cfg.get("output_root", "outputs/step2_tarflow")
    out_data = out_root / "data"
    out_figs = out_root / "figures"
    out_data.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)

    if not args.collect:
        names = args.runs or list(cfg["runs"])
        for name in names:
            rc = cfg["runs"][name]
            device = args.device or rc.get("device", "cuda:0")
            run_one(name, rc, cfg, out_data, out_figs, device)

    results = collect(cfg, out_data, out_figs)
    prov = dict(
        step="step2_tarflow",
        command=" ".join(["python"] + (argv if argv is not None else sys.argv)),
        config=cfg,
        tarflow_repo="https://github.com/apple/ml-tarflow",
        tarflow_commit=tarflow_commit(cfg),
        paper="Zhai et al., Normalizing Flows are Capable Generative Models, arXiv:2412.06329",
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        versions=dict(python=sys.version.split()[0], torch=torch.__version__,
                      torchvision=tv.__version__),
        results=[r["name"] for r in results],
    )
    with open(out_data / "provenance.json", "w") as f:
        json.dump(prov, f, indent=2)
    print(f"step2 done: {len(results)} run results in {out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

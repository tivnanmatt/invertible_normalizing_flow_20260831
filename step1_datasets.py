#!/usr/bin/env python
"""step1_datasets.py -- Datasets and dataloaders for the invertible normalizing flow project.

Builds PyTorch datasets and dataloaders, with deterministic train/val/test
splits, for the standard generative-image-modeling benchmarks:

  auto-download : MNIST, FashionMNIST, CIFAR-10, CIFAR-100, SVHN,
                  the MedMNIST v2/v3 2D collection, Imagenette (ImageNet-10
                  subset), AFHQ (animal faces, StarGAN-v2 release), CelebA
  manual        : ImageNet (ILSVRC2012), CelebA-HQ, FFHQ, LSUN bedroom

Running this file is also the step-1 test: for every enabled dataset it
instantiates the three splits, pulls batches through real DataLoaders, and
writes to outputs/step1_datasets/
  data/    per-dataset stats JSON, summary.csv, provenance.json, and
           auto-generated LaTeX fragments (summary_table.tex, figures_step1.tex)
           that paper/main.tex includes verbatim
  figures/ sample grids, pixel histograms, class distributions

Config: configs/step1_datasets.yml

Usage:
  python step1_datasets.py                       # run every enabled dataset
  python step1_datasets.py --datasets mnist afhq # smoke-test a subset

Library use:
  from step1_datasets import load_config, build_dataset, get_dataloaders
  bundle  = build_dataset("cifar10", load_config())
  loaders = get_dataloaders(bundle, load_config())   # {'train':..,'val':..,'test':..}

Split conventions (recorded per-dataset in meta["split_note"]):
  * official train/test only  -> val carved from train (split.val_fraction, seeded)
  * official train/val/test   -> used as-is (MedMNIST, CelebA)
  * official train/val only   -> official val becomes TEST, val carved from train
                                 (Imagenette, ImageNet, AFHQ)
  * single undivided pool     -> seeded random train/val/test
                                 (FFHQ, CelebA-HQ)
"""

import argparse
import json
import os
import sys
import traceback
import urllib.request
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import yaml
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "step1_datasets.yml"


class DatasetUnavailable(Exception):
    """Raised when a dataset cannot be built; message says how to obtain it."""


@dataclass
class DatasetBundle:
    name: str
    train: Dataset
    val: Dataset
    test: Dataset
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# config / small helpers
# --------------------------------------------------------------------------

def load_config(path=DEFAULT_CONFIG):
    with open(path) as f:
        return yaml.safe_load(f)


def _seeded_generator(seed):
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g


def split_train_val(train_ds, val_fraction, seed):
    """Deterministically carve a validation set out of a training set."""
    n = len(train_ds)
    n_val = max(1, int(round(n * val_fraction)))
    n_train = n - n_val
    return random_split(train_ds, [n_train, n_val], generator=_seeded_generator(seed))


def split_three_way(ds, val_fraction, test_fraction, seed):
    """Deterministic train/val/test split of a single undivided pool."""
    n = len(ds)
    n_val = max(1, int(round(n * val_fraction)))
    n_test = max(1, int(round(n * test_fraction)))
    n_train = n - n_val - n_test
    return random_split(ds, [n_train, n_val, n_test], generator=_seeded_generator(seed))


def _image_transform(image_size=None):
    ops = []
    if image_size:
        ops += [transforms.Resize(image_size), transforms.CenterCrop(image_size)]
    ops.append(transforms.ToTensor())  # [0,1] float — dequantize downstream in the flow
    return transforms.Compose(ops)


def _download(url, dest, timeout=60):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    tmp.rename(dest)


class FlatImageDataset(Dataset):
    """Recursively collects images under a directory; returns (image, 0)."""

    EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.paths = sorted(
            p for p in self.root.rglob("*") if p.suffix.lower() in self.EXTS
        )
        self.transform = transform
        if not self.paths:
            raise DatasetUnavailable(f"no images found under {root}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from PIL import Image

        img = Image.open(self.paths[i]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, 0


# --------------------------------------------------------------------------
# dataset builders (registry at bottom)
# --------------------------------------------------------------------------

def _std_torchvision(cls_name, name, cfg, root, gray=False, svhn=False):
    """MNIST-style datasets: official train/test, val carved from train."""
    cls = getattr(torchvision.datasets, cls_name)
    tfm = _image_transform()
    kw = dict(root=str(root), download=True, transform=tfm)
    if svhn:
        full_train = cls(split="train", **kw)
        test = cls(split="test", **kw)
    else:
        full_train = cls(train=True, **kw)
        test = cls(train=False, **kw)
    seed = cfg["seed"]
    vf = cfg["split"]["val_fraction"]
    train, val = split_train_val(full_train, vf, seed)
    n_classes = len(getattr(full_train, "classes", [])) or (10 if svhn else None)
    return DatasetBundle(
        name, train, val, test,
        meta=dict(
            source=f"torchvision.datasets.{cls_name} (auto-download)",
            n_classes=n_classes,
            split_note=f"official test; val = {vf:.0%} of official train (seed {seed})",
            label_source=full_train,
        ),
    )


def build_mnist(name, dcfg, cfg, root):
    return _std_torchvision("MNIST", name, cfg, root / "mnist", gray=True)


def build_fashion_mnist(name, dcfg, cfg, root):
    return _std_torchvision("FashionMNIST", name, cfg, root / "fashion_mnist", gray=True)


def build_cifar10(name, dcfg, cfg, root):
    return _std_torchvision("CIFAR10", name, cfg, root / "cifar10")


def build_cifar100(name, dcfg, cfg, root):
    return _std_torchvision("CIFAR100", name, cfg, root / "cifar100")


def build_svhn(name, dcfg, cfg, root):
    return _std_torchvision("SVHN", name, cfg, root / "svhn", svhn=True)


def build_medmnist_one(flag, size, name, cfg, root):
    import medmnist
    from medmnist import INFO

    info = INFO[flag]
    cls = getattr(medmnist, info["python_class"])
    tfm = _image_transform()
    root.mkdir(parents=True, exist_ok=True)  # medmnist requires an existing root
    kw = dict(root=str(root), download=True, transform=tfm, size=size)
    train = cls(split="train", **kw)
    val = cls(split="val", **kw)
    test = cls(split="test", **kw)
    return DatasetBundle(
        name, train, val, test,
        meta=dict(
            source=f"medmnist.{info['python_class']} v{medmnist.__version__} (auto-download)",
            n_classes=len(info["label"]),
            task=info["task"],
            split_note="official MedMNIST train/val/test",
            label_source=train,
        ),
    )


def build_celeba(name, dcfg, cfg, root):
    tfm = _image_transform(dcfg.get("image_size", 64))
    try:
        kw = dict(root=str(root / "celeba"), target_type="attr", transform=tfm, download=True)
        train = torchvision.datasets.CelebA(split="train", **kw)
        val = torchvision.datasets.CelebA(split="valid", **kw)
        test = torchvision.datasets.CelebA(split="test", **kw)
    except Exception as e:
        raise DatasetUnavailable(
            "CelebA auto-download failed (Google Drive quota is a known issue: "
            f"{type(e).__name__}). Manually place img_align_celeba/ plus the "
            "list_*.txt / identity_* files under data/celeba/celeba/ "
            "(see torchvision.datasets.CelebA docs), then re-run."
        )
    return DatasetBundle(
        name, train, val, test,
        meta=dict(
            source="torchvision.datasets.CelebA (aligned 178x218, auto-download)",
            n_classes=None,
            split_note="official CelebA train/valid/test partition; 40 binary "
                       "attributes per image (multi-label, no class histogram)",
        ),
    )


def build_imagenette(name, dcfg, cfg, root):
    variant = dcfg.get("variant", "160px")
    tfm = _image_transform(dcfg.get("image_size", 64))
    r = root / "imagenette"
    r.mkdir(parents=True, exist_ok=True)
    try:
        full_train = torchvision.datasets.Imagenette(
            str(r), split="train", size=variant, download=not (r / f"imagenette2-160").exists(),
            transform=tfm)
    except RuntimeError:
        full_train = torchvision.datasets.Imagenette(str(r), split="train", size=variant, transform=tfm)
    test = torchvision.datasets.Imagenette(str(r), split="val", size=variant, transform=tfm)
    seed, vf = cfg["seed"], cfg["split"]["val_fraction"]
    train, val = split_train_val(full_train, vf, seed)
    return DatasetBundle(
        name, train, val, test,
        meta=dict(
            source=f"torchvision.datasets.Imagenette {variant} (fast.ai ImageNet-10 subset, auto-download)",
            n_classes=10,
            split_note=f"official val used as TEST; val = {vf:.0%} of official train (seed {seed})",
            label_source=full_train,
        ),
    )


def build_imagenet(name, dcfg, cfg, root):
    r = root / "imagenet"
    if not ((r / "train").is_dir() and (r / "val").is_dir()):
        raise DatasetUnavailable(
            "ImageNet (ILSVRC2012) requires manual download (terms of access): "
            "obtain from https://image-net.org, extract to data/imagenet/train/<wnid>/*.JPEG "
            "and data/imagenet/val/<wnid>/*.JPEG, then re-run."
        )
    tfm = _image_transform(dcfg.get("image_size", 64))
    full_train = torchvision.datasets.ImageFolder(str(r / "train"), transform=tfm)
    test = torchvision.datasets.ImageFolder(str(r / "val"), transform=tfm)
    seed, vf = cfg["seed"], cfg["split"]["val_fraction"]
    train, val = split_train_val(full_train, vf, seed)
    return DatasetBundle(
        name, train, val, test,
        meta=dict(
            source="ILSVRC2012 via ImageFolder (manual download)",
            n_classes=len(full_train.classes),
            split_note=f"official val used as TEST; val = {vf:.0%} of official train (seed {seed})",
            label_source=full_train,
        ),
    )


AFHQ_URLS = {
    "v1": "https://www.dropbox.com/s/t9l9o3vsx2jai3z/afhq.zip?dl=1",
    "v2": "https://www.dropbox.com/s/vkzjokiwof5h8w6/afhq_v2.zip?dl=1",
}


def build_afhq(name, dcfg, cfg, root):
    version = dcfg.get("version", "v1")
    r = root / "afhq"

    def find_train_dir(base):
        if (base / "train").is_dir():
            return base
        for p in base.rglob("train"):
            if p.is_dir():
                return p.parent
        return None

    base = find_train_dir(r) if r.exists() else None
    if base is None and dcfg.get("attempt_download", True):
        url = AFHQ_URLS[version]
        zpath = r / f"afhq_{version}.zip"
        try:
            print(f"    downloading AFHQ {version} from dropbox ...")
            _download(url, zpath, timeout=120)
            with zipfile.ZipFile(zpath) as z:
                z.extractall(r)
            base = find_train_dir(r)
        except Exception as e:
            raise DatasetUnavailable(
                f"AFHQ auto-download failed ({type(e).__name__}: {e}). Download the "
                "StarGAN-v2 AFHQ release (github.com/clovaai/stargan-v2) and extract "
                "to data/afhq/train/{cat,dog,wild} and data/afhq/val/{cat,dog,wild}."
            )
    if base is None:
        raise DatasetUnavailable(
            "AFHQ not found: extract to data/afhq/train/{cat,dog,wild} and "
            "data/afhq/val/{cat,dog,wild} (StarGAN-v2 release)."
        )
    tfm = _image_transform(dcfg.get("image_size", 64))
    full_train = torchvision.datasets.ImageFolder(str(base / "train"), transform=tfm)
    val_dir = base / "val" if (base / "val").is_dir() else base / "test"
    test = torchvision.datasets.ImageFolder(str(val_dir), transform=tfm)
    seed, vf = cfg["seed"], cfg["split"]["val_fraction"]
    train, val = split_train_val(full_train, vf, seed)
    return DatasetBundle(
        name, train, val, test,
        meta=dict(
            source=f"AFHQ {version} (StarGAN-v2 release, cat/dog/wild, auto-download from dropbox)",
            n_classes=len(full_train.classes),
            split_note=f"official val used as TEST; val = {vf:.0%} of official train (seed {seed})",
            label_source=full_train,
        ),
    )


def _flat_pool(name, dirname, human, cfg, dcfg, root, hint):
    d = root / dirname
    if not d.is_dir():
        raise DatasetUnavailable(f"{human} requires manual download: {hint}")
    tfm = _image_transform(dcfg.get("image_size", 128))
    pool = FlatImageDataset(d, transform=tfm)
    seed = cfg["seed"]
    vf, tf = cfg["split"]["val_fraction"], cfg["split"]["test_fraction"]
    train, val, test = split_three_way(pool, vf, tf, seed)
    return DatasetBundle(
        name, train, val, test,
        meta=dict(
            source=f"{human} via FlatImageDataset (manual download)",
            n_classes=None,
            split_note=f"single pool split {1 - vf - tf:.0%}/{vf:.0%}/{tf:.0%} (seed {seed})",
        ),
    )


def build_ffhq(name, dcfg, cfg, root):
    return _flat_pool(
        name, "ffhq", "FFHQ", cfg, dcfg, root,
        "place images (e.g. thumbnails128x128 or images1024x1024 from "
        "github.com/NVlabs/ffhq-dataset) under data/ffhq/",
    )


def build_celeba_hq(name, dcfg, cfg, root):
    return _flat_pool(
        name, "celeba_hq", "CelebA-HQ", cfg, dcfg, root,
        "place the 30k CelebA-HQ images (e.g. celeba_hq_256 from the "
        "progressive-GAN release) under data/celeba_hq/",
    )


def build_lsun_bedroom(name, dcfg, cfg, root):
    r = root / "lsun"
    if not (r / "bedroom_train_lmdb").is_dir():
        raise DatasetUnavailable(
            "LSUN bedroom requires manual download: fetch bedroom_train_lmdb / "
            "bedroom_val_lmdb with github.com/fyu/lsun download.py into data/lsun/ "
            "(also needs `pip install lmdb`)."
        )
    tfm = _image_transform(dcfg.get("image_size", 64))
    full_train = torchvision.datasets.LSUN(str(r), classes=["bedroom_train"], transform=tfm)
    test = torchvision.datasets.LSUN(str(r), classes=["bedroom_val"], transform=tfm)
    seed, vf = cfg["seed"], cfg["split"]["val_fraction"]
    train, val = split_train_val(full_train, vf, seed)
    return DatasetBundle(
        name, train, val, test,
        meta=dict(
            source="torchvision.datasets.LSUN lmdb (manual download)",
            n_classes=1,
            split_note=f"official val used as TEST; val = {vf:.0%} of official train (seed {seed})",
        ),
    )


BUILDERS = {
    "mnist": build_mnist,
    "fashion_mnist": build_fashion_mnist,
    "cifar10": build_cifar10,
    "cifar100": build_cifar100,
    "svhn": build_svhn,
    "celeba": build_celeba,
    "imagenette": build_imagenette,
    "imagenet": build_imagenet,
    "afhq": build_afhq,
    "ffhq": build_ffhq,
    "celeba_hq": build_celeba_hq,
    "lsun_bedroom": build_lsun_bedroom,
}


def expand_dataset_list(cfg):
    """Enabled dataset names from config; medmnist expands to one per flag."""
    names = []
    for key, dcfg in cfg["datasets"].items():
        if not dcfg.get("enabled", False):
            continue
        if key == "medmnist":
            names += [f"medmnist_{flag}" for flag in dcfg.get("flags", [])]
        else:
            names.append(key)
    return names


def build_dataset(name, cfg):
    """Build one DatasetBundle by name (medmnist_<flag> for MedMNIST members)."""
    root = REPO_ROOT / cfg.get("data_root", "data")
    root.mkdir(parents=True, exist_ok=True)
    if name.startswith("medmnist_"):
        flag = name[len("medmnist_"):]
        mcfg = cfg["datasets"]["medmnist"]
        return build_medmnist_one(flag, mcfg.get("size", 28), name, cfg, root / "medmnist")
    if name not in BUILDERS:
        raise KeyError(f"unknown dataset {name!r}; known: {sorted(BUILDERS)}")
    return BUILDERS[name](name, cfg["datasets"].get(name, {}), cfg, root)


def get_dataloaders(bundle, cfg, shuffle_train=True):
    """DataLoaders for the three splits of a DatasetBundle."""
    lc = cfg["loader"]
    kw = dict(
        batch_size=lc["batch_size"],
        num_workers=lc["num_workers"],
        pin_memory=lc.get("pin_memory", True),
        drop_last=False,
    )
    return {
        "train": DataLoader(bundle.train, shuffle=shuffle_train, **kw),
        "val": DataLoader(bundle.val, shuffle=False, **kw),
        "test": DataLoader(bundle.test, shuffle=False, **kw),
    }


# --------------------------------------------------------------------------
# step-1 test harness: stats + figures per dataset
# --------------------------------------------------------------------------

def extract_labels(ds):
    """Integer class labels of a dataset if cheaply recoverable, else None."""
    if isinstance(ds, Subset):
        base = extract_labels(ds.dataset)
        return None if base is None else base[list(ds.indices)]
    for attr in ("targets", "labels"):
        if hasattr(ds, attr):
            arr = np.asarray(getattr(ds, attr))
            if arr.ndim == 2 and arr.shape[1] == 1:
                arr = arr[:, 0]
            if arr.ndim == 1 and np.issubdtype(arr.dtype, np.number):
                return arr.astype(int)
            return None
    if hasattr(ds, "samples"):  # ImageFolder
        return np.asarray([s[1] for s in ds.samples], dtype=int)
    if hasattr(ds, "_samples"):  # Imagenette
        return np.asarray([s[1] for s in ds._samples], dtype=int)
    return None


def _to_grid_image(x):
    """(C,H,W) [0,1] tensor -> HxW or HxWx3 numpy for imshow."""
    x = x.clamp(0, 1).numpy()
    return x[0] if x.shape[0] == 1 else np.transpose(x, (1, 2, 0))


def fig_samples(bundle, loaders, n_per_split, path):
    fig, axes = plt.subplots(3, n_per_split, figsize=(1.6 * n_per_split, 5.2))
    for row, split in enumerate(["train", "val", "test"]):
        batch = next(iter(loaders[split]))
        imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
        for col in range(n_per_split):
            ax = axes[row, col]
            if col < imgs.shape[0]:
                im = _to_grid_image(imgs[col])
                ax.imshow(im, cmap="gray" if im.ndim == 2 else None)
            ax.set_xticks([]), ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(split, fontsize=11)
    fig.suptitle(f"{bundle.name}: samples per split", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fig_pixel_hist(bundle, batch, path):
    imgs = (batch[0] if isinstance(batch, (list, tuple)) else batch).numpy()
    C = imgs.shape[1]
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    colors = ["red", "green", "blue"] if C == 3 else ["black"]
    labels = ["R", "G", "B"] if C == 3 else ["intensity"]
    for c in range(C):
        ax.hist(imgs[:, c].ravel(), bins=64, range=(0, 1), histtype="step",
                density=True, color=colors[c % len(colors)], label=labels[c % len(labels)])
    ax.set_xlabel("pixel value (ToTensor scale [0,1])")
    ax.set_ylabel("density")
    ax.set_title(f"{bundle.name}: pixel intensity (one train batch)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fig_class_dist(bundle, labels, path):
    vals, counts = np.unique(labels, return_counts=True)
    if len(vals) > 120:
        return False
    fig, ax = plt.subplots(figsize=(max(5.5, 0.09 * len(vals) + 3), 3.4))
    ax.bar(vals, counts, width=0.8)
    ax.set_xlabel("class index")
    ax.set_ylabel("count (official train pool)")
    ax.set_title(f"{bundle.name}: class distribution ({len(vals)} classes)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def channel_stats(loader, max_batches):
    """Per-channel mean/std over up to max_batches batches."""
    s = s2 = n = None
    count = 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        x = (batch[0] if isinstance(batch, (list, tuple)) else batch).double()
        dims = (0, 2, 3)
        if s is None:
            s = x.sum(dims)
            s2 = (x ** 2).sum(dims)
            n = x.shape[0] * x.shape[2] * x.shape[3]
        else:
            s += x.sum(dims)
            s2 += (x ** 2).sum(dims)
            n += x.shape[0] * x.shape[2] * x.shape[3]
        count += 1
    mean = s / n
    std = (s2 / n - mean ** 2).clamp(min=0).sqrt()
    return mean.tolist(), std.tolist(), count


def run_one(name, cfg, out_data, out_figs):
    """Build, loader-test, and document one dataset. Returns its record dict."""
    rec = {"name": name, "status": "ok", "figures": []}
    try:
        bundle = build_dataset(name, cfg)
    except DatasetUnavailable as e:
        rec.update(status="unavailable", detail=str(e))
        print(f"    UNAVAILABLE: {e}")
        return rec
    loaders = get_dataloaders(bundle, cfg)

    batch = next(iter(loaders["train"]))
    x = batch[0] if isinstance(batch, (list, tuple)) else batch
    rec["image_shape"] = list(x.shape[1:])
    rec["batch_shape"] = list(x.shape)
    rec["splits"] = {k: len(getattr(bundle, k)) for k in ("train", "val", "test")}
    rec["n_classes"] = bundle.meta.get("n_classes")
    rec["source"] = bundle.meta.get("source")
    rec["split_note"] = bundle.meta.get("split_note")

    mean, std, nb = channel_stats(loaders["train"], cfg["figures"]["stats_max_batches"])
    rec["train_channel_mean"] = [round(v, 4) for v in mean]
    rec["train_channel_std"] = [round(v, 4) for v in std]
    rec["stats_batches"] = nb

    n_show = cfg["figures"]["samples_per_split"]
    f = out_figs / f"{name}_samples.png"
    fig_samples(bundle, loaders, n_show, f)
    rec["figures"].append(f.name)
    f = out_figs / f"{name}_pixel_hist.png"
    fig_pixel_hist(bundle, batch, f)
    rec["figures"].append(f.name)

    label_src = bundle.meta.get("label_source")
    labels = extract_labels(label_src) if label_src is not None else None
    if labels is not None:
        f = out_figs / f"{name}_class_dist.png"
        if fig_class_dist(bundle, labels, f):
            rec["figures"].append(f.name)

    with open(out_data / f"{name}.json", "w") as fp:
        json.dump(rec, fp, indent=2)
    print(f"    ok: splits {rec['splits']}, image {rec['image_shape']}, "
          f"{len(rec['figures'])} figures")
    return rec


# --------------------------------------------------------------------------
# LaTeX fragment generation (consumed by paper/main.tex)
# --------------------------------------------------------------------------

def _tex(s):
    s = (str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")
         .replace("#", r"\#").replace("{", r"\{").replace("}", r"\}"))
    return s.replace("<", r"\textless{}").replace(">", r"\textgreater{}")


def write_latex_fragments(records, out_data):
    lines = [
        r"\begin{tabular}{llrrrll}",
        r"\hline",
        r"dataset & status & train & val & test & image & classes \\",
        r"\hline",
    ]
    for r in records:
        if r["status"] == "ok":
            sp = r["splits"]
            shape = r"$" + r"\times".join(str(v) for v in r["image_shape"]) + "$"
            ncls = r["n_classes"] if r["n_classes"] is not None else "--"
            lines.append(
                f"{_tex(r['name'])} & ok & {sp['train']} & {sp['val']} & "
                f"{sp['test']} & {shape} & {ncls} \\\\"
            )
        else:
            lines.append(f"{_tex(r['name'])} & unavailable & -- & -- & -- & -- & -- \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    (out_data / "summary_table.tex").write_text("\n".join(lines) + "\n")

    figs = []
    for r in records:
        if r["status"] != "ok":
            continue
        for fname in r["figures"]:
            kind = fname.rsplit("_", 1)[-1].replace(".png", "")
            width = 0.95 if fname.endswith("_samples.png") else 0.62
            figs += [
                r"\begin{figure}[htbp]",
                r"\centering",
                rf"\includegraphics[width={width}\textwidth]{{outputs/step1_datasets/figures/{fname}}}",
                rf"\caption{{\textbf{{{_tex(r['name'])}}} ({kind}). Source: {_tex(r['source'])}. "
                rf"Split: {_tex(r['split_note'])}. "
                rf"File: \texttt{{outputs/step1\_datasets/figures/{_tex(fname)}}}, "
                rf"produced by \texttt{{step1\_datasets.py}}.}}",
                r"\end{figure}",
                "",
            ]
        figs.append(r"\clearpage" if len(r["figures"]) >= 2 else "")
    (out_data / "figures_step1.tex").write_text("\n".join(figs) + "\n")

    unav = [r for r in records if r["status"] != "ok"]
    lines = [r"\begin{itemize}"]
    for r in unav:
        lines.append(rf"\item \textbf{{{_tex(r['name'])}}}: {_tex(r['detail'])}")
    lines += [r"\end{itemize}"] if unav else []
    (out_data / "unavailable_step1.tex").write_text(
        "\n".join(lines) + "\n" if unav else "All configured datasets were available.\n")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="subset of dataset names to run (default: all enabled in config)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    # some dataset mirrors (notably fashion-mnist s3) stall silently and the
    # stdlib downloader has no timeout; turn hangs into per-dataset errors
    import socket
    socket.setdefaulttimeout(120)

    out_root = REPO_ROOT / cfg.get("output_root", "outputs/step1_datasets")
    out_data = out_root / "data"
    out_figs = out_root / "figures"
    out_data.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)

    names = args.datasets or expand_dataset_list(cfg)
    print(f"step1_datasets: {len(names)} datasets: {', '.join(names)}")

    records = []
    for name in names:
        print(f"  [{len(records) + 1}/{len(names)}] {name}")
        try:
            records.append(run_one(name, cfg, out_data, out_figs))
        except Exception as e:
            traceback.print_exc()
            records.append({"name": name, "status": "error",
                            "detail": f"{type(e).__name__}: {e}", "figures": []})

    with open(out_data / "summary.json", "w") as fp:
        json.dump(records, fp, indent=2)
    import csv
    with open(out_data / "summary.csv", "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["name", "status", "train", "val", "test", "image_shape", "n_classes"])
        for r in records:
            sp = r.get("splits", {})
            w.writerow([r["name"], r["status"], sp.get("train"), sp.get("val"),
                        sp.get("test"), "x".join(map(str, r.get("image_shape", []))),
                        r.get("n_classes")])
    write_latex_fragments(records, out_data)

    prov = dict(
        step="step1_datasets",
        command=" ".join(["python"] + (argv if argv is not None else sys.argv)),
        config_file=str(Path(args.config).resolve()),
        config=cfg,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        versions=dict(
            python=sys.version.split()[0],
            torch=torch.__version__,
            torchvision=torchvision.__version__,
            numpy=np.__version__,
        ),
        cuda_available=torch.cuda.is_available(),
    )
    try:
        import medmnist
        prov["versions"]["medmnist"] = medmnist.__version__
    except ImportError:
        pass
    with open(out_data / "provenance.json", "w") as fp:
        json.dump(prov, fp, indent=2)

    n_ok = sum(r["status"] == "ok" for r in records)
    print(f"done: {n_ok}/{len(records)} datasets ok; outputs in {out_root}")
    return 0 if all(r["status"] != "error" for r in records) else 1


if __name__ == "__main__":
    sys.exit(main())

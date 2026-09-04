#!/usr/bin/env python
"""step12_lidc_dataset.py -- LIDC-IDRI axial CT slices on one common physical grid.

The ChestMNIST radiographs of steps 9-11 are replaced by real CT slices. Every
CT series of the TCIA LIDC-IDRI collection (1,018 series, 1,010 subjects,
243,958 DICOM images at /data/LIDC) is read from its DICOM headers:

  * axial images only (ImageOrientationPatient = identity), sorted by the z of
    ImagePositionPatient, duplicate z positions dropped;
  * slices are SUBSAMPLED to a minimum z spacing (selection.z_stride_mm), so a
    0.6 mm thin-slice scan and a 2.5 mm scan contribute at about the same rate;
  * pixel values -> Hounsfield units with RescaleSlope/Intercept, clipped;
  * every slice is resampled from the scanner grid (Rows x Columns pixels of
    PixelSpacing mm, the reconstruction diameter differs per scan) onto ONE
    physical window of grid.fov_mm x grid.fov_mm centred on the body (centroid
    of HU > -500, one in-plane shift per series), at grid.base_res pixels, with
    Gaussian anti-aliasing + bilinear interpolation; outside the scan is air;
  * the lower resolutions are 2 x 2 average pools of the master grid, so 256,
    128, 64 and 32 px images all show the same physical field of view;
  * the model data are 8-bit: the window intensity.window_hu maps to 0..255
    (uniform dequantization then makes bits/dim a real data likelihood, as for
    ChestMNIST); the master 256 px cache also keeps the HU as uint16;
  * the split is BY SUBJECT (seeded 80/10/10), never by slice.

Outputs (outputs/step12_lidc_dataset/):
  data/     series.csv (one row per CT series, incl. rejected ones), stats.json,
            build.json, verify.json, example_resample.npz, summary.tex, provenance.json
  figures/  examples_256.png, examples_lungwindow.png, resolutions.png,
            resample_check.png, stats.png
Caches (cache_root, default /data/image_benchmarks/lidc): manifest.csv,
  slices_256_hu.u16, slices_{256,128,64,32}.u8 (memmaps), build.json.

Usage:
  python step12_lidc_dataset.py --scan      # headers -> series.csv + slice selection (throw-away container with /data/LIDC)
  python step12_lidc_dataset.py --build     # pixels -> caches (same container)
  python step12_lidc_dataset.py --verify    # resampling self-tests (anywhere)
  python step12_lidc_dataset.py --collect   # figures + fragments from the caches (recon-dev)
Library: build_lidc(res, cfg) -> step1_datasets.DatasetBundle for steps 13+.
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "step12_lidc_dataset.yml"
HEADER_TAGS = ["ImagePositionPatient", "ImageOrientationPatient", "PixelSpacing", "Rows", "Columns", "RescaleSlope",
               "RescaleIntercept", "InstanceNumber", "SOPInstanceUID", "SliceThickness", "ImageType", "Modality",
               "Manufacturer", "ConvolutionKernel", "KVP", "ReconstructionDiameter"]


def load_config(path=DEFAULT_CONFIG):
    with open(path) as f:
        return yaml.safe_load(f)


def _tex(s):
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("#", r"\#")


# --------------------------------------------------------------------------
# 1. headers: series -> selected slices
# --------------------------------------------------------------------------

def list_ct_series(cfg):
    """CT series from TCIA's metadata.csv, in file order."""
    root = Path(cfg["lidc_root"])
    out = []
    with open(root / "metadata.csv") as f:
        for r in csv.DictReader(f):
            if r["Modality"] != "CT":
                continue
            loc = r["File Location"].replace("\\", "/")
            loc = loc[2:] if loc.startswith("./") else loc
            out.append(dict(subject=r["Subject ID"], series_uid=r["Series UID"], path=str(root / loc),
                            n_files_meta=int(r["Number of Images"]), manufacturer=r["Manufacturer"]))
    return out


def _read_headers(job):
    """All DICOM headers of one series directory (worker)."""
    import pydicom
    d = Path(job["path"])
    recs = []
    for f in sorted(d.glob("*.dcm")):
        try:
            h = pydicom.dcmread(str(f), stop_before_pixels=True, specific_tags=HEADER_TAGS)
        except Exception as e:  # unreadable file: recorded, not fatal
            recs.append(dict(file=str(f), error=repr(e)))
            continue
        ipp = getattr(h, "ImagePositionPatient", None)
        iop = getattr(h, "ImageOrientationPatient", None)
        ps = getattr(h, "PixelSpacing", None)
        recs.append(dict(
            file=str(f), modality=str(getattr(h, "Modality", "")), image_type="/".join(getattr(h, "ImageType", [])),
            z=float(ipp[2]) if ipp is not None else None, iop=[float(v) for v in iop] if iop is not None else None,
            spacing=[float(v) for v in ps] if ps is not None else None,
            rows=int(getattr(h, "Rows", 0)), cols=int(getattr(h, "Columns", 0)),
            slope=float(getattr(h, "RescaleSlope", 1.0)), intercept=float(getattr(h, "RescaleIntercept", 0.0)),
            instance=int(getattr(h, "InstanceNumber", 0) or 0), sop=str(getattr(h, "SOPInstanceUID", "")),
            thickness=float(getattr(h, "SliceThickness", 0) or 0), kernel=str(getattr(h, "ConvolutionKernel", "")),
            kvp=float(getattr(h, "KVP", 0) or 0), recon_diameter=float(getattr(h, "ReconstructionDiameter", 0) or 0)))
    return recs


def axial_rotation(iop, tol=1e-3):
    """Number of 90-degree array rotations that bring an axial image to the standard
    (row = +x, column = +y) orientation; None if the image is not axial."""
    for k, ref in ((0, [1, 0, 0, 0, 1, 0]), (2, [-1, 0, 0, 0, -1, 0])):
        if max(abs(a - b) for a, b in zip(iop, ref)) < tol:
            return k
    return None


def select_slices(recs, sel):
    """Axial CT images sorted by z, de-duplicated, subsampled to z_stride_mm. Returns
    (selected records, info dict with counts and the rejection reason if any)."""
    info = dict(n_files=len(recs), n_errors=sum("error" in r for r in recs))
    ok = [r for r in recs if "error" not in r and r["modality"] == "CT" and r["z"] is not None
          and r["spacing"] is not None and r["rows"] > 0]
    if sel.get("require_axial", True):   # identity, or its 180-degree in-plane rotation (14 series; un-rotated when read)
        ok = [r for r in ok if r["iop"] is not None and axial_rotation(r["iop"]) is not None]
    info["n_axial"] = len(ok)
    # one geometry per series: the modal (rows, cols, spacing)
    if ok:
        keys = [(r["rows"], r["cols"], round(r["spacing"][0], 4), round(r["spacing"][1], 4)) for r in ok]
        mode = max(set(keys), key=keys.count)
        ok = [r for r, k in zip(ok, keys) if k == mode]
    ok.sort(key=lambda r: (r["z"], r["instance"]))
    dedup = []
    for r in ok:
        if not dedup or r["z"] - dedup[-1]["z"] > 1e-3:
            dedup.append(r)
    info["n_unique_z"] = len(dedup)
    if len(dedup) < sel["min_slices"]:
        info["reject"] = f"only {len(dedup)} axial slices"
        return [], info
    zs = np.array([r["z"] for r in dedup])
    dz = np.diff(zs)
    info["z_spacing_mm"] = float(np.median(dz))
    info["z_extent_mm"] = float(zs[-1] - zs[0])
    stride = sel["z_stride_mm"]
    chosen, last = [], -np.inf
    for r in dedup:
        if r["z"] >= last + stride - 1e-6:
            chosen.append(r)
            last = r["z"]
    info["n_selected"] = len(chosen)
    return chosen, info


def scan(cfg, out_data):
    series = list_ct_series(cfg)
    sel = cfg["selection"]
    print(f"[step12] {len(series)} CT series from metadata.csv; reading headers with {cfg['workers']} workers", flush=True)
    t0 = time.time()
    rows, slices = [], []
    with Pool(cfg["workers"]) as pool:
        for k, (s, recs) in enumerate(zip(series, pool.imap(_read_headers, series, chunksize=2))):
            chosen, info = select_slices(recs, sel)
            g = chosen[0] if chosen else next((r for r in recs if "error" not in r and r["spacing"]), None)
            row = dict(subject=s["subject"], series_uid=s["series_uid"], path=s["path"], manufacturer=s["manufacturer"],
                       rows=g["rows"] if g else 0, cols=g["cols"] if g else 0,
                       spacing_row_mm=g["spacing"][0] if g else 0, spacing_col_mm=g["spacing"][1] if g else 0,
                       fov_mm=(g["rows"] * g["spacing"][0]) if g else 0, thickness_mm=g["thickness"] if g else 0,
                       kernel=g["kernel"] if g else "", kvp=g["kvp"] if g else 0, slope=g["slope"] if g else 1,
                       intercept=g["intercept"] if g else 0, **{k: v for k, v in info.items()})
            rows.append(row)
            for j, r in enumerate(chosen):
                slices.append(dict(series_index=k, subject=s["subject"], series_uid=s["series_uid"], file=r["file"],
                                   z_mm=r["z"], instance=r["instance"], slice_in_series=j))
            if k % 100 == 0 or k == len(series) - 1:
                print(f"[step12]   {k + 1}/{len(series)} series, {len(slices)} slices selected, {time.time() - t0:.0f} s", flush=True)
    # split by subject
    subjects = sorted({r["subject"] for r in rows if "reject" not in r})
    g = torch.Generator().manual_seed(cfg["seed"])
    perm = [subjects[i] for i in torch.randperm(len(subjects), generator=g).tolist()]
    n_val = int(round(cfg["split"]["val_fraction"] * len(subjects)))
    n_test = int(round(cfg["split"]["test_fraction"] * len(subjects)))
    split_of = {s: "test" for s in perm[:n_test]}
    split_of.update({s: "val" for s in perm[n_test:n_test + n_val]})
    split_of.update({s: "train" for s in perm[n_test + n_val:]})
    for r in rows:
        r["split"] = split_of.get(r["subject"], "")
    for r in slices:
        r["split"] = split_of[r["subject"]]
    _write_csv(out_data / "series.csv", rows)
    _write_csv(out_data / "slices.csv", slices)
    kept = [r for r in rows if "reject" not in r]
    stats = dict(
        n_series=len(rows), n_series_kept=len(kept), n_subjects=len({r["subject"] for r in rows}),
        n_subjects_kept=len(subjects), n_files=sum(r["n_files"] for r in rows), n_axial=sum(r["n_axial"] for r in rows),
        n_unique_z=sum(r.get("n_unique_z", 0) for r in rows), n_selected=len(slices),
        rejected={r["series_uid"]: r["reject"] for r in rows if "reject" in r},
        n_errors=sum(r["n_errors"] for r in rows),
        per_split={sp: dict(subjects=sum(1 for s in subjects if split_of[s] == sp),
                            slices=sum(1 for r in slices if r["split"] == sp)) for sp in ("train", "val", "test")},
        fov_mm=_pct([r["fov_mm"] for r in kept]), spacing_mm=_pct([r["spacing_row_mm"] for r in kept]),
        thickness_mm=_pct([r["thickness_mm"] for r in kept]), z_spacing_mm=_pct([r["z_spacing_mm"] for r in kept]),
        z_extent_mm=_pct([r["z_extent_mm"] for r in kept]), selected_per_series=_pct([r["n_selected"] for r in kept]),
        rows=_count([r["rows"] for r in kept]), intercept=_count([r["intercept"] for r in kept]),
        manufacturer=_count([r["manufacturer"] for r in kept]), kernel=_count([r["kernel"] for r in kept], 12),
        scan_seconds=time.time() - t0)
    json.dump(stats, open(out_data / "stats.json", "w"), indent=1)
    print(f"[step12] kept {len(kept)}/{len(rows)} series, {len(subjects)} subjects, {len(slices)} slices "
          f"(train/val/test {[stats['per_split'][s]['slices'] for s in ('train', 'val', 'test')]}); "
          f"FOV median {stats['fov_mm']['p50']:.0f} mm [{stats['fov_mm']['min']:.0f}, {stats['fov_mm']['max']:.0f}]; "
          f"rejected {len(stats['rejected'])}; {time.time() - t0:.0f} s", flush=True)
    return stats


def _pct(v):
    a = np.asarray(v, dtype=float)
    if a.size == 0:
        return {}
    return dict(min=float(a.min()), p10=float(np.percentile(a, 10)), p50=float(np.median(a)),
                p90=float(np.percentile(a, 90)), max=float(a.max()), mean=float(a.mean()))


def _count(v, top=8):
    c = {}
    for x in v:
        c[str(x)] = c.get(str(x), 0) + 1
    return dict(sorted(c.items(), key=lambda kv: -kv[1])[:top])


def _write_csv(path, rows):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------------
# 2. pixels: resampling onto the common grid
# --------------------------------------------------------------------------

def resample_slice(hu, spacing_rc, centre_rc, fov_mm, res, pad):
    """One slice (R x C, HU) with pixel spacing (sr, sc) mm -> res x res pixels covering
    fov_mm x fov_mm around the source pixel coordinate centre_rc (row, col).
    Gaussian anti-aliasing for the downsampling factor (skimage's rule, sigma =
    (factor - 1) / 2), then bilinear interpolation at the target pixel centres."""
    from scipy.ndimage import gaussian_filter, map_coordinates
    d = fov_mm / res
    fr, fc = d / spacing_rc[0], d / spacing_rc[1]
    sig = (max(fr - 1, 0) / 2, max(fc - 1, 0) / 2)
    src = gaussian_filter(hu, sig, mode="constant", cval=pad) if max(sig) > 0 else hu
    u = (np.arange(res) - (res - 1) / 2) * d
    rr = centre_rc[0] + u / spacing_rc[0]
    cc = centre_rc[1] + u / spacing_rc[1]
    grid = np.meshgrid(rr, cc, indexing="ij")
    return map_coordinates(src, grid, order=1, mode="constant", cval=pad).astype(np.float32)


def window_u8(hu, window):
    lo, hi = window
    return np.rint(np.clip((hu - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def pool2(a, k):
    n = a.shape[-1] // k
    return a.reshape(*a.shape[:-2], n, k, n, k).mean(axis=(-3, -1))


def _build_series(job):
    """Worker: read the selected slices of one series, HU, centre, resample, quantise."""
    import pydicom
    cfg, files = job["cfg"], job["files"]
    g, it = cfg["grid"], cfg["intensity"]
    lo_clip, hi_clip = it["hu_clip"]
    vols, geom = [], None
    for f in files:
        ds = pydicom.dcmread(f)
        hu = ds.pixel_array.astype(np.float32) * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
        k = axial_rotation([float(v) for v in ds.ImageOrientationPatient])
        if k:
            hu = np.rot90(hu, k)
        vols.append(np.clip(hu, lo_clip, hi_clip))
        if geom is None:
            geom = ([float(v) for v in ds.PixelSpacing], int(ds.Rows), int(ds.Columns))
    vol = np.stack(vols)
    spacing, R, C = geom
    centre = ((R - 1) / 2, (C - 1) / 2)
    if g["centre"] == "body":
        m = vol > -500.0
        if m.sum() > 0:
            idx = np.nonzero(m)
            centre = (float(idx[1].mean()), float(idx[2].mean()))
    shift_mm = ((centre[0] - (R - 1) / 2) * spacing[0], (centre[1] - (C - 1) / 2) * spacing[1])
    base = g["base_res"]
    out_hu = np.stack([resample_slice(s, spacing, centre, g["fov_mm"], base, g["pad_hu"]) for s in vol])
    out = {"hu_u16": np.rint(np.clip(out_hu, lo_clip, hi_clip) - lo_clip).astype(np.uint16)}
    for res in g["resolutions"]:
        a = out_hu if res == base else pool2(out_hu, base // res)
        out[f"u8_{res}"] = window_u8(a, it["window_hu"])
    example = None
    if job.get("example"):
        j = len(files) // 2
        example = dict(source_hu=vol[j].astype(np.float16), spacing=spacing, centre=centre, fov_mm=g["fov_mm"],
                       resampled_hu=out_hu[j].astype(np.float16), source_file=files[j])
    return dict(series_index=job["series_index"], n=len(files), shift_mm=shift_mm, out=out, example=example)


def build(cfg, out_data):
    cache = Path(cfg["cache_root"])
    cache.mkdir(parents=True, exist_ok=True)
    slices = list(csv.DictReader(open(out_data / "slices.csv")))
    N = len(slices)
    g, it = cfg["grid"], cfg["intensity"]
    base = g["base_res"]
    # jobs in series order; slices of a series are contiguous in the manifest
    jobs = []
    for r in slices:
        k = int(r["series_index"])
        if not jobs or jobs[-1]["series_index"] != k:
            jobs.append(dict(series_index=k, files=[], cfg=cfg))
        jobs[-1]["files"].append(r["file"])
    off, pos = {}, 0
    for j in jobs:
        off[j["series_index"]] = pos
        pos += len(j["files"])
    assert pos == N
    jobs[0]["example"] = True
    mm = {"hu": np.lib.format.open_memmap(cache / f"slices_{base}_hu.u16.npy", mode="w+", dtype=np.uint16, shape=(N, base, base))}
    for res in g["resolutions"]:
        mm[res] = np.lib.format.open_memmap(cache / f"slices_{res}.u8.npy", mode="w+", dtype=np.uint8, shape=(N, res, res))
    print(f"[step12] building {N} slices from {len(jobs)} series -> {cache} with {cfg['workers']} workers", flush=True)
    t0, done, shifts = time.time(), 0, {}
    with Pool(cfg["workers"]) as pool:
        for k, res_ in enumerate(pool.imap_unordered(_build_series, jobs, chunksize=1)):
            o, n = off[res_["series_index"]], res_["n"]
            mm["hu"][o:o + n] = res_["out"]["hu_u16"]
            for res in g["resolutions"]:
                mm[res][o:o + n] = res_["out"][f"u8_{res}"]
            shifts[res_["series_index"]] = res_["shift_mm"]
            if res_["example"] is not None:
                np.savez_compressed(out_data / "example_resample.npz", **res_["example"])
            done += n
            if k % 50 == 0 or k == len(jobs) - 1:
                print(f"[step12]   {k + 1}/{len(jobs)} series, {done}/{N} slices, {time.time() - t0:.0f} s", flush=True)
    for m in mm.values():
        m.flush()
    # manifest in memmap order (== slices.csv order)
    for i, r in enumerate(slices):
        r["index"] = i
        r["centre_shift_row_mm"], r["centre_shift_col_mm"] = shifts[int(r["series_index"])]
    _write_csv(cache / "manifest.csv", slices)
    info = dict(n=N, base_res=base, resolutions=g["resolutions"], fov_mm=g["fov_mm"], window_hu=it["window_hu"],
                hu_clip=it["hu_clip"], files={"hu": f"slices_{base}_hu.u16.npy", **{str(r): f"slices_{r}.u8.npy" for r in g["resolutions"]}},
                build_seconds=time.time() - t0, built=datetime.now(timezone.utc).isoformat(), seed=cfg["seed"],
                z_stride_mm=cfg["selection"]["z_stride_mm"], centre=g["centre"])
    json.dump(info, open(cache / "build.json", "w"), indent=1)
    json.dump(info, open(out_data / "build.json", "w"), indent=1)
    print(f"[step12] built {N} slices in {(time.time() - t0) / 60:.1f} min", flush=True)
    return info


# --------------------------------------------------------------------------
# 3. the dataset for later steps
# --------------------------------------------------------------------------

class LIDCSlices(Dataset):
    """8-bit windowed slices at one resolution from the memmap cache; items are
    ((1, res, res) float in [0, 1], 0) like the step-1 datasets (label unused)."""

    def __init__(self, cache_root, res, split, manifest=None, info=None):
        self.cache_root, self.res, self.split = str(cache_root), int(res), split
        self.info = info or json.load(open(Path(cache_root) / "build.json"))
        rows = manifest if manifest is not None else list(csv.DictReader(open(Path(cache_root) / "manifest.csv")))
        self.rows = [r for r in rows if r["split"] == split] if split != "all" else rows
        self.indices = np.array([int(r["index"]) for r in self.rows], dtype=np.int64)
        self._mm = None

    def _arr(self):
        if self._mm is None:  # opened lazily so the dataset pickles into DataLoader workers
            self._mm = np.load(Path(self.cache_root) / self.info["files"][str(self.res)], mmap_mode="r")
        return self._mm

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        a = np.asarray(self._arr()[self.indices[i]], dtype=np.float32) / 255.0
        return torch.from_numpy(a)[None], 0

    def hu(self, i):
        """The master-resolution HU slice (float32) of item i."""
        mm = np.load(Path(self.cache_root) / self.info["files"]["hu"], mmap_mode="r")
        return np.asarray(mm[self.indices[i]], dtype=np.float32) + self.info["hu_clip"][0]


def build_lidc(res, cfg=None):
    """DatasetBundle (train/val/test by subject) at resolution `res`."""
    import step1_datasets
    cfg = cfg or load_config()
    cache = Path(cfg["cache_root"])
    info = json.load(open(cache / "build.json"))
    manifest = list(csv.DictReader(open(cache / "manifest.csv")))
    ds = {sp: LIDCSlices(cache, res, sp, manifest, info) for sp in ("train", "val", "test")}
    return step1_datasets.DatasetBundle(
        f"lidc_{res}", ds["train"], ds["val"], ds["test"],
        meta=dict(source="TCIA LIDC-IDRI CT slices, step 12 cache", fov_mm=info["fov_mm"], window_hu=info["window_hu"],
                  split_note="by subject, seeded 80/10/10", n_classes=0, task="unconditional"))


# --------------------------------------------------------------------------
# 4. self-tests
# --------------------------------------------------------------------------

def verify(cfg, out_data):
    """(a) a disc of known physical radius resampled from two different scanner
    spacings lands on the same target pixels; (b) pooling and windowing are
    consistent; (c) the z selection keeps the stride; (d) the cache, if built,
    round-trips: u8 of the 256 cache == window of the HU cache."""
    res = {}
    g = cfg["grid"]
    fov, R = g["fov_mm"], 64
    outs = []
    for sp in (0.6, 0.95):
        n = int(round(fov / sp)) + 40
        yy, xx = np.meshgrid((np.arange(n) - (n - 1) / 2) * sp, (np.arange(n) - (n - 1) / 2) * sp, indexing="ij")
        disc = np.where(np.sqrt((yy - 20) ** 2 + (xx + 35) ** 2) < 80, 0.0, -1000.0).astype(np.float32)
        outs.append(resample_slice(disc, (sp, sp), ((n - 1) / 2, (n - 1) / 2), fov, R, -1000.0))
    d = fov / R
    yy, xx = np.meshgrid((np.arange(R) - (R - 1) / 2) * d, (np.arange(R) - (R - 1) / 2) * d, indexing="ij")
    ideal = np.where(np.sqrt((yy - 20) ** 2 + (xx + 35) ** 2) < 80, 0.0, -1000.0)
    interior = np.abs(np.sqrt((yy - 20) ** 2 + (xx + 35) ** 2) - 80) > 2 * d   # away from the edge
    res["disc_two_spacings_max_abs_diff_interior"] = float(np.abs(outs[0] - outs[1])[interior].max())
    res["disc_vs_ideal_max_abs_interior"] = float(max(np.abs(o - ideal)[interior].max() for o in outs))
    res["disc_area_rel_err"] = [float(((o > -500).sum() - (ideal > -500).sum()) / (ideal > -500).sum()) for o in outs]
    a = np.random.RandomState(0).uniform(-1000, 400, (256, 256)).astype(np.float32)
    res["pool_mean_preserved"] = float(abs(pool2(a, 4).mean() - a.mean()))
    res["window_roundtrip_max_hu"] = float(np.abs(window_u8(a, (-1000, 400)).astype(np.float32) / 255 * 1400 - 1000 - a).max())
    recs = [dict(modality="CT", z=float(z), iop=[1, 0, 0, 0, 1, 0], spacing=[0.7, 0.7], rows=512, cols=512, slope=1, intercept=-1024,
                 instance=i, sop=str(i), thickness=0.625, kernel="", kvp=120, recon_diameter=360) for i, z in enumerate(np.arange(0, 100, 0.625))]
    chosen, info = select_slices(recs + recs[:3], cfg["selection"])
    zs = np.array([r["z"] for r in chosen])
    res["selection_min_dz"] = float(np.diff(zs).min())
    res["selection_n_from_160_thin"] = len(chosen)
    res["selection_dedup_ok"] = bool(info["n_unique_z"] == len(recs))
    cache = Path(cfg["cache_root"])
    if (cache / "build.json").exists():
        info = json.load(open(cache / "build.json"))
        hu = np.load(cache / info["files"]["hu"], mmap_mode="r")
        u8 = np.load(cache / info["files"][str(info["base_res"])], mmap_mode="r")
        idx = np.random.RandomState(1).choice(info["n"], size=min(64, info["n"]), replace=False)
        idx.sort()
        h = np.asarray(hu[idx], dtype=np.float32) + info["hu_clip"][0]
        res["cache_u8_vs_hu_window_max_diff"] = int(np.abs(window_u8(h, info["window_hu"]).astype(int) - np.asarray(u8[idx]).astype(int)).max())
        u8_64 = np.load(cache / info["files"]["64"], mmap_mode="r")
        res["cache_64_vs_pooled_256_max_diff"] = int(np.abs(window_u8(pool2(h, 4), info["window_hu"]).astype(int) - np.asarray(u8_64[idx]).astype(int)).max())
    res["ok"] = bool(res["disc_two_spacings_max_abs_diff_interior"] < 1.0 and res["disc_vs_ideal_max_abs_interior"] < 1.0
                     and max(abs(v) for v in res["disc_area_rel_err"]) < 0.02 and res["pool_mean_preserved"] < 1e-3
                     and res["window_roundtrip_max_hu"] < 2.8 and res["selection_min_dz"] >= cfg["selection"]["z_stride_mm"] - 1e-6
                     and res["selection_dedup_ok"] and res.get("cache_u8_vs_hu_window_max_diff", 0) <= 1
                     and res.get("cache_64_vs_pooled_256_max_diff", 0) <= 1)
    json.dump(res, open(out_data / "verify.json", "w"), indent=1)
    print("[step12] verify:", json.dumps(res), flush=True)
    return res


# --------------------------------------------------------------------------
# 5. figures + fragments
# --------------------------------------------------------------------------

def collect(cfg, out_data, out_figs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cache = Path(cfg["cache_root"])
    stats = json.load(open(out_data / "stats.json")) if (out_data / "stats.json").exists() else {}
    info = json.load(open(cache / "build.json")) if (cache / "build.json").exists() else None
    lines = []
    if info:
        bundle = build_lidc(info["base_res"], cfg)
        tr = bundle.train
        g = torch.Generator().manual_seed(cfg["seed"])
        n_ex = cfg["figures"]["n_examples"]
        pick = torch.randperm(len(tr), generator=g)[:n_ex].tolist()
        ncol = int(math.ceil(math.sqrt(n_ex)))
        # (a) examples in the stored window, (b) the same in a lung window from the HU cache
        for tag, win in (("examples_256", None), ("examples_lungwindow", (-1350.0, 150.0))):
            fig, axes = plt.subplots(int(math.ceil(n_ex / ncol)), ncol, figsize=(1.9 * ncol, 1.95 * math.ceil(n_ex / ncol)))
            for ax, i in zip(axes.flat, pick):
                if win is None:
                    ax.imshow(tr[i][0][0].numpy(), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
                else:
                    ax.imshow(tr.hu(i), cmap="gray", vmin=win[0], vmax=win[1], interpolation="nearest")
                r = tr.rows[i]
                ax.set_title(f"{r['subject'][-4:]} z={float(r['z_mm']):.0f}", fontsize=6)
            for ax in axes.flat:
                ax.axis("off")
            fig.suptitle(f"LIDC train slices, {info['base_res']} px = {info['fov_mm']:.0f} mm; "
                         + (f"model window {info['window_hu'][0]:.0f}..{info['window_hu'][1]:.0f} HU" if win is None
                            else f"lung window {win[0]:.0f}..{win[1]:.0f} HU (from the HU cache)"), fontsize=8)
            fig.tight_layout()
            fig.savefig(out_figs / f"{tag}.png", dpi=150)
            plt.close(fig)
        # (c) the resolution ladder
        res_list = info["resolutions"]
        bundles = {r: build_lidc(r, cfg).train for r in res_list}
        fig, axes = plt.subplots(4, len(res_list), figsize=(2.0 * len(res_list), 8.2))
        for row, i in enumerate(pick[:4]):
            for col, r in enumerate(res_list):
                axes[row, col].imshow(bundles[r][i][0][0].numpy(), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
                axes[row, col].axis("off")
                if row == 0:
                    axes[row, col].set_title(f"{r} px ({info['fov_mm'] / r:.1f} mm/px)", fontsize=8)
        fig.suptitle("the same slices at every cached resolution (2x2 average pooling from 256)", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_figs / "resolutions.png", dpi=150)
        plt.close(fig)
        # (d) the resampling: source grid with the crop window vs the result
        ex_path = out_data / "example_resample.npz"
        if ex_path.exists():
            ex = np.load(ex_path)
            src, sp, c = ex["source_hu"].astype(np.float32), ex["spacing"], ex["centre"]
            fig, axes = plt.subplots(1, 2, figsize=(9, 4.4))
            axes[0].imshow(src, cmap="gray", vmin=-1000, vmax=400)
            half = float(ex["fov_mm"]) / 2
            axes[0].add_patch(plt.Rectangle((c[1] - half / sp[1], c[0] - half / sp[0]), 2 * half / sp[1], 2 * half / sp[0],
                                            fill=False, ec="r", lw=1))
            axes[0].plot([c[1]], [c[0]], "r+")
            axes[0].set_title(f"scanner grid {src.shape[0]}x{src.shape[1]} px, {sp[0]:.3f} mm/px ({src.shape[0] * sp[0]:.0f} mm); "
                              f"red = {2 * half:.0f} mm window at the body centroid", fontsize=7)
            axes[1].imshow(ex["resampled_hu"].astype(np.float32), cmap="gray", vmin=-1000, vmax=400)
            axes[1].set_title(f"common grid {ex['resampled_hu'].shape[0]} px = {2 * half:.0f} mm ({2 * half / ex['resampled_hu'].shape[0]:.2f} mm/px)", fontsize=7)
            for ax in axes:
                ax.axis("off")
            fig.tight_layout()
            fig.savefig(out_figs / "resample_check.png", dpi=150)
            plt.close(fig)
    if stats:
        series = list(csv.DictReader(open(out_data / "series.csv")))
        kept = [r for r in series if not r.get("reject")]
        fig, axes = plt.subplots(1, 5, figsize=(17, 3.2))
        axes[0].hist([float(r["fov_mm"]) for r in kept], bins=30); axes[0].axvline(cfg["grid"]["fov_mm"], color="r")
        axes[0].set(title="reconstruction FOV (mm); red = common grid", xlabel="mm")
        axes[1].hist([float(r["spacing_row_mm"]) for r in kept], bins=30); axes[1].set(title="pixel spacing (mm)", xlabel="mm")
        axes[2].hist([float(r["z_spacing_mm"]) for r in kept], bins=30); axes[2].set(title="slice spacing in z (mm)", xlabel="mm")
        axes[3].hist([int(r["n_selected"]) for r in kept], bins=30); axes[3].set(title=f"slices per series (z stride {cfg['selection']['z_stride_mm']} mm)")
        if info:
            hu = np.load(cache / info["files"]["hu"], mmap_mode="r")
            idx = np.sort(np.random.RandomState(0).choice(info["n"], 200, replace=False))
            h = np.asarray(hu[idx], dtype=np.float32) + info["hu_clip"][0]
            axes[4].hist(h.ravel(), bins=200, range=(-1024, 1500), log=True)
            for v in info["window_hu"]:
                axes[4].axvline(v, color="r")
            axes[4].set(title="HU (200 slices); red = model window", xlabel="HU")
        fig.tight_layout()
        fig.savefig(out_figs / "stats.png", dpi=130)
        plt.close(fig)
        ps = stats["per_split"]
        rej = stats["rejected"]
        rej_txt = (f"{len(rej)} series rejected ({'; '.join(_tex(v) for v in list(rej.values())[:4])}"
                   f"{', ...' if len(rej) > 4 else ''})" if rej else "no series rejected")
        lines.append(
            f"LIDC-IDRI: {stats['n_series']} CT series from {stats['n_subjects']} subjects, {stats['n_files']:,} DICOM files; "
            f"{stats['n_axial']:,} axial CT images, {stats['n_unique_z']:,} unique z positions; {rej_txt}. "
            f"Reconstruction FOV {stats['fov_mm']['min']:.0f}--{stats['fov_mm']['max']:.0f} mm (median {stats['fov_mm']['p50']:.0f}), "
            f"pixel spacing {stats['spacing_mm']['min']:.3f}--{stats['spacing_mm']['max']:.3f} mm, "
            f"z spacing {stats['z_spacing_mm']['min']:.2f}--{stats['z_spacing_mm']['max']:.2f} mm (median {stats['z_spacing_mm']['p50']:.2f}). "
            f"Common grid: {cfg['grid']['fov_mm']:.0f} mm field of view centred on the body centroid, {cfg['grid']['base_res']} px "
            f"({cfg['grid']['fov_mm'] / cfg['grid']['base_res']:.2f} mm/px), lower resolutions by 2$\\times$2 average pooling; "
            f"z stride $\\geq {cfg['selection']['z_stride_mm']}$ mm gives {stats['n_selected']:,} slices "
            f"({stats['selected_per_series']['p50']:.0f} per series, {stats['selected_per_series']['min']:.0f}--{stats['selected_per_series']['max']:.0f}): "
            f"train {ps['train']['slices']:,} ({ps['train']['subjects']} subjects), val {ps['val']['slices']:,} ({ps['val']['subjects']}), "
            f"test {ps['test']['slices']:,} ({ps['test']['subjects']}). Model window {cfg['intensity']['window_hu'][0]:.0f}..{cfg['intensity']['window_hu'][1]:.0f} HU "
            f"$\\to$ 8 bits ({(cfg['intensity']['window_hu'][1] - cfg['intensity']['window_hu'][0]) / 255:.1f} HU per grey level).")
    if (out_data / "verify.json").exists():
        v = json.load(open(out_data / "verify.json"))
        lines.append(f"Self-tests {'passed' if v['ok'] else 'FAILED'}: a disc resampled from 0.6 and 0.95 mm scanner grids agrees to "
                     f"{v['disc_two_spacings_max_abs_diff_interior']:.2g} HU away from its edge (area error "
                     f"{100 * max(abs(x) for x in v['disc_area_rel_err']):.2f}\\%), 8-bit window round-trip error "
                     f"$\\leq {v['window_roundtrip_max_hu']:.1f}$ HU"
                     + (f", cached 8-bit vs. HU window max difference {v['cache_u8_vs_hu_window_max_diff']} grey level(s)"
                        if "cache_u8_vs_hu_window_max_diff" in v else "") + ".")
    if info:
        lines.append(f"Cache built {info['built'][:10]} in {info['build_seconds'] / 60:.0f} min.")
    (out_data / "summary.tex").write_text("\n".join(lines) + "\n")
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--collect", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    out_root = REPO_ROOT / cfg["output_root"]
    out_data, out_figs = out_root / "data", out_root / "figures"
    out_data.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)
    if args.scan:
        scan(cfg, out_data)
    if args.build:
        build(cfg, out_data)
    if args.verify:
        verify(cfg, out_data)
    if args.collect:
        collect(cfg, out_data, out_figs)
    prov = dict(step="step12_lidc_dataset", command=" ".join(["python"] + (argv or sys.argv[1:])),
                finished=datetime.now(timezone.utc).isoformat(), config=cfg)
    with open(out_data / "provenance.json", "w") as f:
        json.dump(prov, f, indent=2)


if __name__ == "__main__":
    main()

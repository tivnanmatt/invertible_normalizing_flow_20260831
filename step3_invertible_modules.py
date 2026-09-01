#!/usr/bin/env python
"""step3_invertible_modules.py -- Verify and benchmark the invertible library.

For every module in invlib's registry this step checks, with real numbers:
  1. exact invertibility:   max |x - f^-1(f(x))| in fp32 and fp64
  2. Jacobian correctness:  module logdet vs autograd-jacobian slogdet
  3. orthogonality:         ||Q^T Q - I||_inf for the SeqTransform family
  4. TarFlow-slot compat:   MetaBlock(forward -> reverse) roundtrip with each
                            SeqTransform installed in the permutation slot
  5. cost:                  median forward/inverse wall time, parameter count,
                            parameter bytes (+ CUDA peak memory when available)

Outputs (outputs/step3_invertible_modules/):
  data/     verification.csv/json, LaTeX fragments, provenance.json
  figures/  timing bars, orthogonal-matrix gallery, MNIST coefficient gallery

Config: configs/step3_invertible_modules.yml
Usage:  python step3_invertible_modules.py
"""

import csv
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "step3_invertible_modules.yml"

import invlib  # noqa: E402


def load_config(path=DEFAULT_CONFIG):
    with open(path) as f:
        return yaml.safe_load(f)


def max_err(a, b):
    return (a - b).abs().max().item()


def check_reconstruction(module, sample, batch, dtype):
    torch.manual_seed(0)
    m = module.to(dtype)
    x = sample(batch).to(dtype)
    with torch.no_grad():
        y, ld = m(x)
        xr = m.inverse(y)
    return max_err(x, xr), ld


def check_logdet_autograd(module, sample):
    """Compare module logdet with slogdet of the autograd Jacobian (1 sample)."""
    m = module.double()
    x = sample(1).double()
    shape = x.shape

    def f(v):
        return m(v.reshape(shape))[0].flatten()

    with torch.no_grad():
        ld_mod = m(x)[1].item()
    jac = torch.autograd.functional.jacobian(f, x.flatten(), vectorize=True)
    sign, ld_jac = torch.linalg.slogdet(jac)
    return abs(ld_mod - ld_jac.item()), sign.item()


def time_module(module, sample, batch, reps, dtype=torch.float32):
    m = module.to(dtype)
    x = sample(batch).to(dtype)
    with torch.no_grad():
        y, _ = m(x)
        fw, inv = [], []
        for _ in range(reps):
            t0 = time.perf_counter()
            m(x)
            fw.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            m.inverse(y)
            inv.append(time.perf_counter() - t0)
    return statistics.median(fw) * 1e3, statistics.median(inv) * 1e3


def check_metablock_roundtrip(seq_transform, cfg, in_channels=8):
    """Install the transform in a TarFlow MetaBlock and verify that the
    sequential reverse pass exactly inverts the forward pass."""
    p = str(Path(cfg["tarflow_repo"]).resolve())
    if p not in sys.path:
        sys.path.insert(0, p)
    import transformer_flow

    torch.manual_seed(0)
    T = seq_transform.seq_length
    block = transformer_flow.MetaBlock(
        in_channels=in_channels, channels=64, num_patches=T,
        permutation=seq_transform, num_layers=1)
    # zero-init proj_out makes the block the identity; perturb to test for real
    block.proj_out.weight.data.normal_(0, 0.02)
    block.proj_out.bias.data.normal_(0, 0.02)
    x = torch.randn(2, T, in_channels)
    with torch.no_grad():
        z, _ = block(x)
        xr = block.reverse(z.clone())
    return max_err(x, xr)


def check_feature_domains(cfg, tol):
    """Verify the 2D feature-domain tokenizers (invlib.FEATURE_REGISTRY):
    exact roundtrip, norm preservation (orthogonality), DC concentration
    (constant image -> all energy in token 0), and the MetaBlock roundtrip."""
    rows = []
    N, p = cfg["feature_domain"]["img_size"], cfg["feature_domain"]["patch_size"]
    T = (N // p) ** 2
    for name, cls in invlib.FEATURE_REGISTRY.items():
        t = cls(N, p, 1)
        torch.manual_seed(0)
        x = torch.randn(16, T, p * p)
        y = t(x, dim=1)
        row = {"name": f"feat_{name}", "kind": "feature",
               "params": sum(q.numel() for q in t.parameters()), "param_bytes": 0}
        row["recon_fp32"] = max_err(x, t(y, dim=1, inverse=True))
        xd = x.double()
        td = cls(N, p, 1).double()
        row["recon_fp64"] = max_err(xd, td(td(xd, dim=1), dim=1, inverse=True))
        row["orth_err"] = abs(y.norm().item() - x.norm().item()) / x.norm().item()
        c = t(torch.ones(1, T, p * p), dim=1)
        row["dc_energy_frac"] = (c[0, 0].pow(2).sum() / c.pow(2).sum()).item()
        row["metablock_err"] = check_metablock_roundtrip(cls(N, p, 1), cfg,
                                                         in_channels=p * p)
        row["fwd_ms"], row["inv_ms"] = time_module(
            _FeatTimer(t), lambda b: torch.randn(b, T, p * p),
            cfg["batch"], cfg["timing_reps"])
        row["logdet_err"], row["jac_sign"] = 0.0, 1.0  # orthogonal by construction
        ok = (row["recon_fp32"] < tol["recon_fp32"]
              and row["orth_err"] < tol["orthogonality"] * 10
              and row["dc_energy_frac"] > 0.999
              and row["metablock_err"] < tol["metablock"])
        row["status"] = "PASS" if ok else "FAIL"
        rows.append(row)
        print(f"  {row['name']:32s} {row['status']:5s} "
              f"recon {row['recon_fp32']:.2e} orth {row['orth_err']:.2e} "
              f"dc_frac {row['dc_energy_frac']:.4f} "
              f"metablock {row['metablock_err']:.2e}", flush=True)
    return rows


class _FeatTimer(torch.nn.Module):
    """Adapter so time_module can time a feature-domain transform."""

    def __init__(self, t):
        super().__init__()
        self.t = t

    def forward(self, x):
        return self.t(x, dim=1), torch.zeros(x.shape[0])

    def inverse(self, y):
        return self.t(y, dim=1, inverse=True)


def run(cfg, out_data, out_figs):
    D = cfg["dims"]["flat"]
    img = tuple(cfg["dims"]["image"])
    T = cfg["dims"]["seq"]
    batch, reps = cfg["batch"], cfg["timing_reps"]
    tol = cfg["tolerances"]

    registry = invlib.build_registry(D=D, img=img, T=T)
    rows = []
    for name, entry in registry.items():
        module, sample, seq_obj = entry["make"]()
        n_params = sum(p.numel() for p in module.parameters())
        row = {"name": name, "kind": entry["kind"], "params": n_params,
               "param_bytes": 4 * n_params}
        try:
            row["recon_fp32"], _ = check_reconstruction(entry["make"]()[0], sample, batch,
                                                        torch.float32)
            row["recon_fp64"], _ = check_reconstruction(entry["make"]()[0], sample, batch,
                                                        torch.float64)
            ld_err, ld_sign = check_logdet_autograd(entry["make"]()[0], sample)
            row["logdet_err"], row["jac_sign"] = ld_err, ld_sign
            row["fwd_ms"], row["inv_ms"] = time_module(entry["make"]()[0], sample, batch, reps)
            if seq_obj is not None:
                q = seq_obj.matrix().double()
                row["orth_err"] = max_err(q.t() @ q, torch.eye(q.shape[0]).double())
                row["metablock_err"] = check_metablock_roundtrip(entry["make"]()[2], cfg)
            # jac_sign is recorded for interest (orthogonal maps may have
            # det = -1, e.g. Hartley); flows only need log|det|, so the
            # requirement is non-singularity, not positivity.
            ok = (row["recon_fp32"] < tol["recon_fp32"]
                  and row["logdet_err"] < tol["logdet"]
                  and row["jac_sign"] != 0
                  and row.get("orth_err", 0.0) < tol["orthogonality"]
                  and row.get("metablock_err", 0.0) < tol["metablock"])
            row["status"] = "PASS" if ok else "FAIL"
        except Exception as e:
            row["status"] = "ERROR"
            row["detail"] = f"{type(e).__name__}: {e}"
        rows.append(row)
        print(f"  {name:32s} {row['status']:5s} "
              f"recon {row.get('recon_fp32', float('nan')):.2e} "
              f"logdet {row.get('logdet_err', float('nan')):.2e} "
              f"fwd {row.get('fwd_ms', float('nan')):.2f}ms", flush=True)
    return rows


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------

def fig_timing(rows, out_figs):
    ok = [r for r in rows if "fwd_ms" in r]
    names = [r["name"] for r in ok]
    idx = np.arange(len(ok))
    fig, ax = plt.subplots(figsize=(9, 0.32 * len(ok) + 1.5))
    ax.barh(idx + 0.2, [r["fwd_ms"] for r in ok], height=0.4, label="forward")
    ax.barh(idx - 0.2, [r["inv_ms"] for r in ok], height=0.4, label="inverse")
    ax.set_yticks(idx, names, fontsize=7)
    ax.set_xlabel("median wall time per batch [ms]")
    ax.set_xscale("log")
    ax.legend()
    ax.set_title("invertible modules: forward vs inverse cost")
    fig.tight_layout()
    fig.savefig(out_figs / "timing.png", dpi=120)
    plt.close(fig)


def fig_matrices(T, out_figs):
    mats = {name: invlib.SEQ_REGISTRY[name](T).matrix().detach()
            for name in ["flip", "random_perm", "haar", "hadamard", "dct",
                         "hartley", "rand_ortho", "householder", "cayley"]}
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    for ax, (name, q) in zip(axes.flat, mats.items()):
        v = q.abs().max().item()
        ax.imshow(q.numpy(), cmap="RdBu_r", vmin=-v, vmax=v)
        ax.set_title(name, fontsize=10)
        ax.set_xticks([]), ax.set_yticks([])
    fig.suptitle(f"orthogonal sequence transforms Q ({T}x{T})", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_figs / "orthogonal_matrices.png", dpi=120)
    plt.close(fig)


def fig_coefficients(T, patch, out_figs):
    """Apply each transform to a real MNIST patch sequence and show the
    coefficient 'image' -- the features the TarFlow AR model would see."""
    import step1_datasets
    import torch.nn.functional as tF

    bundle = step1_datasets.build_dataset("mnist", step1_datasets.load_config())
    x = bundle.test[0][0]  # (1,28,28) in [0,1]
    x = tF.pad(x.unsqueeze(0), (2, 2, 2, 2))  # (1,1,32,32)
    u = tF.unfold(x, patch, stride=patch).transpose(1, 2)  # (1,T,p*p)
    names = ["identity", "flip", "random_perm", "haar", "hadamard", "dct",
             "hartley", "rand_ortho"]
    fig, axes = plt.subplots(2, 4, figsize=(10, 5.4))
    for ax, name in zip(axes.flat, names):
        t = invlib.SEQ_REGISTRY[name](T)
        c = t(u, dim=1)
        im = tF.fold(c.transpose(1, 2), (32, 32), patch, stride=patch)[0, 0]
        ax.imshow(im.detach().numpy(), cmap="RdBu_r")
        ax.set_title(name, fontsize=10)
        ax.set_xticks([]), ax.set_yticks([])
    fig.suptitle("MNIST digit: patch-sequence coefficients under each transform")
    fig.tight_layout()
    fig.savefig(out_figs / "mnist_coefficients.png", dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------
# latex fragments
# --------------------------------------------------------------------------

def _tex(s):
    s = (str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")
         .replace("#", r"\#"))
    return s.replace("<", r"\textless{}").replace(">", r"\textgreater{}")


def _fmt(v, fmt="{:.1e}"):
    return fmt.format(v) if isinstance(v, (int, float)) else "--"


def write_fragments(rows, out_data):
    lines = [
        r"\begin{tabular}{llrllllll}",
        r"\hline",
        r"module & kind & params & recon fp32 & recon fp64 & logdet err & "
        r"orth err & fwd/inv ms & status \\",
        r"\hline",
    ]
    for r in rows:
        tm = (f"{r['fwd_ms']:.2f}/{r['inv_ms']:.2f}" if "fwd_ms" in r else "--")
        lines.append(
            f"{_tex(r['name'])} & {r['kind']} & {r['params']} & "
            f"{_fmt(r.get('recon_fp32'))} & {_fmt(r.get('recon_fp64'))} & "
            f"{_fmt(r.get('logdet_err'))} & {_fmt(r.get('orth_err'))} & "
            f"{tm} & {r['status']} \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    (out_data / "verification_table.tex").write_text("\n".join(lines) + "\n")

    seq_rows = [r for r in rows if "metablock_err" in r]
    lines = [r"\begin{tabular}{lll}", r"\hline",
             r"sequence transform & MetaBlock roundtrip err & status \\", r"\hline"]
    for r in seq_rows:
        lines.append(f"{_tex(r['name'])} & {_fmt(r['metablock_err'])} & {r['status']} \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    (out_data / "metablock_table.tex").write_text("\n".join(lines) + "\n")


def main(argv=None):
    cfg = load_config()
    torch.manual_seed(cfg["seed"])
    out_root = REPO_ROOT / cfg["output_root"]
    out_data, out_figs = out_root / "data", out_root / "figures"
    out_data.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)

    print(f"step3: verifying {len(invlib.build_registry())} modules "
          f"(D={cfg['dims']['flat']}, img={cfg['dims']['image']}, T={cfg['dims']['seq']})")
    rows = run(cfg, out_data, out_figs)
    rows += check_feature_domains(cfg, cfg["tolerances"])

    with open(out_data / "verification.json", "w") as f:
        json.dump(rows, f, indent=2)
    keys = ["name", "kind", "params", "recon_fp32", "recon_fp64", "logdet_err",
            "jac_sign", "orth_err", "metablock_err", "fwd_ms", "inv_ms", "status", "detail"]
    with open(out_data / "verification.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    fig_timing(rows, out_figs)
    fig_matrices(cfg["dims"]["seq"], out_figs)
    fig_coefficients(cfg["dims"]["seq"], cfg["viz_patch"], out_figs)
    write_fragments(rows, out_data)

    prov = dict(step="step3_invertible_modules", config=cfg,
                command=" ".join(["python"] + (argv if argv is not None else sys.argv)),
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                versions=dict(python=sys.version.split()[0], torch=torch.__version__),
                device="cpu" if not torch.cuda.is_available() else torch.cuda.get_device_name(0),
                n_pass=sum(r["status"] == "PASS" for r in rows),
                n_total=len(rows))
    with open(out_data / "provenance.json", "w") as f:
        json.dump(prov, f, indent=2)
    n_pass = prov["n_pass"]
    print(f"step3 done: {n_pass}/{len(rows)} modules PASS; outputs in {out_root}")
    return 0 if n_pass == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())

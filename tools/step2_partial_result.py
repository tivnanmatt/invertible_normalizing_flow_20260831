"""Record a partial (interrupted) step-2 run from its metrics CSV so the
executable paper documents it honestly. Usage: python tools/step2_partial_result.py <run_name> "<reason>" """
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import step2_tarflow as s2  # noqa: E402

name, reason = sys.argv[1], sys.argv[2]
cfg = s2.load_config()
out_root = Path(s2.REPO_ROOT) / cfg["output_root"]
out_data, out_figs = out_root / "data", out_root / "figures"

rows = list(csv.DictReader(open(out_data / f"{name}_metrics.csv")))
rc = cfg["runs"][name]
result = {
    "name": name,
    "config": rc,
    "n_params": sum(1 for _ in ()) or None,  # unknown here; filled if model rebuilt
    "status": f"interrupted at epoch {rows[-1]['epoch']}/{rc['epochs']}: {reason}",
    "val_bpd_curve": [[int(r["epoch"]), float(r["val_bpd"])] for r in rows if r["val_bpd"]],
    "final_train_loss": float(rows[-1]["loss"]),
}
# recover the parameter count without CUDA
tf = s2.import_tarflow(cfg)
model = tf.Model(in_channels=rc["channel_size"], img_size=rc["img_size"],
                 patch_size=rc["patch_size"], channels=rc["channels"],
                 num_blocks=rc["blocks"], layers_per_block=rc["layers_per_block"],
                 nvp=True, num_classes=0)
result["n_params"] = sum(p.numel() for p in model.parameters())

json.dump(result, open(out_data / f"{name}_result.json", "w"), indent=2)
s2.plot_curves(name, out_data / f"{name}_metrics.csv", out_figs)
print("recorded:", result["status"])
print("val_bpd_curve:", result["val_bpd_curve"], "last train loss:", result["final_train_loss"])

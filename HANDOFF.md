# HANDOFF — running the queued GPU work

Operational instructions for an AI agent (or human) picking up this project on
**axis03**. Written 2026-09-01 after the host reboot that restored GPU access.
Read this top to bottom before running anything.

---

## 1. What this project is

Executable-paper repo for an A*-conference paper on invertible normalizing
flows. Convention: each pipeline stage is `step<N>_*.py` at the repo root +
one YAML in `configs/` + one **gitignored** output dir
`outputs/step<N>_*/{data,figures}`. `paper/main.tex` `\input`s auto-generated
LaTeX fragments straight out of those output dirs, so the compiled
`paper/main.pdf` is an exact record of what was actually run. `main.pdf` **is**
committed; `outputs/` and dataset caches are not.

Done so far: step1 (datasets), step2 (TarFlow via official apple/ml-tarflow),
step3 (invertible module library + verification), step4 (permutation-slot
ablation), step5 (feature-domain ablation + KDE DC model).

---

## 2. Environment and access

| item | value |
|---|---|
| host | `axis03`, user `staticct` (uid 1000). **No passwordless sudo.** |
| container | `recon-dev`, image `localhost:5000/recon:acr-20260819-v1`, torch 2.11.0+cu128 |
| repo (in container) | `/dev_ws/invertible_normalizing_flow_20260831` |
| repo (on host) | `/home/staticct/recon-dev/workspace/invertible_normalizing_flow_20260831` |
| TarFlow upstream | `/dev_ws/ml-tarflow` (official Apple clone, imported **unmodified**) |
| datasets | `/data/image_benchmarks` (same path host and container) |
| container HOME | `/dev_ws`; container uid is 1002 (`user`), not yours |

### GPU policy — **GPU 0 ONLY** (user directive, 2026-09-01)

GPU 1 fell off the PCIe bus under sustained load on 2026-08-31 and wedged the
host driver until the reboot. It enumerates healthy again, **but we do not use
it.** Enforced in three places:

- `docker-compose-recon-dev.yml`: `device_ids: ['0']`
- container env: `CUDA_VISIBLE_DEVICES=0`
- configs: every `device:` is `cuda:0`

The container therefore sees exactly **one** device. Consequence: **run jobs
sequentially.** The old "MNIST on GPU 0, CIFAR on GPU 1" pattern is gone. Do
not re-enable GPU 1 without the user explicitly asking.

### If the container is down

```bash
bash /home/staticct/recon-dev/bootstrap-recon-dev.sh
```
This does `compose up` **and** reinstalls the container-layer extras that are
lost on every recreate (texlive, poppler, acl, medmnist, the ssh symlink).
Verify afterwards:
```bash
docker exec -i recon-dev python -c "import torch;print(torch.cuda.is_available(), torch.cuda.device_count())"
# expect: True 1
```

---

## 3. Hard-won gotchas — read these, they cost hours

1. **`docker exec` needs `-i` for stdin.** A heredoc into `docker exec recon-dev
   python -` silently produces *no output*. Use `docker exec -i`.
2. **Run `git` INSIDE the container.** From the host you get
   `fatal: detected dubious ownership` (repo is owned by container uid 1002).
3. **Commit with explicit identity.** `deauth-github.sh` unsets the global
   git user, so use:
   `git -c user.name=tivnanmatt -c user.email=tivnanmatt@gmail.com commit ...`
4. **Auth scripts need `bash` + absolute path** — they are not executable and
   the shell cwd resets between calls:
   `bash /home/staticct/recon-dev/auth-github.sh`
5. **Never `pkill -f <pattern>` where the pattern matches your own command
   line** — it kills your shell (exit 143). Kill by PID.
6. **Backgrounding:** `cd X && nohup a & nohup b &` applies the `cd` only to
   the first job. Put the `cd` inside each `bash -c`, and always redirect to a
   log file.
7. **Sample grids can render all-black.** An undertrained AR reverse pass emits
   rare huge values; `save_image(normalize=True)` then maps everything else to
   black. Samplers clamp to [-1,1] (same as the official FID path) — keep that.
8. **`grad_clip: 1.0` is load-bearing** for coefficient-domain variants (Haar's
   DC coefficient is ~sqrt(T) x the mean; without clipping it diverges to NaN).

---

## 4. Current state (read before running)

Last commits: `be3d344` (GPU0 pinning), `4b0ea6d` (step5), `cdb573e` (step3+4).

### ⚠ STATE WARNING — step 4 outputs are inconsistent right now

A GPU timing probe on 2026-09-01 re-ran **only** `baseline_flip` at the `full`
profile. Its result JSON is therefore full-profile (test bpd 1.5439) while the
other nine variants on disk are still the CPU `reduced` profile. The ranking
table currently mixes the two and **is not a valid comparison**. Job 1 below
fixes this by re-running all ten variants at `full`. Do not publish or draw
conclusions from step-4 outputs until Job 1 completes.

### Also note

- **Step 2 CIFAR does NOT resume.** The crash at epoch 9/60 predates the
  optimizer-checkpoint code, so only `cifar10_uniform_model.pth` (weights, no
  optimizer/LR state) exists — there is no `*_ckpt.pth`. Job 3 starts from
  epoch 0. That is the right call anyway: 9 epochs of a 60-epoch cosine
  schedule with lost optimizer moments is not worth salvaging, and a clean run
  gives a self-consistent metrics CSV. (Future interruptions *will* resume —
  the checkpointing is in place now.)

---

## 5. The queue — run in this order, sequentially

All commands assume `cd /dev_ws/invertible_normalizing_flow_20260831` inside
the container. Runtimes measured on one RTX 4090.

### Job 1 — step 4 full profile (~10 min) — DO THIS FIRST, it fixes the warning above

```bash
docker exec recon-dev bash -c 'cd /dev_ws/invertible_normalizing_flow_20260831 && \
  nohup python step4_tarflow_ablation.py > /dev_ws/step4_full.log 2>&1 &'
```
10 variants x 12 epochs, ~1 min each. Profile auto-selects `full` when CUDA is
available. **Success:** log ends with `ranking (best first): ...` listing all
ten variants; `outputs/step4_tarflow_ablation/data/*_result.json` all carry
`"budget"` with `epochs: 12`.

### Job 2 — step 5 full profile (~6–10 min)

```bash
docker exec recon-dev bash -c 'cd /dev_ws/invertible_normalizing_flow_20260831 && \
  nohup python step5_feature_domains.py > /dev_ws/step5_full.log 2>&1 &'
```
6 variants. `haar2d_pure_dc` additionally fits a Gaussian KDE on the invariant
coarse token after training — its bpd legitimately includes the KDE term.
**Success:** `ranking (best first)` with all six variants.

### Job 3 — step 2 CIFAR-10 likelihood run (~3.5–4 h) — the long pole

```bash
docker exec recon-dev bash -c 'cd /dev_ws/invertible_normalizing_flow_20260831 && \
  nohup python step2_tarflow.py --runs cifar10_uniform > /dev_ws/step2_cifar.log 2>&1 &'
```
60 epochs at ~225 s/epoch, ~18 GB GPU memory, 25.9M params. Do **not** run it
concurrently with Jobs 1–2. Poll with
`docker exec recon-dev tail -3 /dev_ws/step2_cifar.log`.
Then rebuild the summary (do not skip — the run itself does not write it when
invoked with `--runs`):
```bash
docker exec recon-dev bash -c 'cd /dev_ws/invertible_normalizing_flow_20260831 && \
  python step2_tarflow.py --collect'
```
**Success:** `TEST BPD <x.xxxx>` in the log. **Sanity band:** expect roughly
3.0–3.6 bpd. Classic CIFAR-10 flow anchors are RealNVP 3.49 / Glow 3.35 /
Flow++ 3.08; the TarFlow paper's headline 2.99 is ImageNet-64 with ~18x the
parameters, so do not expect to match it here. Anything below ~2.5 or above
~5 means something is wrong — investigate, don't just report it.

---

## 6. After the runs — REQUIRED follow-through

### 6a. Update the paper prose (easy to forget, and it matters)

`paper/main.tex` currently quotes **CPU reduced-profile numbers inline** in the
step-4 and step-5 Analysis subsections (e.g. "baseline\_flip 2.474 ~ cayley
2.528", "haar 4.390", the step-5 "dct2d\_alt 2.686 / haar2d\_alt 2.823"). The
tables and figures regenerate automatically from `outputs/`, but **this prose
does not.** After Jobs 1–2 you must update those sentences to the new
full-profile numbers, or the paper's narrative will contradict its own tables.
Also update the "run on CPU / GPU down" caveats — they are now obsolete.

Check whether the *conclusions* still hold at full scale, and say so honestly
if they changed. In the reduced run the headline findings were: (i) cayley
(learnable near-identity rotation) was the only variant competitive with the
official baseline, winning without grad clipping; (ii) 1D dense global bases
were far behind; (iii) in step 5, pixel/coefficient *alternation* was nearly
competitive while pure coefficient domains were not. If 12 GPU epochs on full
MNIST reverses any of that, report the reversal — do not massage it.

### 6b. Rebuild the PDF (twice, for refs)

```bash
docker exec recon-dev bash -c 'cd /dev_ws/invertible_normalizing_flow_20260831/paper && \
  pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1; \
  pdflatex -interaction=nonstopmode main.tex 2>&1 | grep -E "^!" | head -5; \
  pdfinfo main.pdf | grep Pages'
```
No `^!` lines = clean compile.

### 6c. Commit and push

```bash
docker exec recon-dev bash -c 'cd /dev_ws/invertible_normalizing_flow_20260831 && \
  git -c user.name=tivnanmatt -c user.email=tivnanmatt@gmail.com \
  commit -qam "step4+step5 full GPU profile; step2 cifar complete"'

bash /home/staticct/recon-dev/auth-github.sh >/dev/null 2>&1
docker exec recon-dev bash -c 'cd /dev_ws/invertible_normalizing_flow_20260831 && git push origin main'
bash /home/staticct/recon-dev/deauth-github.sh >/dev/null 2>&1
docker exec recon-dev ls /dev_ws/.ssh   # MUST report "No such file or directory"
```
**Always deauth after pushing** — this box is credential-free at rest by
design. Verify the wipe.

---

## 7. Failure modes

| symptom | meaning / action |
|---|---|
| `CUDA unknown error` / `cuda: False` | driver wedged again. Check `nvidia-smi`. If a GPU shows `Unknown Error`, it fell off the bus — **you cannot fix this**, it needs a host reboot, and you have no sudo. Tell the user; do not attempt privilege escalation. |
| loss `nan` | almost certainly a coefficient-domain variant without `grad_clip`. Confirm `grad_clip: 1.0` is in the config. |
| all-black sample grid | outlier values + `normalize=True`; ensure the sampler clamps to [-1,1]. |
| `dubious ownership` | you ran git on the host. Run it in the container. |
| a job vanished silently | check its log file; background jobs leave no transcript marker. |

---

## 8. Do NOT

- Use GPU 1, or change `device_ids` / `CUDA_VISIBLE_DEVICES`, without the user
  asking.
- Reboot the host, or escalate via the `docker` group to get root. Other users
  (joe, chiara, aoibhe, lir-study) run containers here; a reboot SIGKILLs them
  all. Ask the user.
- Modify anything under `/dev_ws/ml-tarflow` — the scientific claim is that the
  official model code is used **unmodified**.
- Commit `outputs/` (gitignored by design) or delete `data_noclip/`, which is a
  deliberately archived negative result.
- Report a number you did not actually produce, or quietly drop a variant that
  failed. Partial results are fine; unlabeled partial results are not.

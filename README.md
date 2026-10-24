# seg2mri

**Segmentation-conditioned MRI synthesis with diffusion, trained from scratch on public data.**

Given a tumour label map, generate the four MRI modalities (T1, T1ce, T2, FLAIR).

Training is **paired**: each mask is the label map of the same subject whose MRI is
the target, which is the same supervision Med-DDPM uses. So this is not a claim that
paired data can be avoided. The question this repo answers is narrower and more
useful: **does the diffusion process earn its cost, compared with a single forward
pass through the same U-Net on the same data?**

Three methods, one test split, one table:

| method | what it is |
|---|---|
| `diffusion` | 1000-step cosine schedule, v-prediction, DDIM at sampling |
| `regressor` | the same U-Net, one forward pass from a zero image, L1 + gradient loss |
| `class-mean` | training class means painted into the test mask — no spatial model at all |

Everything here trains from scratch on a public dataset. There is no third-party
checkpoint and no restricted cohort: `python scripts/run_all.py` reproduces the
table from an empty `runs/` directory.

```
python -m pip install -e ".[dev]"
python -m seg2mri.smoke --out data/smoke && python -m seg2mri.train --config configs/smoke.yaml
# synthetic data -> 30 training steps -> sample -> evaluate, on CPU, ~60 s
```

---

## Why this repo exists

Most segmentation-to-image work reports numbers that a reader cannot reproduce:
a private cohort, an unpublished checkpoint, an evaluation script that lives in
one lab. This one is deliberately small and deliberately open — the data is
public, the split is written to disk at preprocessing time, and the metrics are
recomputed from the exported NIfTI files by an independent script.

It also reports the comparison that a generative-model paper usually leaves out.
The regressor is a strong baseline: it sees exactly the same conditioning, has
exactly the same capacity, and costs one forward pass instead of fifty. If the
diffusion model does not beat it, that is what the table says. A negative result
at this scale is still a result, and it is the reason the class-mean row is in
the same table rather than in a footnote.

It is small on purpose: a 14M-parameter model at 128×128 trained on one free GPU.
Numbers from it are **not** comparable to BraTS-scale results reported in the
literature, and this README does not claim otherwise.

## Results

Results are generated, never typed. One command runs the whole chain:

```bash
python scripts/run_all.py --work runs/kaggle      # ~4-5 h on a free T4
```

It downloads MSD Task01, preprocesses, trains both models, samples held-out cases,
evaluates all three methods, and writes `results/RESULTS.md`:

```bash
python -m seg2mri.compare \
    --run class-mean=results/class_mean.csv \
    --run regressor=results/regressor.csv \
    --run diffusion=results/diffusion.csv \
    --reference regressor --out results/RESULTS.md
```

| comparison | why it is in the table |
|---|---|
| **diffusion** | the model |
| **regressor** (`seg2mri.regress`) | same U-Net, same data, one step — isolates what the diffusion process contributes |
| **class-mean baseline** (`seg2mri.baseline`) | the best you can do from the mask alone with no spatial model — anything below this is not a result |

Metrics: 3D SSIM, MS-SSIM (2D axial, mean over slices), PSNR, NRMSE. Definitions
and normalisation are fixed in [docs/PROTOCOL.md](docs/PROTOCOL.md) — they are
the part of a paper that reviewers actually check.

## Quickstart (CPU, no data download)

```bash
python -m pip install -e ".[dev]"
python -m seg2mri.smoke --out data/smoke
python -m seg2mri.train --config configs/smoke.yaml
python -m seg2mri.regress --config configs/smoke.yaml --out runs/smoke_reg
python -m seg2mri.sample --ckpt runs/smoke/ckpt-last.pt --split test --n 1 \
    --out samples/smoke --export-real --device cpu
python -m seg2mri.sample --ckpt runs/smoke_reg/ckpt-last.pt --split test --n 1 \
    --out samples/smoke_reg --device cpu
python -m seg2mri.evaluate --gen samples/smoke/gen --real samples/smoke/real \
    --out samples/smoke/results.csv
python -m seg2mri.evaluate --gen samples/smoke_reg/gen --real samples/smoke/real \
    --out samples/smoke_reg/results.csv
pytest -q          # 17 tests, including this exact sequence end to end
```

The smoke model is 30 steps old, so its samples are noise. That is the point:
it proves the plumbing, not the science.

## Real training

### 1. Data

[Medical Segmentation Decathlon](http://medicaldecathlon.com/), Task01 BrainTumour —
484 cases, 4 co-registered modalities, 3 tumour labels. Public S3 mirror (~6 GB):

```bash
wget https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar
tar -xf Task01_BrainTumour.tar
```

Preprocess (80/10/10 split by case, written to `meta.json`; resize to 128³;
per-modality 0.5–99.5 percentile normalisation to [-1, 1]):

```bash
python -m seg2mri.preprocess_msd --src Task01_BrainTumour --out data/msd_prepared \
    --size 128 --mode both
```

### 2. Train

```bash
python -m seg2mri.train --config configs/demo2d.yaml     # 13.9M params, ~2-3 h on a free T4
python -m seg2mri.regress --config configs/reg2d.yaml    # same U-Net, one step, ~1 h
python -m seg2mri.train --config configs/demo3d.yaml     # 8.6M params at 96³, slower
```

For the 3D config, preprocess separately: `--size 96 --mode 3d --out data/msd_prepared_96`.
[`kaggle/README.md`](kaggle/README.md) has the free-GPU route from a VS Code terminal.
`scripts/run_all.py` runs all of the above, in order, and skips stages whose output exists.

Checkpoints store the EMA weights **and** the config that produced them, so a
checkpoint is self-describing — including which architecture it holds (`arch` in
the file). `runs/<name>/log.csv` has loss and sec/step.

### 3. Sample and evaluate

```bash
python -m seg2mri.sample --ckpt runs/demo2d/ckpt-last.pt --split test --n 10 \
    --out samples/demo2d --export-real
python -m seg2mri.sample --ckpt runs/reg2d/ckpt-last.pt --split test --n 10 \
    --out samples/reg2d
python -m seg2mri.evaluate --gen samples/demo2d/gen --real samples/demo2d/real \
    --out results/diffusion.csv
python -m seg2mri.evaluate --gen samples/reg2d/gen --real samples/demo2d/real \
    --out results/regressor.csv
python -m seg2mri.baseline --config configs/demo2d.yaml --out samples/class_mean
python -m seg2mri.evaluate --gen samples/class_mean --real samples/demo2d/real \
    --out results/class_mean.csv
python -m seg2mri.compare --run class-mean=results/class_mean.csv \
    --run regressor=results/regressor.csv --run diffusion=results/diffusion.csv \
    --reference regressor --out results/RESULTS.md
```

`sample` writes `.nii.gz` volumes on the same grid as the ground truth, plus a
PNG montage per case. DDIM is deterministic given `--seed`, so a case is
reproducible from the checkpoint; the regressor is deterministic outright.

## Layout

```
seg2mri/
  data.py            memmap slice dataset + volume dataset, splits from meta.json
  models.py          2D/3D U-Net, time-conditioned, optional bottleneck attention
  diffusion.py       cosine schedule, v-prediction, DDIM (eta = 0)
  train.py           EMA, warmup + cosine LR, CSV log, self-describing checkpoints
  regress.py         single-step U-Net (the control), L1 + gradient loss
  sample.py          DDIM or one forward pass over a volume, NIfTI export, montage
  evaluate.py        SSIM / MS-SSIM / PSNR / NRMSE, per case + mean ± std
  compare.py         merges per-method CSVs into one table with deltas
  baseline.py        class-mean painting (the baseline diffusion must beat)
  metrics.py         the metric definitions, used by both CLI and tests
  preprocess_msd.py  MSD Task01 -> memmap slices / npz volumes
  smoke.py           synthetic data for CPU and CI
scripts/run_all.py            download -> preprocess -> train x2 -> sample -> table -> zip
scripts/verify_metrics.py     standalone NIfTI verifier (no package import)
docs/PROTOCOL.md              splits, normalisation, metric definitions
configs/                      smoke, demo2d, demo3d, reg2d
```

## Design notes

- **The regressor is the point of the repo.** Same U-Net, same conditioning, same
  split, one forward pass. Any difference in the table is attributable to the
  diffusion process, not to capacity or data.
- **v-prediction, cosine schedule.** v-prediction keeps the loss well-conditioned
  at both ends of the schedule; the cosine schedule spends more steps in the
  middle, where structure is decided.
- **EMA weights are what `sample` loads.** Raw weights are kept in the same
  checkpoint for debugging.
- **DDIM, eta = 0.** Deterministic output matters more than stochastic variety
  when the point is a fair comparison against a deterministic baseline.
- **A slice model is a baseline, not the goal.** 2D slices are independent, so
  through-plane continuity is not modelled — visible as stitching artefacts in
  coronal and sagittal views. The 3D config exists for that reason; it is
  slower and its results are reported separately.
- **Attention is off in 3D.** At 96³ the bottleneck has 24³ = 13,824 tokens;
  quadratic attention costs more than it returns at this scale.

## Limitations

- One public dataset, one preprocessing, one split. No external cohort, no
  domain-shift test.
- 128³ (2D) / 96³ (3D) is well below clinical resolution.
- The comparison is at a deliberately small budget. A negative result here says
  this diffusion model, at this resolution and step count, did not beat one
  forward pass — not that diffusion cannot help at larger scale.
- SSIM and PSNR do not establish clinical utility; no downstream segmentation
  or reader study is included here.
- Small enhancing lesions are under-represented — the failure mode to look at
  first when reading the samples.

## Licence

Code MIT. MSD Task01 data is distributed under its own terms — check them before
redistributing any derived volume.

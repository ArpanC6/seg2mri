# Evaluation protocol

Fixed before the first run, so that no number in the README can be chosen after
seeing which metric it flatters. Everything here is enforced by code, not by
discipline.

## Data

| | |
|---|---|
| Source | MSD Task01 BrainTumour (public) |
| Cases | 484 (fewer if `--limit`) |
| Modalities | T1, T1ce, T2, FLAIR — generated jointly as 4 channels |
| Labels | 0 background, 1 edema, 2 non-enhancing tumour, 3 enhancing tumour |
| Split | 80 / 10 / 10 by **case**, seeded shuffle, written to `meta.json` |
| Supervision | **paired** — the mask is the same subject's label map, as in Med-DDPM |
| Preprocessing | per-modality 0.5–99.5 percentile clip → [0, 1] → [-1, 1]; trilinear resize to 128³ (2D) or 96³ (3D); masks nearest-neighbour |
| Storage | uint8 (quantisation step 2/255); identical for generated and real volumes |

Splits are by case, never by slice: slices from one subject never appear in two
splits. `tests/test_data.py` asserts the split sets are disjoint.

## Methods compared

| method | one-line definition |
|---|---|
| diffusion | conditional DDIM, 50 steps, eta = 0, EMA weights |
| regressor | same U-Net, zero image input, t = 0, one forward pass, L1 + 0.1 × gradient loss |
| class-mean | training-split mean intensity per class painted into the test mask |

All three see the same masks, the same split, and the same preprocessing. The
regressor differs from the diffusion model only in how the image is produced, so
it is the primary comparison; the class-mean row bounds what the mask alone
determines.

## Metrics

| metric | definition |
|---|---|
| SSIM (3D) | `skimage.metrics.structural_similarity`, `win_size=7`, `data_range=1`, full volume |
| MS-SSIM | Wang et al. 2003, 5 scales, Gaussian window 11×11 σ=1.5, computed per axial slice and averaged over slices |
| PSNR | `10 log10(1 / MSE)` after per-volume min-max normalisation |
| NRMSE | RMSE divided by the dynamic range of the reference volume |

Every volume — generated and real — is **independently** min-max normalised to
[0, 1] before scoring, and the NIfTI export already applies the same mapping
before the file is written. This removes global intensity scale and offset, which
is standard for synthesis evaluation but means a metric cannot detect a systematic
brightness error. It applies identically to all three methods, so the comparison
between them is unaffected; absolute values are not comparable to papers that use
a different convention. Report it if it matters to you; do not claim otherwise.

## What is comparable, and what is not

- **Within this repo:** yes. Same split, same preprocessing, same code path.
- **Diffusion vs regressor:** yes, and this is the comparison the repo exists for.
  Same U-Net, same conditioning, same optimiser budget family; only the number of
  forward passes differs.
- **Against BraTS 2021 results in the literature:** no. Different dataset,
  different resolution, different crop. The only honest comparison would be to
  re-run the published method on this split.
- **SSIM against a paper's MS-SSIM:** no. They differ by roughly 0.1–0.2 on
  brain MRI. If a table mixes them, the table is wrong.
- **FID:** not reported here. Slice-level FID against a 3D dataset is not the
  same quantity as volumetric FID, and mixing the two is a common error. If FID
  is added later, it goes in its own table with its own protocol line.

## Reporting rules

1. Mean ± standard deviation over held-out cases, with `n` stated.
2. Per-case CSV always shipped alongside any summary number, so a reader can
   check that the mean is not carried by three easy subjects.
3. The regressor and the class-mean baseline are reported in the same table as
   the diffusion model. A diffusion model that does not beat one forward pass,
   or painting class means with a mask, is not a result — and is reported as such.
4. Any change to preprocessing, split, or metric parameters invalidates earlier
   numbers. Re-run, do not edit.

## Reproducing a reported number

```bash
python -m seg2mri.sample --ckpt runs/demo2d/ckpt-last.pt --split test --n 10 \
    --out samples/diffusion --export-real        # deterministic given --seed 42
python -m seg2mri.sample --ckpt runs/reg2d/ckpt-last.pt --split test --n 10 \
    --out samples/regressor                      # deterministic outright
python -m seg2mri.evaluate --gen samples/diffusion/gen --real samples/diffusion/real \
    --out results/diffusion.csv
python -m seg2mri.evaluate --gen samples/regressor/gen --real samples/diffusion/real \
    --out results/regressor.csv
python scripts/verify_metrics.py --gen samples/diffusion/gen --real samples/diffusion/real \
    --out results/verify.csv                     # independent implementation
```

The last command deliberately does not import `seg2mri`: if the two scripts
disagree, one of them is wrong, and that is worth knowing before a reviewer
finds it.

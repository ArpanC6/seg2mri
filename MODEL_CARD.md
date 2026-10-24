# Model card — seg2mri

## What it is

A conditional diffusion model that generates four MRI modalities (T1, T1ce, T2,
FLAIR) from a tumour segmentation mask. Research code, not a medical device.

The same repo contains a **single-step regressor** with the same U-Net, trained on
the same pairs with L1 + gradient loss. It is the control: the diffusion model has
to beat it for the generative process to be worth its sampling cost. Both are
reported in one table, next to a class-mean baseline.

## Architecture

| | 2D config | 3D config |
|---|---|---|
| Estimator | U-Net, time-conditioned residual blocks | same, 3D convolutions |
| Parameters | 13.9 M | 8.6 M |
| Input | 4 noisy image channels + 3 mask channels | same |
| Output | 4 channels | 4 channels |
| Attention | bottleneck level | disabled (quadratic cost at 24³ tokens) |
| Diffusion | 1000 steps, cosine, v-prediction | same |
| Sampling | DDIM, 50 steps, eta = 0 | same |

Regressor: identical U-Net and conditioning, zero-filled image input and t = 0,
one forward pass, trained with L1 + 0.1 × gradient loss. It is loaded from the
same checkpoint format and sampled by the same script.

## Training data

MSD Task01 BrainTumour — 484 public cases, 80/10/10 split by case. Per-modality
0.5–99.5 percentile normalisation to [-1, 1], resized to 128³ (2D) or 96³ (3D).

Supervision is **paired**: the mask is the label map of the same subject whose MRI
is the target. Masks are not available without their image at training time.

## Intended use

- Research on segmentation-conditioned synthesis and on whether a diffusion
  process is justified against a deterministic regressor.
- A reference implementation for evaluation protocol and reproducibility.

## Out of scope

- Clinical decision support, diagnosis, or treatment planning.
- Reconstructing a subject's real scan. Given that subject's mask the model
  generates a plausible volume, which can differ substantially from the true
  image; it is not an image-recovery method.
- Anything requiring anatomical fidelity at clinical resolution. 128³/96³ is
  far below diagnostic resolution.

## Known failure modes

- Small enhancing lesions are under-represented relative to large ones.
- The 2D model has no through-plane consistency; coronal and sagittal views
  show stitching.
- Training data is a single source; no domain-shift evaluation is reported.
- Metrics are intensity-based (SSIM, MS-SSIM, PSNR, NRMSE) and do not measure
  clinical utility.

## Ethical notes

Synthetic medical images can be mistaken for real ones. Any figure produced from
this repo should be labelled as synthetic. Do not mix generated volumes into a
dataset used for training a diagnostic model without disclosure.

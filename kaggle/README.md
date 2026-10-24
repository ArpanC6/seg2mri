# Kaggle — one command, one notebook

You do **not** need to download the 6 GB dataset to your laptop. Everything runs
on Kaggle from a single command, and the run directory is saved as notebook output.

## Why this works

`/kaggle/temp` is scratch space that is not kept as output; `/kaggle/working` is.
`scripts/run_all.py` puts the raw download and the prepared arrays in `temp`, and
the runs, samples, metrics and zip in `working`. That keeps you well under the
20 GB limit.

## Setup (one time, 2 minutes)

1. kaggle.com → sign in.
2. **Create → New Notebook**.
3. Right sidebar → **Session options**:
   - Accelerator: **GPU T4 ×2**
   - Internet: **On**
4. Rename the notebook to `seg2mri-run`.

## Cell 1 — clone and check the wiring (~15 min)

```python
!git clone https://github.com/ArpanC6/seg2mri.git /kaggle/working/seg2mri
%cd /kaggle/working/seg2mri
!pip install -e . -q

!nvidia-smi --query-gpu=name,memory.total --format=csv
!python scripts/run_all.py --quick --work /kaggle/working/quick
```

`--quick` uses 60 cases, 128 px, 600 + 300 steps and 4 sampled cases. It is not a
result — it proves the download, preprocessing, both trainings, sampling, metrics
and the comparison table all work before you spend four hours on the real run.
Look for `results/RESULTS.md` at the end.

## Cell 2 — the real run (~4–5 h)

```python
%cd /kaggle/working/seg2mri
!python scripts/run_all.py --work /kaggle/working/run
```

Defaults: 300 cases, 128 px, 40,000 diffusion steps, 15,000 regressor steps,
DDIM-50, every test case sampled. Stages whose outputs already exist are skipped,
so if the session dies you can re-run this exact line and it continues.

If the diffusion model hits `CUDA out of memory`, halve the batch for both models:

```python
!python scripts/run_all.py --work /kaggle/working/run --batch 8
```

## Cell 3 — inspect and get the files out

```python
!cat /kaggle/working/run/results/RESULTS.md
!ls /kaggle/working/run/samples/diffusion | head
!du -sh /kaggle/working/run/seg2mri_results.zip
```

Then: right sidebar → **Output** → download `seg2mri_results.zip`. It contains
`results/` (CSVs + `RESULTS.md`), `samples/` (NIfTI volumes and montages),
`runs/` (configs and `log.csv`; checkpoints are excluded), and `MANIFEST.txt`
with the environment and a file listing.

For the email, attach **2–3 `.nii.gz` files from `samples/diffusion/gen/`** plus
the montages — that is what was asked for, and a few files fit in one mail.

## If the session dies

Free sessions are recycled. Interactive runs are lost when that happens, so for
the 4-hour run use **Save Version → Run All** instead: it executes in the
background and keeps the output even if you close the browser. The stage-skipping
above means a restart costs only the stage that was interrupted.

## Long-term alternative

Once the data is prepared, uploading the prepared folder as a private Kaggle
dataset means later sessions skip the download and preprocessing:

```bash
kaggle datasets init -p /kaggle/temp/msd_prepared_2d_128      # edit the generated json
kaggle datasets create -p /kaggle/temp/msd_prepared_2d_128 --dir-mode zip
```

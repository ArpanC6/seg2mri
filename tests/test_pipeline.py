"""End-to-end: train 8 steps -> sample -> evaluate -> baseline -> compare, through the CLIs."""

import subprocess
import sys

import yaml


def _run(args, cwd):
    result = subprocess.run([sys.executable, *args], cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise AssertionError(f"{' '.join(args)} failed\n{result.stdout}\n{result.stderr}")
    return result


def test_cli_end_to_end(tmp_path, smoke_root):
    cfg = {
        "seed": 0,
        "data": {"root": smoke_root, "mode": "2d", "augment": False},
        "model": {"base_channels": 16, "channel_mults": [1, 2], "attn_levels": []},
        "diffusion": {"timesteps": 20, "ddim_steps": 4},
        "train": {"steps": 8, "batch_size": 4, "warmup_steps": 2, "log_every": 4,
                  "ckpt_every": 0, "out_dir": str(tmp_path / "run"), "grad_loss_weight": 0.1},
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    _run(["-m", "seg2mri.train", "--config", str(cfg_path)], cwd=tmp_path)
    _run(["-m", "seg2mri.regress", "--config", str(cfg_path), "--out", str(tmp_path / "reg")], cwd=tmp_path)

    diff_ckpt = tmp_path / "run" / "ckpt-last.pt"
    reg_ckpt = tmp_path / "reg" / "ckpt-last.pt"
    assert diff_ckpt.exists() and reg_ckpt.exists()

    _run(["-m", "seg2mri.sample", "--ckpt", str(diff_ckpt), "--split", "test", "--n", "1",
          "--out", str(tmp_path / "samples"), "--export-real", "--device", "cpu"], cwd=tmp_path)
    _run(["-m", "seg2mri.sample", "--ckpt", str(reg_ckpt), "--split", "test", "--n", "1",
          "--out", str(tmp_path / "samples_reg"), "--device", "cpu"], cwd=tmp_path)
    assert len(list((tmp_path / "samples" / "gen").glob("*.nii.gz"))) == 4
    assert len(list((tmp_path / "samples_reg" / "gen").glob("*.nii.gz"))) == 4

    real = str(tmp_path / "samples" / "real")
    for name, gen in (("diffusion", tmp_path / "samples" / "gen"), ("regressor", tmp_path / "samples_reg" / "gen")):
        _run(["-m", "seg2mri.evaluate", "--gen", str(gen), "--real", real,
              "--out", str(tmp_path / f"{name}.csv")], cwd=tmp_path)
        rows = (tmp_path / f"{name}.csv").read_text(encoding="utf-8").strip().splitlines()
        assert len(rows) == 5  # header + 4 modalities

    _run(["-m", "seg2mri.baseline", "--config", str(cfg_path),
          "--out", str(tmp_path / "baseline")], cwd=tmp_path)
    _run(["-m", "seg2mri.evaluate", "--gen", str(tmp_path / "baseline"), "--real", real,
          "--out", str(tmp_path / "class_mean.csv")], cwd=tmp_path)
    assert len(list((tmp_path / "baseline").glob("*.nii.gz"))) == 4

    _run(["-m", "seg2mri.compare",
          "--run", f"class-mean={tmp_path / 'class_mean.csv'}",
          "--run", f"regressor={tmp_path / 'regressor.csv'}",
          "--run", f"diffusion={tmp_path / 'diffusion.csv'}",
          "--reference", "regressor", "--out", str(tmp_path / "RESULTS.md")], cwd=tmp_path)
    table = (tmp_path / "RESULTS.md").read_text(encoding="utf-8")
    for name in ("class-mean", "regressor", "diffusion"):
        assert name in table
    assert "Δ vs regressor" in table

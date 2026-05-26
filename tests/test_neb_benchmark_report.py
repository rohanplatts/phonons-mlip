from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml

from NEB import run_neb_raw_v2
from NEB.neb_tools import benchmark_report


class NEBBenchmarkReportTests(unittest.TestCase):
    def _copy_family_demo(self, root: Path, family: str = "mace") -> Path:
        src = Path(__file__).resolve().parents[1] / "demo" / "fine_tuning" / family
        dst = root / "demo" / "fine_tuning" / family
        shutil.copytree(src, dst, symlinks=True)
        return dst

    def _seed_benchmark_results(self, family_root: Path, config: dict) -> None:
        names = [str(name) for name in config["defaults"]["model_name"]]
        dft_root = family_root / "0_raw_inputs" / "output1"
        dft_data = np.loadtxt(dft_root / "neb.dat")
        dft_data = np.atleast_2d(dft_data)
        s = dft_data[:, 1].astype(float)
        e = dft_data[:, 2].astype(float) - float(dft_data[0, 2])

        image_dirs = sorted([path for path in dft_root.iterdir() if path.is_dir() and path.name.isdigit()])
        for name in names:
            model_root = family_root / "4_benchmark" / "results" / name
            raw_root = model_root / "raw"
            vasp_ci_root = raw_root / "vasp_ci"
            raw_root.mkdir(parents=True, exist_ok=True)
            vasp_ci_root.mkdir(parents=True, exist_ok=True)

            np.savez(raw_root / "neb_raw.npz", s_mlip=s, e_mlip=e)
            (raw_root / "summary.txt").write_text(
                f"barrier_eV={float(np.max(e))}\n"
                f"delta_e={float(e[-1])}\n",
                encoding="utf-8",
            )

            for image_dir in image_dirs:
                target_dir = vasp_ci_root / image_dir.name
                target_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(image_dir / "POSCAR", target_dir / "POSCAR")

            (model_root / "config.yml").write_text(
                yaml.safe_dump(
                    {
                        "defaults": {
                            "model_name": name,
                            "outputs_root": ".",
                        }
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

    def test_report_generation_is_idempotent_and_writes_plots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            family_root = self._copy_family_demo(root, "mace")
            config_path = family_root / "4_benchmark" / "config.yml"
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

            self._seed_benchmark_results(family_root, config)

            with patch.object(benchmark_report, "REPO_ROOT", root):
                reports = benchmark_report.generate_family_benchmark_report(config_path, config)
                self.assertEqual(len(reports), 2)
                reports_again = benchmark_report.generate_family_benchmark_report(config_path, config)
                self.assertEqual(len(reports_again), 2)

            self.assertTrue((family_root / "4_benchmark" / "plot" / "energy_profiles.png").exists())
            self.assertTrue((family_root / "4_benchmark" / "plot" / "path_fidelity.png").exists())
            self.assertTrue((family_root / "4_benchmark" / "plot" / "report.json").exists())
            self.assertTrue((family_root / "4_benchmark" / "plot" / "report.md").exists())

            readme = (family_root / "README.md").read_text(encoding="utf-8")
            self.assertIn("--report-benchmark", readme)
            self.assertEqual(readme.count(benchmark_report.BENCHMARK_START), 1)
            self.assertEqual(readme.count(benchmark_report.BENCHMARK_END), 1)

    def test_report_flag_skips_neb_rerun_when_benchmark_results_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            family_root = self._copy_family_demo(root, "mace")
            config_path = family_root / "4_benchmark" / "config.yml"
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            self._seed_benchmark_results(family_root, config)

            with (
                patch.object(run_neb_raw_v2, "maybe_fan_out", side_effect=AssertionError("fan-out should not run")),
                patch.object(run_neb_raw_v2, "generate_family_benchmark_report", return_value=[]) as report_mock,
            ):
                rc = run_neb_raw_v2.main(["--inputs", str(config_path), "--report-benchmark"], repo_root=root)

            self.assertEqual(rc, 0)
            self.assertEqual(report_mock.call_count, 1)

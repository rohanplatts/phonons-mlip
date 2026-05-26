from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from common import benchmarking
from mlip_phonons.config_classes import ExecutiveCfg, ModelCfg, OutputPlan, StructureCfg


class BenchmarkingTests(unittest.TestCase):
    def _write_supported_models(self, root: Path) -> Path:
        supported = root / "SUPPORTED_MODELS.yml"
        supported.write_text(
            """
models:
  mace-mpa-0-medium:
    environment: mace_env
  orb-v3-direct-inf-omat:
    environment: mace_env
  snb-model:
    environment: mace_env
""",
            encoding="utf-8",
        )
        return supported

    def test_scalar_model_passthrough_uses_one_launch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = root / "case"
            inputs.mkdir()
            config_path = inputs / "config.yml"
            config_path.write_text(
                """
defaults:
  model_name: mace-mpa-0-medium
  outputs_root: results
""",
                encoding="utf-8",
            )
            supported = self._write_supported_models(root)

            calls: list[dict[str, object]] = []

            def fake_run(command, cwd=None, check=False, env=None):
                calls.append({"command": command, "cwd": cwd, "check": check, "env": env})
                return subprocess.CompletedProcess(command, 0)

            with patch.object(benchmarking, "SUPPORTED_MODELS_PATH", supported), patch("subprocess.run", side_effect=fake_run):
                config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
                rc = benchmarking.maybe_fan_out("phonons", config_path=config_path, config=config)

            self.assertIsNone(rc)
            self.assertEqual(calls, [])

    def test_model_list_fanout_rewrites_config_and_selects_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = root / "case"
            inputs.mkdir()
            config_path = inputs / "config.yml"
            config_path.write_text(
                """
defaults:
  model_name:
    - mace-mpa-0-medium
    - orb-v3-direct-inf-omat
  outputs_root: results
  device: cpu
workflow:
  keep_me: yes
""",
                encoding="utf-8",
            )
            supported = self._write_supported_models(root)

            calls: list[dict[str, object]] = []

            def fake_run(command, cwd=None, check=False, env=None):
                calls.append({"command": command, "cwd": cwd, "check": check, "env": env})
                return subprocess.CompletedProcess(command, 0)

            with patch.object(benchmarking, "SUPPORTED_MODELS_PATH", supported), patch("subprocess.run", side_effect=fake_run):
                config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
                rc = benchmarking.maybe_fan_out("snb", config_path=config_path, config=config)

            self.assertEqual(rc, 0)
            self.assertEqual(len(calls), 2)

            benchmark_root = inputs / "results"
            first = benchmark_root / "mace-mpa-0-medium" / "config.yml"
            second = benchmark_root / "orb-v3-direct-inf-omat" / "config.yml"
            self.assertTrue(first.exists())
            self.assertTrue(second.exists())

            first_cfg = yaml.safe_load(first.read_text(encoding="utf-8"))
            second_cfg = yaml.safe_load(second.read_text(encoding="utf-8"))
            self.assertEqual(first_cfg["defaults"]["model_name"], "mace-mpa-0-medium")
            self.assertEqual(second_cfg["defaults"]["model_name"], "orb-v3-direct-inf-omat")
            self.assertEqual(first_cfg["defaults"]["outputs_root"], ".")
            self.assertEqual(second_cfg["defaults"]["outputs_root"], ".")
            self.assertEqual(first_cfg["workflow"]["keep_me"], True)
            self.assertEqual(second_cfg["workflow"]["keep_me"], True)

            self.assertEqual(
                calls[0]["command"],
                [
                    "conda",
                    "run",
                    "--no-capture-output",
                    "-n",
                    "mace_env",
                    "python",
                    "-m",
                    "defect_landscape.snb.cli",
                    "--config",
                    str(first),
                ],
            )
            self.assertEqual(
                calls[1]["command"],
                [
                    "conda",
                    "run",
                    "--no-capture-output",
                    "-n",
                    "mace_env",
                    "python",
                    "-m",
                    "defect_landscape.snb.cli",
                    "--config",
                    str(second),
                ],
            )
            self.assertEqual(calls[0]["cwd"], first.parent)
            self.assertEqual(calls[1]["cwd"], second.parent)

    def test_missing_model_or_environment_fails_early(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = root / "case"
            inputs.mkdir()
            config_path = inputs / "config.yml"
            config_path.write_text(
                """
defaults:
  model_name: missing-model
  outputs_root: results
""",
                encoding="utf-8",
            )
            supported = root / "SUPPORTED_MODELS.yml"
            supported.write_text("models: {}\n", encoding="utf-8")

            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            config["defaults"]["model_name"] = ["missing-model", "mace-mpa-0-medium"]
            with patch.object(benchmarking, "SUPPORTED_MODELS_PATH", supported):
                with self.assertRaises(KeyError):
                    benchmarking.maybe_fan_out("phonons", config_path=config_path, config=config)

            supported.write_text(
                """
models:
  missing-model:
    environment: ""
  mace-mpa-0-medium:
    environment: mace_env
""",
                encoding="utf-8",
            )
            config["defaults"]["model_name"] = ["missing-model", "mace-mpa-0-medium"]
            with patch.object(benchmarking, "SUPPORTED_MODELS_PATH", supported):
                with self.assertRaises(ValueError):
                    benchmarking.maybe_fan_out("phonons", config_path=config_path, config=config)

    def test_empty_model_list_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = root / "case"
            inputs.mkdir()
            config_path = inputs / "config.yml"
            config_path.write_text(
                """
defaults:
  model_name: []
  outputs_root: results
""",
                encoding="utf-8",
            )
            supported = self._write_supported_models(root)

            with patch.object(benchmarking, "SUPPORTED_MODELS_PATH", supported):
                with self.assertRaises(ValueError):
                    benchmarking.main(["phonons", "--inputs", str(config_path)])

    def test_config_rewrite_preserves_unrelated_keys(self) -> None:
        config = {
            "defaults": {
                "model_name": ["mace-mpa-0-medium"],
                "outputs_root": "results",
                "device": "cpu",
            },
            "workflow": {"keep_me": True},
            "nested": {"value": 42},
        }

        rewritten = benchmarking.rewrite_config_for_model(
            config,
            "orb-v3-direct-inf-omat",
            config_path=Path("demo/fine_tuning/orb/4_benchmark/config.yml"),
        )

        self.assertEqual(rewritten["defaults"]["model_name"], "orb-v3-direct-inf-omat")
        self.assertEqual(rewritten["defaults"]["outputs_root"], ".")
        self.assertEqual(rewritten["defaults"]["device"], "cpu")
        self.assertEqual(rewritten["workflow"], {"keep_me": True})
        self.assertEqual(rewritten["nested"], {"value": 42})

    def test_phonons_output_plan_no_longer_nests_model_name(self) -> None:
        exec_cfg = ExecutiveCfg(results_root=Path("results"))
        model_cfg = ModelCfg(
            name="mace-mpa-0-medium",
            environment="mace_env",
            model_path=Path("assets/models/mace/mace-mpa-0-medium.model"),
            default_structure="demo",
        )
        structure_cfg = StructureCfg(
            name="demo",
            group="pure",
            unitcell_path=Path("POSCAR"),
            primitive_cell_path=None,
            is_file_relaxed=False,
            supercell_matrix=(1, 1, 1),
            delta=0.01,
            want_band_structure=True,
            kpts=[1, 1, 1],
            npts=10,
            width_ev=0.1,
        )

        plan = OutputPlan.build_output_plan(exec_cfg, model_cfg, structure_cfg, Path("/tmp/run"))

        expected = Path("/tmp/run/results/demo")
        self.assertEqual(plan.results_root, expected)
        self.assertNotIn("mace-mpa-0-medium", str(plan.results_root))

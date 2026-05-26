from __future__ import annotations

import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from mlip_workflows.cli import parse_common
from mlip_workflows.config import workflow_context
from mlip_workflows.coupling.cli import main as coupling_main
from mlip_workflows.neb.cli import main as neb_main
from mlip_workflows.phonons.cli import main as phonons_main
from mlip_workflows.snb.cli import main as snb_main


class CommonCliContractTests(unittest.TestCase):
    def test_help_does_not_import_workflow_backends(self) -> None:
        modules = [
            "mlip_workflows.phonons.cli",
            "mlip_workflows.neb.cli",
            "mlip_workflows.snb.cli",
            "mlip_workflows.coupling.cli",
        ]
        for module in modules:
            with self.subTest(module=module):
                result = subprocess.run(
                    [sys.executable, "-m", module, "--help"],
                    cwd=Path(__file__).resolve().parents[1],
                    env={"PYTHONPATH": "src"},
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("--inputs", result.stdout)
                self.assertIn("--outputs", result.stdout)

    def test_config_discovery_prefers_input_folder_and_can_be_overridden(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = root / "case"
            inputs.mkdir()
            (inputs / "config.yml").write_text(
                """
defaults:
  model_name: input-model
workflows:
  neb:
    model_name: input-neb
""",
                encoding="utf-8",
            )
            override = root / "override.yml"
            override.write_text(
                """
defaults:
  model_name: override-model
workflows:
  neb:
    model_name: override-neb
""",
                encoding="utf-8",
            )

            ctx = workflow_context("neb", inputs=inputs)
            self.assertEqual(ctx.config_path, inputs / "config.yml")
            self.assertEqual(ctx.workflow["model_name"], "input-neb")

            override_ctx = workflow_context("neb", config=override, inputs=inputs)
            self.assertEqual(override_ctx.config_path, override)
            self.assertEqual(override_ctx.workflow["model_name"], "override-neb")

    def test_missing_input_config_fails_clearly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = root / "case"
            inputs.mkdir()

            with self.assertRaises(FileNotFoundError) as caught:
                workflow_context("neb", inputs=inputs)
            self.assertIn("Missing config.yml in input directory", str(caught.exception))

    def test_neb_reads_local_config_and_writes_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = root / "neb-case"
            inputs.mkdir()
            (inputs / "config.yml").write_text(
                """
defaults:
  device: cpu
  dtype: float64
workflows:
  neb:
    model_name: mace-mpa-0-medium
    poscar_i: POSCAR_i
    poscar_f: POSCAR_f
    outputs: results/neb
    settings:
      n_images_fallback: 5
""",
                encoding="utf-8",
            )
            (inputs / "POSCAR_i").write_text("i", encoding="utf-8")
            (inputs / "POSCAR_f").write_text("f", encoding="utf-8")
            (inputs / "neb.dat").write_text("0 0 0\n", encoding="utf-8")

            fake = types.ModuleType("NEB.run_neb_raw_v2")
            calls: list[list[str]] = []
            fake.main = lambda argv: calls.append(argv) or 0

            with patch.dict(sys.modules, {"NEB.run_neb_raw_v2": fake}):
                rc = neb_main(["--inputs", str(inputs)])

            self.assertEqual(rc, 0)
            self.assertTrue((inputs / "results" / "neb" / "resolved-config.yml").exists())
            self.assertIn("--config", calls[0])
            self.assertIn(str(inputs / "config.yml"), calls[0])
            self.assertIn(str(inputs / "POSCAR_i"), calls[0])
            self.assertIn(str(inputs / "POSCAR_f"), calls[0])

    def test_phonons_injects_shared_registry_into_legacy_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = root / "phonons-case"
            inputs.mkdir()
            (inputs / "config.yml").write_text(
                """
workflows:
  phonons:
    model_name: mace-mpa-0-medium
    structure_name: test-structure
    unitcell_path: POSCAR
    primitive_cell_path: primitive.poscar
    supercell_matrix: [1, 1, 1]
    kpts: [1, 1, 1]
    npts: 20
""",
                encoding="utf-8",
            )
            (inputs / "POSCAR").write_text("POSCAR", encoding="utf-8")
            (inputs / "primitive.poscar").write_text("primitive", encoding="utf-8")

            fake = types.ModuleType("mlip_phonons.main")
            calls: list[list[str]] = []

            def _fake_main(argv: list[str]) -> int:
                calls.append(argv)
                config_path = Path(argv[argv.index("--config") + 1])
                legacy = yaml.safe_load(config_path.read_text(encoding="utf-8"))
                self.assertIn("models", legacy)
                self.assertIn("mace-mpa-0-medium", legacy["models"])
                self.assertIn("structures", legacy)
                self.assertEqual(legacy["workflows"]["phonons"]["structure"], "test-structure")
                self.assertEqual(legacy["structures"]["pure"]["test-structure"]["unitcell_path"], "POSCAR")
                return 0

            fake.main = _fake_main

            with patch.dict(sys.modules, {"mlip_phonons.main": fake}):
                rc = phonons_main(["--inputs", str(inputs), "--outputs", str(inputs / "results")])

            self.assertEqual(rc, 0)
            self.assertTrue((inputs / "results" / "resolved-config.yml").exists())
            self.assertIn("--structure", calls[0])
            self.assertIn("test-structure", calls[0])

    def test_coupling_resolves_band_paths_relative_to_input_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = root / "coupling-case"
            inputs.mkdir()
            (inputs / "config.yml").write_text(
                """
phonon_coupling:
  threshold: 0.8
  band_ml_paths:
    - ml/band.yaml
""",
                encoding="utf-8",
            )
            (inputs / "CONTCAR_GS").write_text("gs", encoding="utf-8")
            (inputs / "CONTCAR_ES").write_text("es", encoding="utf-8")
            (inputs / "band.yaml").write_text("dft", encoding="utf-8")
            ml_dir = inputs / "ml"
            ml_dir.mkdir()
            (ml_dir / "band.yaml").write_text("ml", encoding="utf-8")

            fake_analysis = types.ModuleType("coupling_modes.coup_tools.phon_analysis")
            fake_plot = types.ModuleType("coupling_modes.coup_tools.phon_plot")
            captured: dict[str, object] = {}

            def _run(**kwargs):
                captured["run"] = kwargs
                return object()

            def _render_report(*args, **kwargs):
                captured["report"] = {"args": args, "kwargs": kwargs}
                return "report"

            fake_analysis.run = _run
            fake_plot.render_report = _render_report

            with patch.dict(
                sys.modules,
                {
                    "coupling_modes.coup_tools.phon_analysis": fake_analysis,
                    "coupling_modes.coup_tools.phon_plot": fake_plot,
                },
            ):
                rc = coupling_main(["--inputs", str(inputs), "--outputs", str(inputs / "results")])

            self.assertEqual(rc, 0)
            self.assertTrue((inputs / "results" / "resolved-config.yml").exists())
            band_ml_paths = captured["run"]["band_ml_paths"]
            self.assertEqual(len(band_ml_paths), 1)
            self.assertEqual(Path(band_ml_paths[0]), ml_dir / "band.yaml")


if __name__ == "__main__":
    unittest.main()

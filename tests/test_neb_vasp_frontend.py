from __future__ import annotations

import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

from NEB.vasp_frontend import run_vasp_neb_directory, translate_vasp_neb_directory
from NEB import run_neb_raw_v2


class VaspNebFrontendTests(unittest.TestCase):
    def _directory(
        self,
        root: Path,
        *,
        images: int = 7,
        incar: str | None = None,
        missing_poscar: str | None = None,
        missing_folder: str | None = None,
        extra_folder: str | None = None,
    ) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        (root / "INCAR").write_text(
            incar or "SYSTEM = test\nIMAGES = 7\n",
            encoding="utf-8",
        )
        for index in range(images + 2):
            folder = root / f"{index:02d}"
            if folder.name == missing_folder:
                continue
            folder.mkdir()
            if folder.name != missing_poscar:
                (folder / "POSCAR").write_text("minimal POSCAR\n", encoding="utf-8")
        if extra_folder is not None:
            (root / extra_folder).mkdir()
            (root / extra_folder / "POSCAR").write_text("minimal POSCAR\n", encoding="utf-8")
        return root

    def test_translation_maps_images_endpoints_and_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._directory(Path(tmp) / "neb")
            config = translate_vasp_neb_directory(root)
            workflow = config["workflows"]["neb"]

            self.assertEqual(workflow["n_images"], 9)
            self.assertEqual(workflow["poscar_i"], str((root / "00/POSCAR").resolve()))
            self.assertEqual(workflow["poscar_f"], str((root / "08/POSCAR").resolve()))
            self.assertEqual(workflow["vasp_inputs_dir"], str(root.resolve()))
            self.assertFalse(workflow["relax_endpoints"])
            self.assertNotIn("model_name", workflow)

    def test_negative_ediffg_sets_final_force_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._directory(Path(tmp) / "neb", incar="IMAGES = 7\nEDIFFG = -0.015\n")
            workflow = translate_vasp_neb_directory(root)["workflows"]["neb"]
            self.assertEqual(workflow["settings"], {"fmax_ci": 0.015})

    def test_negative_spring_maps_its_magnitude_to_both_neb_stages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._directory(Path(tmp) / "neb", incar="IMAGES = 7\nSPRING = -5\n")
            workflow = translate_vasp_neb_directory(root)["workflows"]["neb"]
            self.assertEqual(workflow["settings"], {"k_spring_mlip": 5.0, "k_spring": 5.0})

    def test_nonnegative_spring_warns_and_is_not_mapped(self) -> None:
        for spring in (0, 5):
            with self.subTest(spring=spring), tempfile.TemporaryDirectory() as tmp:
                root = self._directory(Path(tmp) / "neb", incar=f"IMAGES = 7\nSPRING = {spring}\n")
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    workflow = translate_vasp_neb_directory(root)["workflows"]["neb"]
                self.assertNotIn("settings", workflow)
                self.assertTrue(any("different NEB semantics" in str(item.message) for item in caught))

    def test_nsw_warns_without_changing_stage_limits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._directory(Path(tmp) / "neb", incar="IMAGES = 7\nNSW = 300\n")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                workflow = translate_vasp_neb_directory(root)["workflows"]["neb"]
            self.assertNotIn("settings", workflow)
            self.assertTrue(any("cannot be mapped defensibly" in str(item.message) for item in caught))

    def test_positive_ediffg_warns_and_keeps_mlips_force_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._directory(Path(tmp) / "neb", incar="IMAGES = 7\nEDIFFG = 0.02\n")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                workflow = translate_vasp_neb_directory(root)["workflows"]["neb"]
            self.assertNotIn("settings", workflow)
            self.assertTrue(any("energy-change criterion" in str(item.message) for item in caught))

    def test_fixed_cell_isif_values_are_accepted(self) -> None:
        for isif in (0, 1, 2):
            with self.subTest(isif=isif), tempfile.TemporaryDirectory() as tmp:
                root = self._directory(Path(tmp) / "neb", incar=f"IMAGES = 7\nISIF = {isif}\n")
                translate_vasp_neb_directory(root)

    def test_cell_changing_isif_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._directory(Path(tmp) / "neb", incar="IMAGES = 7\nISIF = 3\n")
            with self.assertRaisesRegex(ValueError, "fixed-cell NEB"):
                translate_vasp_neb_directory(root)

    def test_directives_and_incar_syntax_are_parsed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._directory(
                Path(tmp) / "neb",
                incar=(
                    "# mlip_workflow = neb\n"
                    "! MLIP_MODEL = mace-omat-0-medium\n"
                    "IMAGES = 7; ENCUT = 500 ! unrelated setting\n"
                ),
            )
            workflow = translate_vasp_neb_directory(root)["workflows"]["neb"]
            self.assertEqual(workflow["model_name"], "mace-omat-0-medium")

    def test_run_translates_in_memory_without_writing_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._directory(Path(tmp) / "neb")
            with patch("NEB.run_neb_raw_v2.run_neb_from_config", return_value=23) as runner:
                rc = run_vasp_neb_directory(root, repo_root=Path(tmp) / "repo")

            self.assertEqual(rc, 23)
            config, = runner.call_args.args
            self.assertEqual(config["workflows"]["neb"]["n_images"], 9)
            self.assertEqual(runner.call_args.kwargs["run_root"], root.resolve())
            self.assertFalse((root / "config.yml").exists())

    def test_cli_forwards_vasp_directory_without_yaml_lookup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._directory(Path(tmp) / "neb")
            with patch("NEB.vasp_frontend.run_vasp_neb_directory", return_value=31) as frontend:
                rc = run_neb_raw_v2.main(["--vasp", str(root)], repo_root=Path(tmp) / "repo")

            self.assertEqual(rc, 31)
            frontend.assert_called_once_with(root, repo_root=Path(tmp) / "repo")
            self.assertFalse((root / "config.yml").exists())

    def test_cli_rejects_vasp_with_config_or_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._directory(Path(tmp) / "neb")
            for conflicting in ("--config", "--inputs"):
                with self.subTest(conflicting=conflicting):
                    with self.assertRaises(SystemExit):
                        run_neb_raw_v2.main(
                            ["--vasp", str(root), conflicting, str(Path(tmp) / "other")],
                            repo_root=Path(tmp) / "repo",
                        )

    def test_cli_rejects_legacy_vasp_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._directory(Path(tmp) / "neb")
            for extra in (("--n-images", "11"), ("model-name",)):
                with self.subTest(extra=extra):
                    with self.assertRaises(SystemExit):
                        run_neb_raw_v2.main(
                            ["--vasp", str(root), *extra],
                            repo_root=Path(tmp) / "repo",
                        )

    def test_invalid_incar_values_fail(self) -> None:
        cases = [
            ("SYSTEM = test\n", "missing required IMAGES"),
            ("IMAGES = seven\n", "must be an integer"),
            ("IMAGES = 0\n", "at least 1"),
            ("IMAGES = 7\n# MLIP_WORKFLOW = phonons\n", "must be NEB"),
            ("IMAGES = 7\n# MLIP_MODEL =\n", "cannot be empty"),
            ("IMAGES = 7\n# MLIP_MODEL = first\n# MLIP_MODEL = second\n", "repeated"),
            ("IMAGES = 7\nEDIFFG = nan\n", "finite"),
            ("IMAGES = 7\nEDIFFG = -0.01\nEDIFFG = -0.02\n", "repeated EDIFFG"),
            ("IMAGES = 7\nSPRING = nan\n", "finite"),
            ("IMAGES = 7\nSPRING = -5\nSPRING = -4\n", "repeated SPRING"),
            ("IMAGES = 7\nNSW = -1\n", "non-negative"),
            ("IMAGES = 7\nNSW = 10\nNSW = 20\n", "repeated NSW"),
        ]
        for incar, expected in cases:
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as tmp:
                root = self._directory(Path(tmp) / "neb", incar=incar)
                with self.assertRaisesRegex(ValueError, expected):
                    translate_vasp_neb_directory(root)

    def test_invalid_image_layout_fails_clearly(self) -> None:
        cases = [
            {"missing_poscar": "04"},
            {"missing_poscar": "00"},
            {"missing_poscar": "08"},
            {"missing_folder": "04"},
            {"extra_folder": "09"},
        ]
        for options in cases:
            with self.subTest(options=options), tempfile.TemporaryDirectory() as tmp:
                root = self._directory(Path(tmp) / "neb", **options)
                with self.assertRaisesRegex(ValueError, "POSCAR|expected.*observed"):
                    translate_vasp_neb_directory(root)

    def test_malformed_workflow_directive_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._directory(
                Path(tmp) / "neb",
                incar="IMAGES = 7\n# MLIP_WORKFLOW NEB\n",
            )
            with self.assertRaisesRegex(ValueError, "malformed"):
                translate_vasp_neb_directory(root)

    def test_missing_root_files_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "neb"
            with self.assertRaisesRegex(ValueError, "does not exist"):
                translate_vasp_neb_directory(root)
            root.mkdir()
            with self.assertRaisesRegex(ValueError, "missing INCAR"):
                translate_vasp_neb_directory(root)


if __name__ == "__main__":
    unittest.main()

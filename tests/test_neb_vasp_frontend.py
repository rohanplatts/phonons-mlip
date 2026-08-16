from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from NEB.vasp_frontend import run_vasp_neb_directory, translate_vasp_neb_directory


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
            self.assertNotIn("model_name", workflow)

    def test_directives_and_incar_syntax_are_parsed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._directory(
                Path(tmp) / "neb",
                incar=(
                    "# mlip_workflow = neb\n"
                    "! MLIP_MODEL = mace-omat-0-medium\n"
                    "IMAGES = 7; NSW = 100 ! unrelated setting\n"
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

    def test_invalid_incar_values_fail(self) -> None:
        cases = [
            ("SYSTEM = test\n", "missing required IMAGES"),
            ("IMAGES = seven\n", "must be an integer"),
            ("IMAGES = 0\n", "at least 1"),
            ("IMAGES = 7\n# MLIP_WORKFLOW = phonons\n", "must be NEB"),
            ("IMAGES = 7\n# MLIP_MODEL =\n", "cannot be empty"),
            ("IMAGES = 7\n# MLIP_MODEL = first\n# MLIP_MODEL = second\n", "repeated"),
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

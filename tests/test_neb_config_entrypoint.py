from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from NEB import run_neb_raw_v2


class NebConfigEntrypointTests(unittest.TestCase):
    def _config(self) -> dict:
        return {
            "defaults": {
                "model_name": "global-model",
                "device": "cpu",
                "dtype": "float64",
            },
            "neb": {
                "defaults": {
                    "model_name": "legacy-model",
                    "models_root": "legacy-models",
                },
                "settings": {"fmax_mlip_guess": 0.07},
            },
            "workflows": {
                "neb": {
                    "model_name": "workflow-model",
                    "poscar_i": "initial/POSCAR",
                    "poscar_f": "final/POSCAR",
                    "n_images": 9,
                    "results_root": "results/neb",
                    "vasp_inputs_dir": "vasp-inputs",
                    "settings": {"steps_ci": 42},
                }
            },
        }

    def test_raw_config_lowers_to_resolved_inputs_and_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs, defaults = run_neb_raw_v2._build_neb_inputs_and_defaults(
                self._config(),
                run_root=root,
                repo_root=Path(__file__).resolve().parents[1],
            )

            self.assertEqual(inputs.model_name, "workflow-model")
            self.assertEqual(inputs.n_images, 9)
            self.assertEqual(inputs.poscar_i, root / "initial" / "POSCAR")
            self.assertEqual(inputs.poscar_f, root / "final" / "POSCAR")
            self.assertEqual(inputs.models_root, root / "legacy-models")
            self.assertEqual(inputs.results_root, root / "results" / "neb")
            self.assertEqual(inputs.vasp_inputs_dir, root / "vasp-inputs")
            self.assertEqual(inputs.device, "cpu")
            self.assertEqual(inputs.dtype, "float64")
            self.assertEqual(defaults.fmax_mlip_guess, 0.07)
            self.assertEqual(defaults.steps_ci, 42)

    def test_cli_n_images_overrides_config_n_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            default_inputs, _ = run_neb_raw_v2._build_neb_inputs_and_defaults(
                self._config(),
                run_root=root,
                repo_root=Path(__file__).resolve().parents[1],
            )
            parsed = run_neb_raw_v2._parse_args(
                ["--n-images", "11"],
                default_config_path=root / "config.yml",
                default_inputs=default_inputs,
            )

            self.assertEqual(parsed.n_images, 11)

    def test_run_from_config_uses_shared_engine_without_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(run_neb_raw_v2, "_run_neb", return_value=17) as engine:
                rc = run_neb_raw_v2.run_neb_from_config(
                    self._config(),
                    run_root=root,
                    repo_root=Path(__file__).resolve().parents[1],
                )

            self.assertEqual(rc, 17)
            inputs, defaults = engine.call_args.args
            self.assertEqual(inputs.n_images, 9)
            self.assertEqual(inputs.poscar_i, root / "initial" / "POSCAR")
            self.assertEqual(defaults.steps_ci, 42)
            self.assertFalse(engine.call_args.kwargs["auto_submit_vasp"])
            self.assertFalse((root / "config.yml").exists())

    def test_yaml_cli_path_uses_shared_loaded_config_engine(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.yml").write_text(
                yaml.safe_dump(self._config(), sort_keys=False),
                encoding="utf-8",
            )

            with (
                patch.object(run_neb_raw_v2, "_run_neb", return_value=0) as engine,
                patch.object(run_neb_raw_v2, "maybe_fan_out", return_value=None),
                patch.object(run_neb_raw_v2.env_manager, "dispatch_if_needed", return_value=None),
            ):
                rc = run_neb_raw_v2.main(
                    ["--inputs", str(root), "--n-images", "11"],
                    repo_root=Path(__file__).resolve().parents[1],
                )

            self.assertEqual(rc, 0)
            inputs, _ = engine.call_args.args
            self.assertEqual(inputs.n_images, 11)


if __name__ == "__main__":
    unittest.main()

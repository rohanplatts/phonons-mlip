from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import yaml

from common import environment


class EnvironmentTests(unittest.TestCase):
    def _registry(self, root: Path) -> Path:
        path = root / "SUPPORTED_MODELS.yml"
        path.write_text(
            yaml.safe_dump(
                {"models": {"mace": {"environment": "mace_env"}, "orb": {"environment": "orb_env"}}}
            ),
            encoding="utf-8",
        )
        return path

    def test_model_environment_lookup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry = self._registry(Path(tmp))
            models = environment.load_supported_models(registry)
            self.assertEqual(environment.required_environment("mace", models, registry_path=registry), "mace_env")

    def test_unknown_model_is_clear(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry = self._registry(Path(tmp))
            with self.assertRaisesRegex(KeyError, "unknown"):
                environment.required_environment("unknown", registry_path=registry)

    def test_current_environment_does_not_relaunch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry = self._registry(Path(tmp))
            env = {"CONDA_DEFAULT_ENV": "mace_env"}
            with patch.object(environment, "run_in_environment") as run:
                result = environment.dispatch_if_needed(
                    "mace", ["python", "-m", "workflow"], env=env, registry_path=registry
                )
            self.assertIsNone(result)
            run.assert_not_called()

    def test_wrong_environment_builds_conda_command_and_preserves_args(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry = self._registry(Path(tmp))
            command = ["/usr/bin/python", "-m", "NEB.run_neb_raw_v2", "--vasp", "/tmp/case"]
            with patch.object(environment, "run_in_environment", return_value=0) as run:
                result = environment.dispatch_if_needed(
                    "orb", command, cwd=Path(tmp), env={}, registry_path=registry
                )
            self.assertEqual(result, 0)
            run.assert_called_once_with("orb_env", command, cwd=Path(tmp), model_name="orb")

    def test_guard_prevents_recursive_dispatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry = self._registry(Path(tmp))
            env = {environment.DISPATCH_GUARD: "1", "CONDA_DEFAULT_ENV": "wrong_env"}
            with self.assertRaisesRegex(RuntimeError, "dispatch guard"):
                environment.dispatch_if_needed("mace", ["python", "-m", "workflow"], env=env, registry_path=registry)

    def test_run_in_environment_sets_guard_and_pythonpath(self) -> None:
        with patch.dict(os.environ, {}, clear=True), patch(
            "common.environment.subprocess.run",
            return_value=subprocess.CompletedProcess([], 0),
        ) as run:
            result = environment.run_in_environment("mace_env", ["python", "-m", "workflow"])
        self.assertEqual(result, 0)
        command, = run.call_args.args
        self.assertEqual(command[:5], ["conda", "run", "--no-capture-output", "-n", "mace_env"])
        child_env = run.call_args.kwargs["env"]
        self.assertEqual(child_env[environment.DISPATCH_GUARD], "1")
        self.assertIn("src", child_env["PYTHONPATH"])

    def test_failed_environment_launch_names_model_and_environment(self) -> None:
        output = StringIO()
        with patch(
            "common.environment.subprocess.run",
            return_value=subprocess.CompletedProcess([], 17),
        ), redirect_stdout(output):
            result = environment.run_in_environment(
                "missing_env", ["python", "-m", "workflow"], model_name="orb"
            )
        self.assertEqual(result, 17)
        self.assertIn("missing_env", output.getvalue())
        self.assertIn("orb", output.getvalue())


if __name__ == "__main__":
    unittest.main()

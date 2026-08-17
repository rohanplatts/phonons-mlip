from __future__ import annotations

import io
import stat
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from NEB.vasp_submit import (
    copy_known_submission_script,
    find_known_submission_script,
    submit_vasp_ci,
)


class VaspSubmissionTests(unittest.TestCase):
    def test_only_known_script_is_discovered_and_copied(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "inputs"
            destination = root / "vasp_ci"
            source.mkdir()
            destination.mkdir()
            (source / "run.sh").write_text("#!/bin/sh\n", encoding="utf-8")
            self.assertIsNone(find_known_submission_script(source))
            self.assertIsNone(copy_known_submission_script(source, destination))

            known = source / "vasp_bunya.sh"
            known.write_text("#!/bin/sh\n", encoding="utf-8")
            known.chmod(known.stat().st_mode | stat.S_IXUSR)
            copied = copy_known_submission_script(source, destination)
            assert copied is not None
            self.assertEqual(copied.name, "vasp_bunya.sh")
            self.assertEqual(copied.read_text(encoding="utf-8"), "#!/bin/sh\n")
            self.assertTrue(copied.stat().st_mode & stat.S_IXUSR)

    def test_missing_sbatch_is_a_manual_handoff(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ci = Path(tmp)
            (ci / "vasp_bunya.sh").write_text("#!/bin/sh\n", encoding="utf-8")
            output = io.StringIO()
            with patch("NEB.vasp_submit.shutil.which", return_value=None), redirect_stdout(output):
                submit_vasp_ci(ci)
            self.assertIn("VASP-ready inputs", output.getvalue())
            self.assertIn("vasp_bunya.sh", output.getvalue())

    def test_sbatch_command_uses_ci_directory_and_script(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ci = Path(tmp)
            script = ci / "vasp_bunya.sh"
            script.write_text("#!/bin/sh\n", encoding="utf-8")
            with (
                patch("NEB.vasp_submit.shutil.which", return_value="/usr/bin/sbatch"),
                patch("NEB.vasp_submit.subprocess.run", return_value=type("Result", (), {"returncode": 0, "stdout": "Submitted batch job 42", "stderr": ""})()) as run,
            ):
                submit_vasp_ci(ci)
            run.assert_called_once_with(
                ["/usr/bin/sbatch", f"--chdir={ci.resolve()}", str(script.resolve())],
                cwd=ci.resolve(),
                check=False,
                text=True,
                capture_output=True,
            )

    def test_sbatch_failure_does_not_raise(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ci = Path(tmp)
            (ci / "vasp_bunya.sh").write_text("#!/bin/sh\n", encoding="utf-8")
            result = type("Result", (), {"returncode": 1, "stdout": "", "stderr": "bad job"})()
            with patch("NEB.vasp_submit.shutil.which", return_value="/usr/bin/sbatch"), patch(
                "NEB.vasp_submit.subprocess.run", return_value=result
            ):
                submit_vasp_ci(ci)

    def test_no_script_reports_manual_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = io.StringIO()
            with redirect_stdout(output):
                submit_vasp_ci(Path(tmp))
            self.assertIn("No unambiguous submission script", output.getvalue())


if __name__ == "__main__":
    unittest.main()

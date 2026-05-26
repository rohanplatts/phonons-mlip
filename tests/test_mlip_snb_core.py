from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from defect_landscape.snb.cli import _case_name, build_parser, normalise_argv
from defect_landscape.snb.compare import compare_dft
from defect_landscape.snb.config import load_config
from defect_landscape.snb.dft_parse import parse_dft_folder
from defect_landscape.snb.io import import_snb_inputs, read_csv_rows, write_csv_rows
from defect_landscape.snb.relax import relax_model
from defect_landscape.snb.cluster import cluster_model
from defect_landscape.snb.report import write_report
from defect_landscape.snb.select import SELECTED_FIELDS, select_clusters
from defect_landscape.snb.vasp import prepare_dft


POSCAR = """Cs Pb I
1.0
4.0 0.0 0.0
0.0 4.0 0.0
0.0 0.0 4.0
Cs Pb I
1 1 1
Direct
0.0 0.0 0.0
0.5 0.5 0.5
0.25 0.25 0.25
"""

H_POSCAR = """H
1.0
5.0 0.0 0.0
0.0 5.0 0.0
0.0 0.0 5.0
H
1
Direct
0.0 0.0 0.0
"""


def require_pymatgen(test_case: unittest.TestCase) -> None:
    try:
        import pymatgen  # noqa: F401
    except ImportError as exc:
        test_case.skipTest(str(exc))


class MlipSnbCoreTests(unittest.TestCase):
    def test_cli_shortcut_and_boolean_flags(self) -> None:
        self.assertEqual(normalise_argv([]), ["run"])
        self.assertEqual(normalise_argv(["--config", "config.yml"]), ["--config", "config.yml", "run"])
        self.assertEqual(
            normalise_argv(["mace-mpa-0-medium", "--snb-dir", "inputs", "--case-name", "case"]),
            ["run", "mace-mpa-0-medium", "--snb-dir", "inputs", "--case-name", "case"],
        )
        parser = build_parser(load_config())
        args = parser.parse_args(["run", "mace-mpa-0-medium", "--snb-dir", "inputs", "--case-name", "case", "--no-include-vdw"])
        self.assertFalse(args.include_vdw)

    def test_case_name_is_inferred_from_defect_or_snb_folder(self) -> None:
        parser = build_parser(load_config())
        config = load_config()

        args = parser.parse_args(["run", "mace-mpa-0-medium", "--defect", "/work/v_I_q0/POSCAR"])
        self.assertEqual(_case_name(args, config), "v_I_q0")

        args = parser.parse_args(["run", "mace-mpa-0-medium", "--snb-dir", "/work/snb_outputs"])
        self.assertEqual(_case_name(args, config), "snb_outputs")

        args = parser.parse_args(["generate", "--bulk", "/work/bulk/POSCAR", "--defect", "/work/v_Br_q+1/POSCAR", "--oxidation-states", "Cs=1"])
        self.assertEqual(_case_name(args, config), "v_Br_q+1")

    def test_config_supports_oxidation_states_and_prepare_dft_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.yml"
            config_path.write_text(
                """
snb:
  defaults:
    bulk: /work/bulk/POSCAR
    defect: /work/v_I_q0/POSCAR
    oxidation_states:
      Cs: 1
      Pb: 2
      I: -1
    prepare_dft: true
    copy_potcar: true
"""
            )
            config = load_config(config_path)

            self.assertEqual(config.defaults.oxidation_states, {"Cs": 1, "Pb": 2, "I": -1})
            self.assertTrue(config.defaults.prepare_dft)
            self.assertTrue(config.defaults.copy_potcar)

    def test_import_existing_snb_folder_discovers_nested_poscars(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "snb"
            (source / "Dimer").mkdir(parents=True)
            (source / "Unperturbed").mkdir()
            (source / "Dimer" / "POSCAR").write_text(POSCAR)
            (source / "Unperturbed" / "POSCAR").write_text(POSCAR)

            manifest = import_snb_inputs(snb_dir=source, results_root=root / "results", case_name="case")
            rows = read_csv_rows(manifest)

            self.assertEqual(len(rows), 2)
            self.assertEqual({row["candidate_id"] for row in rows}, {"Dimer", "Unperturbed"})
            for row in rows:
                self.assertTrue(Path(row["staged_poscar"]).exists())

    def test_select_respects_energy_window_and_copies_poscar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            analysis = root / "case" / "analysis"
            reps = root / "reps"
            reps.mkdir(parents=True)
            for idx in range(3):
                (reps / f"CONTCAR_{idx}").write_text(POSCAR)
            write_csv_rows(
                analysis / "clusters_model.csv",
                [
                    {
                        "model_name": "model",
                        "model_label": "model",
                        "cluster_id": 0,
                        "n_members": 1,
                        "best_candidate_id": "a",
                        "best_mlip_energy_eV": -10.0,
                        "best_mlip_dE_eV": 0.0,
                        "representative_contcar": str(reps / "CONTCAR_0"),
                        "members": "a",
                    },
                    {
                        "model_name": "model",
                        "model_label": "model",
                        "cluster_id": 1,
                        "n_members": 1,
                        "best_candidate_id": "b",
                        "best_mlip_energy_eV": -9.7,
                        "best_mlip_dE_eV": 0.3,
                        "representative_contcar": str(reps / "CONTCAR_1"),
                        "members": "b",
                    },
                    {
                        "model_name": "model",
                        "model_label": "model",
                        "cluster_id": 2,
                        "n_members": 1,
                        "best_candidate_id": "c",
                        "best_mlip_energy_eV": -9.2,
                        "best_mlip_dE_eV": 0.8,
                        "representative_contcar": str(reps / "CONTCAR_2"),
                        "members": "c",
                    },
                ],
                [
                    "model_name",
                    "model_label",
                    "cluster_id",
                    "n_members",
                    "best_candidate_id",
                    "best_mlip_energy_eV",
                    "best_mlip_dE_eV",
                    "representative_contcar",
                    "members",
                ],
            )

            selected = select_clusters(results_root=root, case_name="case", model_name="model", energy_window_eV=0.5, max_clusters=10)
            rows = read_csv_rows(selected)

            self.assertEqual([row["cluster_id"] for row in rows], ["0", "1"])
            for row in rows:
                self.assertTrue(Path(row["selected_poscar"]).exists())

    def test_prepare_dft_symlinks_potcar_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            selected_poscar = root / "selected" / "POSCAR"
            selected_poscar.parent.mkdir(parents=True)
            selected_poscar.write_text(POSCAR)
            write_csv_rows(
                root / "case" / "analysis" / "selected_for_dft_manifest.csv",
                [
                    {
                        "selection_id": "model_cluster_000",
                        "case_name": "case",
                        "model_name": "model",
                        "model_label": "model",
                        "cluster_id": 0,
                        "best_candidate_id": "a",
                        "best_mlip_energy_eV": -10.0,
                        "best_mlip_dE_eV": 0.0,
                        "representative_contcar": str(selected_poscar),
                        "selected_poscar": str(selected_poscar),
                        "source_models": "model",
                        "source_model_clusters": "model:0",
                    }
                ],
                SELECTED_FIELDS,
            )
            template = root / "template"
            template.mkdir()
            for name in ["INCAR", "KPOINTS", "POTCAR", "submit.sh"]:
                (template / name).write_text(name)

            manifest = prepare_dft(results_root=root, case_name="case", vasp_inputs_dir=template)
            rows = read_csv_rows(manifest)
            dft_dir = Path(rows[0]["dft_dir"])

            self.assertTrue((dft_dir / "POSCAR").exists())
            self.assertTrue((dft_dir / "POTCAR").is_symlink())

    def test_compare_dft_writes_energy_and_geometry_metrics(self) -> None:
        require_pymatgen(self)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            selected_poscar = root / "selected" / "POSCAR"
            selected_poscar.parent.mkdir(parents=True)
            selected_poscar.write_text(H_POSCAR)
            write_csv_rows(
                root / "case" / "analysis" / "selected_for_dft_manifest.csv",
                [
                    {
                        "selection_id": "model_cluster_000",
                        "case_name": "case",
                        "model_name": "model",
                        "model_label": "model",
                        "cluster_id": 0,
                        "best_candidate_id": "a",
                        "best_mlip_energy_eV": -10.0,
                        "best_mlip_dE_eV": 0.0,
                        "representative_contcar": str(selected_poscar),
                        "selected_poscar": str(selected_poscar),
                        "source_models": "model",
                        "source_model_clusters": "model:0",
                    }
                ],
                SELECTED_FIELDS,
            )
            dft_manifest = prepare_dft(results_root=root, case_name="case")
            dft_dir = Path(read_csv_rows(dft_manifest)[0]["dft_dir"])
            (dft_dir / "CONTCAR").write_text(H_POSCAR)
            (dft_dir / "OSZICAR").write_text(" 1 F= -.500000 E0= -.500000 d E =0\n")

            comparison = compare_dft(results_root=root, case_name="case")
            rows = read_csv_rows(comparison)
            geometry = read_csv_rows(root / "case" / "analysis" / "geometry_comparisons.csv")

            self.assertEqual(len(rows), 1)
            self.assertIn(rows[0]["top1_correct"].lower(), {"true", "1"})
            self.assertEqual(float(rows[0]["dft_energy_penalty_eV"]), 0.0)
            self.assertEqual(len(geometry), 1)
            self.assertEqual(float(geometry[0]["mlip_to_dft_rms_A"]), 0.0)

    def test_dft_parser_reports_restart_for_missing_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dft_dir = Path(tmp)
            (dft_dir / "POSCAR").write_text(POSCAR)
            parsed = parse_dft_folder(dft_dir)

            self.assertTrue(parsed["restart_needed"])
            self.assertIn("missing final energy", parsed["restart_reason"])

    def test_end_to_end_dry_run_with_mocked_calculator(self) -> None:
        require_pymatgen(self)
        try:
            import numpy as np
            from ase.calculators.calculator import Calculator, all_changes
        except ImportError as exc:
            self.skipTest(str(exc))

        class ZeroCalculator(Calculator):
            implemented_properties = ["energy", "forces"]

            def calculate(self, atoms=None, properties=None, system_changes=all_changes):
                super().calculate(atoms, properties, system_changes)
                self.results = {"energy": 0.0, "forces": np.zeros((len(atoms), 3))}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "snb"
            (source / "candidate_a").mkdir(parents=True)
            (source / "candidate_b").mkdir()
            (source / "candidate_a" / "POSCAR").write_text(H_POSCAR)
            (source / "candidate_b" / "POSCAR").write_text(H_POSCAR)
            import_snb_inputs(snb_dir=source, results_root=root / "results", case_name="dry")

            with patch("mlip_phonons.get_calc.get_calc_object", return_value=ZeroCalculator()):
                relax_model(
                    model_name="fake-model",
                    results_root=root / "results",
                    case_name="dry",
                    models_root=root,
                    device="cpu",
                    include_vdw=False,
                    max_steps=5,
                )
            cluster_model(model_name="fake-model", results_root=root / "results", case_name="dry")
            selected = select_clusters(results_root=root / "results", case_name="dry", model_name="fake-model")
            report = write_report(results_root=root / "results", case_name="dry")

            self.assertEqual(len(read_csv_rows(selected)), 1)
            self.assertTrue(report.exists())


if __name__ == "__main__":
    unittest.main()

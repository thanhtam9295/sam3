"""Done-when audit: every "Done when" criterion of PROMPT.md, PROMPT_ADDON.md, PROMPT_ILASTIK.md
and PROMPT_PLATFORM.md, the tests that prove it, and its status.

    python scripts/done_when_audit.py            # run the tests, write docs/done_when.md (+ .json)
    python scripts/done_when_audit.py --no-run   # write the table from the mapping only

Status:
- ``met``: proven by automated tests on synthetic or bundled reference data.
- ``met (synthetic)``: proven on synthetic data; the criterion also asks for real data, which
  has not been provided yet (the test or tool for it is ready).
- ``waiting for data``: the criterion is about the lab's files; a test or tool is ready and
  skips until they are provided.
- ``open``: not demonstrated (the reason is given).

Run with the test environment of ``scripts/ci.sh`` (xvfb for GUI tests; PostgreSQL tests run
when ``HCS_TEST_POSTGRES_URL`` is set).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Criterion:
    phase: str
    text: str
    tests: list[str]
    status: str  # met | met (synthetic) | waiting for data | open
    note: str = ""
    needs: str = ""  # what we need from the lab
    outcomes: dict[str, str] = field(default_factory=dict)


CRITERIA = [
    # ---------------------------------------------------------------- PROMPT.md §9, §10
    Criterion(
        "0",
        "App launches; pytest passes; GPU status shown in the status bar",
        [
            "tests/test_main_window.py::test_window_has_docks_and_gpu_status",
            "tests/test_main_window.py::test_test_image_is_shown_in_napari_with_micrometre_scale",
            "tests/test_hardening.py::test_selftest_passes",
        ],
        "met",
    ),
    Criterion(
        "1",
        "One real plate from each instrument opens correctly; metadata matches the vendor software",
        [
            "tests/test_readers.py::test_operetta_plate",
            "tests/test_readers.py::test_imagexpress_plate",
            "tests/test_readers.py::test_nikon_single_file_with_well_positions",
            "tests/test_readers.py::test_ome_zarr_plate",
            "tests/test_gui_browser.py::test_open_operetta_plate_and_view_a_field",
        ],
        "met (synthetic)",
        "Readers pass on synthetic vendor layouts (pixel values encode "
        "well/field/channel/z). No public .nd2 sample was reachable.",
        "One real plate export per instrument (Nikon .nd2, Operetta, ImageXpress).",
    ),
    Criterion(
        "3",
        "A bead stack shows reduced FWHM vs raw; runtime per 2048×2048×30 stack is reported",
        [
            "tests/test_deconvolution.py::test_bead_stack_fwhm_is_reduced_in_3d",
            "tests/test_deconvolution.py::test_benchmark_command",
        ],
        "met (synthetic)",
        "Synthetic bead stack; `hcs-studio benchmark` reports runtime.",
        "A real bead PSF stack (with the Nikon Z-stack).",
    ),
    Criterion(
        "5",
        "Colocalization tests on synthetic images give expected values",
        [
            "tests/test_measurements.py::test_identical_and_anticorrelated_channels",
            "tests/test_measurements.py::test_manders_known_overlap",
            "tests/test_measurements.py::test_costes_threshold_matches_brute_force_and_separates_background",
            "tests/test_measurements.py::test_randomization_separates_real_from_chance_correlation",
        ],
        "met",
    ),
    Criterion(
        "5",
        "Coloc results match Fiji Coloc 2 on 3 real fields within tolerance "
        "(§10: Pearson/Manders within ±0.02)",
        ["tests/test_acceptance.py::test_coloc_against_coloc2_log"],
        "waiting for data",
        "`hcs-studio acceptance coloc` compares with a Coloc 2 log.",
        "3 real fields with their Fiji Coloc 2 results.",
    ),
    Criterion(
        "10",
        "Hardening: 384-well performance profiling, memory limits, error handling, user docs",
        [
            "tests/test_hardening.py::test_run_records_timing_and_performance",
            "tests/test_hardening.py::test_memory_limit_caps_workers",
            "tests/test_hardening.py::test_disk_full_stops_the_run_and_it_resumes",
            "tests/test_hardening.py::test_gui_error_handler_logs_and_keeps_running",
            "tests/test_hardening.py::test_user_guide_builds_as_html",
        ],
        "met",
        "384-well timing numbers are in docs/validation.md (scripts/pipelines/perf_384.yaml).",
    ),
    Criterion(
        "10",
        "Windows installer",
        [
            "tests/test_packaging.py",
            "tests/test_encoding.py",
            "tests/test_hardening.py::test_viewer_without_opengl_says_what_to_do",
        ],
        "met",
        "hcs-studio-windows workflow (2026-11-25, run 9): Windows tests 599 passed / 0 failed; "
        "the installer builds, installs silently and the installed copy passes its self-test "
        "(10/10). The hosted runner has no OpenGL driver, so the window was only checked to fail "
        "cleanly there (graphics message, exit code 5).",
        "Open the installed app on a lab PC with a graphics driver (and over Remote Desktop).",
    ),
    Criterion(
        "§10",
        "Full pipeline on a 384-well plate completes unattended, resumes after a "
        "forced kill, and produces the output folder",
        [
            "tests/test_batch.py::test_forced_kill_then_cli_resume",
            "tests/test_ilastik_done_when.py::test_384_well_projected_plate_unattended_with_kill_and_resume",
        ],
        "met (synthetic)",
        "",
        "A real 384-well plate.",
    ),
    Criterion(
        "§10",
        "Cell counts per well within ±5% of manual counts on 10 annotated fields",
        ["tests/test_acceptance.py"],
        "waiting for data",
        "`hcs-studio acceptance counts` (tested on synthetic fields).",
        "10 annotated fields with manual counts.",
    ),
    Criterion(
        "§10",
        "Clicking any well in index.html opens its enhanced previews offline",
        ["tests/test_results.py::test_report_works_offline_in_a_browser"],
        "met",
    ),
    # ---------------------------------------------------------------- PROMPT_ADDON.md
    Criterion(
        "11",
        "6×6 combo and 10-point DRC layouts built in under 2 minutes with the helpers",
        ["tests/test_gui_layout.py::test_combo_and_drc_layouts_are_quick_to_build"],
        "met",
    ),
    Criterion(
        "11",
        "CSV import → export → import round trip gives an identical layout",
        ["tests/test_annotation.py::test_round_trip_csv_xlsx_platemap_json"],
        "met",
    ),
    Criterion(
        "11",
        "Validation catches each seeded error in a test layout",
        ["tests/test_annotation.py::test_validation_catches_each_seeded_error"],
        "met",
    ),
    Criterion(
        "12",
        "MIP equals the known maximum; background removal keeps spot intensity within ±5%",
        [
            "tests/test_projection_background.py::test_projections_equal_known_values",
            "tests/test_projection_background.py::test_background_keeps_spot_intensity_within_5_percent",
        ],
        "met",
    ),
    Criterion(
        "12",
        "A 384-well 3-channel 20-plane plate projects in the batch queue with resume",
        [
            "tests/test_projection_batch.py::test_384_well_3_channel_20_plane_plate_projects_with_resume"
        ],
        "met",
    ),
    Criterion(
        "13",
        "Fits match GraphPad Prism 4PL (IC50 within 5%, Hill within 0.1) on 5 curves",
        ["tests/test_pharmacology.py::test_matches_graphpad_prism"],
        "waiting for data",
        "Skips while tests/data/pharmacology/prism/ is empty; known-answer synthetic fits "
        "pass (test_regression).",
        "5 curves with their Prism fits.",
    ),
    Criterion(
        "13",
        "Bliss/HSA/Loewe/ZIP match SynergyFinder on a reference 6×6 matrix",
        [
            "tests/test_pharmacology.py::test_matches_synergyfinder",
            "tests/test_pharmacology.py::test_bliss_additive_matrix_scores_zero",
            "tests/test_pharmacology.py::test_sham_combination_scores_zero_with_loewe",
        ],
        "waiting for data",
        "Known-answer matrices pass; the SynergyFinder comparison skips "
        "while tests/data/pharmacology/synergyfinder/ is empty.",
        "A reference 6×6 matrix with SynergyFinder scores.",
    ),
    Criterion(
        "13",
        "Full report from a 384-well combo plate in under a minute",
        [
            "tests/test_pharmacology.py::test_full_report_from_a_384_well_combo_plate_in_under_a_minute"
        ],
        "met",
    ),
    Criterion(
        "14",
        "2×6 linked comparison exports SVG/PNG with correct scale bars; the sidecar "
        "regenerates a pixel-identical PNG",
        [
            "tests/test_compare.py::test_2x6_linked_comparison_png_scale_bars",
            "tests/test_compare.py::test_2x6_comparison_svg_scale_bars_and_text",
            "tests/test_compare.py::test_sidecar_regenerates_a_pixel_identical_png",
        ],
        "met",
    ),
    Criterion(
        "15",
        "Synthetic 384-well regression suite runs end to end in CI (known "
        "dose-response, additive and synergistic combinations)",
        ["tests/test_regression.py"],
        "met",
    ),
    # ---------------------------------------------------------------- PROMPT_ILASTIK.md
    Criterion(
        "16",
        "A 384-well projected plate runs through ilastik → objects → analyzer "
        "unattended, resumes after a forced kill, run-folder structure",
        ["tests/test_ilastik_done_when.py"],
        "met (synthetic)",
        "Stand-in ilastik (real command line); 24 wells in CI, 384 with HCS_SLOW=1.",
        "Trained .ilp projects with matching images; a PC with ilastik.",
    ),
    Criterion(
        "16",
        "HeadlessRunner and FijiRunner results agree on the sample",
        [
            "tests/test_ilastik_fiji.py::test_headless_and_fiji_agree",
            "tests/test_ilastik_fiji.py::test_pipeline_with_the_fiji_runner_matches_headless",
            "tests/test_ilastik_fiji.py::test_real_headless_vs_fiji_parity",
        ],
        "met (synthetic)",
        "Stand-in Fiji runs the generated macros as ilastik4ij does; the "
        "real parity test skips without HCS_ILASTIK/HCS_FIJI/HCS_ILASTIK_PIXEL_PROJECT.",
        "Run `hcs-studio ilastik --test <project.ilp>` on a PC with ilastik and Fiji.",
    ),
    Criterion(
        "16",
        "Object Classification on Cellpose masks gives per-well class fractions on "
        "the plate map and in dose-response curves",
        ["tests/test_ilastik_objects.py::test_classifies_existing_objects_per_well"],
        "met (synthetic)",
    ),
    Criterion(
        "16",
        "run_manifest.json records the .ilp hash, ilastik version and runner",
        ["tests/test_ilastik_steps.py::test_pipeline_runs_ilastik_once_per_well"],
        "met",
    ),
    # ---------------------------------------------------------------- PROMPT_PLATFORM.md
    Criterion(
        "17",
        "The scanner indexes every run folder; re-running adds nothing",
        ["tests/test_registry.py::test_scan_indexes_every_run_and_rescanning_adds_nothing"],
        "met",
    ),
    Criterion(
        "17",
        "All IC50s for compound X in cell line Y across ≥ 3 runs",
        ["tests/test_registry.py::test_all_ic50s_of_a_compound_in_a_cell_line_across_runs"],
        "met",
    ),
    Criterion(
        "17",
        "B-score removes a row/column artifact keeping seeded hits; blocked on "
        "dose-response layouts",
        [
            "tests/test_normalization.py::test_b_score_removes_a_row_column_artifact_and_keeps_seeded_hits",
            "tests/test_normalization.py::test_b_score_is_blocked_on_dose_response_and_combination_layouts",
        ],
        "met",
    ),
    Criterion(
        "17",
        "Control charts flag a seeded drift; MSR matches a hand calculation",
        [
            "tests/test_monitoring.py::test_control_chart_flags_a_seeded_drift_and_not_the_baseline",
            "tests/test_monitoring.py::test_msr_matches_a_hand_calculation",
        ],
        "met",
        "MSR formula from Eastwood 2006; not yet checked against the NCATS AGM.",
        "Confirm the MSR formula you use.",
    ),
    Criterion(
        "18",
        "Each artifact class detected with ≥ 95% sensitivity and ≤ 5% false "
        "positives on clean fields",
        ["tests/test_qc.py::test_each_artifact_class_is_detected_with_few_false_positives"],
        "met (synthetic)",
        "Thresholds are tuned on synthetic artifacts.",
        "20–50 bad and ~50 good real fields per instrument.",
    ),
    Criterion(
        "18",
        "QC on a 384-well plate runs in the queue and finishes before segmentation",
        [
            "tests/test_qc_runs.py::test_qc_of_a_384_well_plate_runs_in_the_queue_before_segmentation"
        ],
        "met",
    ),
    Criterion(
        "18",
        "Excluded wells are removed from IC50 fits and listed in the report",
        ["tests/test_qc_runs.py::test_wells_failing_image_qc_leave_ic50_fits_and_are_listed"],
        "met",
    ),
    Criterion(
        "19",
        "A dropped export or a posted event triggers QC → analysis → report → "
        "notification with no clicks",
        ["tests/test_automation_flow.py::test_dropped_export_runs_without_clicks"],
        "met (synthetic)",
        "Synthetic ImageXpress exports, in-test SMTP server.",
        "Real exports per instrument, share paths, SMTP host and sender.",
    ),
    Criterion(
        "19",
        "A slow copy is not processed early; a duplicate event starts no second run",
        [
            "tests/test_automation.py::test_slow_copy_is_not_started_early",
            "tests/test_automation.py::test_duplicates_and_reimaging",
        ],
        "met",
    ),
    Criterion(
        "19",
        "A plate failing QC produces a schema-valid reimage.request",
        ["tests/test_automation.py::test_duplicates_and_reimaging"],
        "met",
    ),
    Criterion(
        "20",
        "Synthetic known answers: G1/G2 ratio within 2 pp; foci ±1 per nucleus; N/C "
        "ratio within 5% (plus apoptosis and proliferation)",
        [
            "tests/test_cell_cycle.py::test_dean_jett_fox_fit_recovers_known_phase_fractions",
            "tests/test_foci.py::test_foci_counts_per_nucleus_from_images",
            "tests/test_translocation.py::test_nc_ratio_per_cell_from_images",
            "tests/test_apoptosis.py::test_apoptosis_from_images",
            "tests/test_proliferation.py::test_markers_micronuclei_and_multinucleation_from_images",
        ],
        "met",
    ),
    Criterion(
        "20",
        "On a real control plate thresholds calibrate and positive controls "
        "separate from vehicle, with Z′",
        ["tests/test_apoptosis.py::test_categories_viable_counts_and_gr_match_truth"],
        "waiting for data",
        "Calibration and Z′ are tested on synthetic control plates.",
        "One real control plate per assay with your current results.",
    ),
    Criterion(
        "21",
        "Synthetic spheroids: volume within ±5%, core fraction within ±5 pp (invasion recovered)",
        [
            "tests/test_spheroids.py::test_volume_and_qc_flags_from_brightfield",
            "tests/test_spheroids.py::test_core_viability_and_invasion_from_images",
            "tests/test_spheroids.py::test_confocal_3d_volume_core_viability_and_coverage",
        ],
        "met",
    ),
    Criterion(
        "21",
        "A 384-well ULA plate runs in batch with galleries and a volume dose-response",
        ["tests/test_spheroids.py::test_384_plate_batch_and_dose_response_on_volume"],
        "met (synthetic)",
        "",
        "A brightfield and a confocal spheroid plate with treatment.",
    ),
    Criterion(
        "22",
        "Doubling time recovered within 5% on a synthetic time series",
        [
            "tests/test_kinetics_analysis.py::test_growth_models_recover_doubling_time",
            "tests/test_kinetics_analysis.py::test_doubling_time_from_images",
        ],
        "met",
    ),
    Criterion(
        "22",
        "IC50(t) and GR over time computed and plotted for a real kinetic plate, "
        "timestamps checked against instrument metadata",
        [
            "tests/test_kinetics_analysis.py::test_ic50_and_gr_over_time_known_answers",
            "tests/test_kinetics.py::test_imagexpress_time_points_from_image_timestamps",
            "tests/test_kinetics.py::test_nd2_time_points_from_frame_events",
        ],
        "met (synthetic)",
        "GR interval convention (Hafner et al.) not checked: the paper was not readable from here.",
        "A kinetic plate per instrument with the treatment time; confirm the GR convention.",
    ),
    Criterion(
        "23",
        "≥ 95% of seeded hits recovered with B-score; the positional artifact makes no false hits",
        ["tests/test_screening.py::test_seeded_hits_recovered_and_artifact_makes_no_false_hits"],
        "met",
    ),
    Criterion(
        "23",
        "Pick list matches a sample Echo file; confirmation layouts pre-registered",
        [
            "tests/test_cherrypick.py::test_cherry_pick_command_registers_and_sends",
            "tests/test_cherrypick.py::test_pick_list_formats",
        ],
        "waiting for data",
        "Layouts are pre-registered (tested); the Echo columns are configurable but unverified.",
        "A sample Echo pick list, the library plate map, your hit rules.",
    ),
    Criterion(
        "24",
        "Reproduces pycytominer's outputs on its own example data",
        [
            "tests/test_cellpainting_profiles.py::test_reproduces_pycytominer_on_jump_data",
            "tests/test_cellpainting_profiles.py::test_whole_chain_on_jump_data",
            "tests/test_cellpainting.py::test_cellprofiler_sqlite_matches_pycytominer",
        ],
        "met",
    ),
    Criterion(
        "24",
        "Seeded phenotypes active (mAP p < 0.05), vehicle wells not",
        [
            "tests/test_cellpainting_evaluation.py::test_seeded_phenotypes_active_vehicles_not",
            "tests/test_cellpainting_evaluation.py::test_painted_plate_end_to_end",
        ],
        "met (synthetic)",
        "",
        "A Cell Painting plate with reference compounds and MoA.",
    ),
    Criterion(
        "25",
        "Fine-tuning on ~20 annotated fields improves held-out AP@0.5 vs the base "
        "model; the comparison report is saved",
        [
            "tests/test_model_registry.py::test_tuning_improves_held_out_ap",
            "tests/test_model_qa.py::test_comparison_report_is_saved",
            "tests/test_model_registry.py::test_real_cellpose_fine_tune_registers",
        ],
        "open",
        "Shown for tuned classical segmentation (AP gain > 0.15). Cellpose "
        "fine-tuning runs end to end with the real cpsam_v2 weights (opt-in test passed on CPU, 1 epoch, 2026-11-25), but its "
        "accuracy gain needs a GPU and ~100 epochs: not demonstrated here.",
        "20–40 annotated fields of the hardest cell line, and a GPU run.",
    ),
    Criterion(
        "25",
        "A pinned model version reproduces identical masks; changing it is "
        "recorded in the manifest",
        [
            "tests/test_model_registry.py::test_pinned_model_reproduces_masks_and_manifest_records_it"
        ],
        "met",
    ),
    Criterion(
        "26",
        "Every action of a scripted scenario is in the audit log with user and "
        "reason; a locked run can't change; re-analysis makes a linked version",
        ["tests/test_signoff.py::test_scripted_scenario_is_fully_audited"],
        "met",
    ),
    Criterion(
        "26",
        "Editing an audit row directly in the database is detected",
        [
            "tests/test_security.py::test_chain_verifies_and_detects_tampering",
            "tests/test_security.py::test_checkpoints_catch_a_rewritten_chain",
        ],
        "met",
    ),
    Criterion(
        "27",
        "Two clients submit jobs at once and both complete; a worker killed mid-job "
        "is restarted and the job resumes",
        [
            "tests/test_server_queue.py::test_two_clients_jobs_complete",
            "tests/test_server_api.py::test_two_clients_at_once",
            "tests/test_server_queue.py::test_killed_worker_is_restarted_and_job_resumes",
        ],
        "met",
    ),
    Criterion(
        "27",
        "A 384-well plate is processed on the server while the laptop is closed; "
        "results visible in web and desktop clients",
        [
            "tests/test_server_e2e.py::test_384_well_plate_while_the_client_is_away",
            "tests/test_server_e2e.py::test_portal_in_a_browser",
        ],
        "met",
    ),
    Criterion(
        "28",
        "≥ 90% of evaluation answers match SQL ground truth; every answer shows its "
        "supporting queries",
        [
            "tests/test_assistant_eval.py::test_ground_truth_and_grading_with_an_oracle",
            "tests/test_assistant.py::test_question_with_a_query_shows_its_work",
            "tests/test_assistant_eval.py::test_live_evaluation",
        ],
        "waiting for data",
        "Grading and 'show work' are tested with a mocked model; the "
        "live evaluation needs an API key.",
        "Run `hcs-studio assistant eval --workdir <new>` with your key.",
    ),
    Criterion(
        "28",
        "A payload log proves nothing beyond the curated views leaves the app",
        [
            "tests/test_assistant.py::test_payload_log_contains_only_curated_data",
            "tests/test_assistant.py::test_snapshot_holds_only_curated_views",
        ],
        "met",
    ),
]


def run_tests(nodes: list[str]) -> dict[str, str]:
    """Outcome per test id (passed / failed / skipped)."""
    with tempfile.TemporaryDirectory() as tmp:
        report = Path(tmp) / "junit.xml"
        subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rs",
             f"--junitxml={report}", *sorted(set(nodes))],
            cwd=ROOT, check=False,
        )  # fmt: skip
        tree = ET.parse(report)
    out: dict[str, str] = {}
    for case in tree.iter("testcase"):
        module = case.get("classname", "").replace(".", "/") + ".py"
        name = case.get("name", "").split("[")[0]
        node = f"{module}::{name}"
        status = "passed"
        if case.find("failure") is not None or case.find("error") is not None:
            status = "failed"
        elif case.find("skipped") is not None:
            status = "skipped"
        previous = out.get(node)
        out[node] = "failed" if "failed" in (status, previous) else status
    return out


def outcome_of(test: str, outcomes: dict[str, str]) -> str:
    if "::" in test:
        return outcomes.get(test, "not run")
    found = [v for k, v in outcomes.items() if k.startswith(test + "::")]
    if not found:
        return "not run"
    if "failed" in found:
        return "failed"
    return "passed" if "passed" in found else "skipped"


def write(criteria: list[Criterion], ran: bool) -> None:
    order = ["met", "met (synthetic)", "waiting for data", "open"]
    counts = {s: sum(c.status == s for c in criteria) for s in order}
    failed = [c for c in criteria if "failed" in c.outcomes.values()]
    lines = [
        "# Done-when audit",
        "",
        "Every *Done when* criterion of the four specifications, the tests that prove it and its "
        "status. Generated by `python scripts/done_when_audit.py` (re-run it after changes).",
        "",
        f"**{len(criteria)} criteria:** "
        + ", ".join(f"{counts[s]} {s}" for s in order)
        + (f". **Test failures: {len(failed)}**." if ran and failed else "."),
        "",
        "Status: *met* = proven by automated tests on synthetic or bundled reference data; "
        "*met (synthetic)* = proven on synthetic data, the real-data part waits for your files; "
        "*waiting for data* = a test or tool is ready and skips until your files arrive; "
        "*open* = not demonstrated (reason given).",
        "",
        "| Phase | Criterion | Status | Tests | Notes | Needed from the lab |",
        "|---|---|---|---|---|---|",
    ]
    for c in criteria:
        tests = "<br>".join(
            f"`{t.removeprefix('tests/')}`" + (f" — {c.outcomes[t]}" if ran else "")
            for t in c.tests
        )
        lines.append(f"| {c.phase} | {c.text} | {c.status} | {tests} | {c.note} | {c.needs} |")
    lines.append("")
    (ROOT / "docs" / "done_when.md").write_text("\n".join(lines), encoding="utf-8")
    (ROOT / "docs" / "done_when.json").write_text(
        json.dumps([asdict(c) for c in criteria], indent=1), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-run", action="store_true", help="write the table without testing")
    args = parser.parse_args()
    ran = not args.no_run
    if ran:
        outcomes = run_tests([t for c in CRITERIA for t in c.tests])
        for c in CRITERIA:
            c.outcomes = {t: outcome_of(t, outcomes) for t in c.tests}
    write(CRITERIA, ran)
    failed = [c for c in CRITERIA if "failed" in c.outcomes.values()]
    print(f"{len(CRITERIA)} criteria; {len(failed)} with failing tests; docs/done_when.md written")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

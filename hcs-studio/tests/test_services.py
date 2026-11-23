"""Service layer (PROMPT_PLATFORM.md §2.4): actors, reasons, events, and that the GUI and the
command line change state only through services."""

from __future__ import annotations

import ast
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from hcs_studio import services
from hcs_studio.services import Actor, Event, ReasonRequired, context, recent_events, service

SRC = Path(__file__).resolve().parent.parent / "src" / "hcs_studio"


@pytest.fixture
def registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    from hcs_studio.db.engine import dispose_engines

    monkeypatch.setenv("HCS_STUDIO_HOME", str(tmp_path))
    monkeypatch.delenv("HCS_DATABASE_URL", raising=False)
    dispose_engines()
    yield tmp_path
    dispose_engines()


@service("test.change", object_type="thing", reason_required=lambda ctx, x, **kw: x < 0)
def change(ctx: services.ServiceContext, x: int) -> int:
    ctx.note(object_id=f"thing-{x}", before={"x": None}, after={"x": x})
    if x == 13:
        raise RuntimeError("unlucky")
    return x * 2


@service("test.outer")
def outer(ctx: services.ServiceContext) -> int:
    ctx.note(object_id="outer")
    inner = change(ctx, 2)  # a nested service records its own event
    return inner


def test_events_record_actor_reason_object_and_failures(registry: Path) -> None:
    ctx = context("calibration", actor=Actor("alice", "Alice A.", "account"))
    assert change(ctx, 4) == 8
    with pytest.raises(ReasonRequired, match="give a reason"):
        change(context(), -1)  # negative values need a reason: nothing runs, nothing recorded
    assert change(context("checked by hand"), -1) == -2
    with pytest.raises(RuntimeError):
        change(context(), 13)
    assert outer(context()) == 4
    events = recent_events(10)
    assert [e["type"] for e in events] == [
        "test.outer",
        "test.change",
        "test.change",
        "test.change",
        "test.change",
    ]
    first = events[-1]
    assert first["actor"] == "alice" and first["reason"] == "calibration"
    assert first["object_type"] == "thing" and first["object_id"] == "thing-4"
    assert first["before"] == {"x": None} and first["after"] == {"x": 4}
    failed = events[2]
    assert failed["status"] == "failed" and "unlucky" in failed["error"]
    assert events[0]["object_id"] == "outer" and events[1]["object_id"] == "thing-2"
    with pytest.raises(TypeError, match="ServiceContext"):
        change(4)  # type: ignore[arg-type, call-arg]


def test_a_failing_sink_never_breaks_the_change(registry: Path) -> None:
    seen: list[Event] = []

    def broken(event: Event) -> None:
        raise OSError("disk full")

    services.BUS.add(broken)
    services.BUS.add(seen.append)
    try:
        assert change(context(), 1) == 2
    finally:
        services.BUS.remove(broken)
        services.BUS.remove(seen.append)
    assert [e.type for e in seen] == ["test.change"]


def test_layout_and_annotation_services_are_recorded(registry: Path, tmp_path: Path) -> None:
    from hcs_studio.io.layout_store import LayoutStore
    from hcs_studio.pharmacology.synthetic import layout
    from hcs_studio.services.operations import annotate_run, save_layout_version
    from tests.compare_builders import make_run

    store = LayoutStore("PLATE-1", root=tmp_path / "layouts")
    saved = save_layout_version(context("first version"), store, layout())
    again = save_layout_version(context(), store, saved)
    assert again.version == 2
    event = recent_events(1, "layout.save")[0]
    assert event["object_id"] == "PLATE-1/v0002" and event["before"] == {"version": 1}
    run = make_run(tmp_path / "run")
    new = layout()
    annotate_run(context("new doses from the Echo file"), run, new)
    event = recent_events(1, "run.annotate")[0]
    assert event["reason"] == "new doses from the Echo file" and event["status"] == "ok"
    assert event["object_id"] == f"ABSOLUTE:{run}" and event["after"]["wells"] == len(new.wells)
    assert event["before"] is not None and event["before"]["sha256"] != event["after"]["sha256"]


def test_cli_records_reason_and_refuses_exclusions_without_one(
    registry: Path, tmp_path: Path
) -> None:
    from hcs_studio.pharmacology.synthetic import write_run

    run = write_run(tmp_path / "run")
    env = {"HCS_STUDIO_HOME": str(registry), "PATH": "/usr/bin:/bin"}
    cli = [sys.executable, "-m", "hcs_studio", "pharm", str(run), "--readout", "nuclei_count"]
    cli += ["--normalize", "percent_of_control", "--no-pdf", "--no-figures", "--exclude", "P:B03"]
    refused = subprocess.run(cli, capture_output=True, text=True, env=env, timeout=300)
    assert refused.returncode == 2 and "give a reason" in refused.stdout
    done = subprocess.run(
        [*cli, "--reason", "bubble in B03"], capture_output=True, text=True, env=env, timeout=300
    )
    assert done.returncode == 0, done.stdout + done.stderr
    event = recent_events(5, "pharmacology.write")
    assert event[0]["status"] == "ok" and event[0]["reason"] == "bubble in B03"
    assert event[0]["after"]["excluded_wells"] == [["P", "B03"]]


# Core functions that change state; the GUI and the CLI must call the services instead.
WRAPPED = {
    "hcs_studio.pipeline.executor": {"run_pipeline", "analyze_run", "annotate_run"},
    "hcs_studio.output.report": {"write_report"},
    "hcs_studio.analysis.classifier": {"save_classifier"},
    "hcs_studio.segmentation.cellpose_backend": {"fine_tune"},
    "hcs_studio.analysis.profiling": {"run_profiling"},
    "hcs_studio.qc.run": {"run_qc"},
    "hcs_studio.qc.review": {"apply_override"},
    "hcs_studio.screening.screen": {"run_screen", "add_override"},
    "hcs_studio.db.library": {"import_library"},
}
ALLOWED = {
    # read-only exploration (write=False) is not a change
    ("hcs_studio/gui/pharmacology_panel.py", "run_pharmacology"),
}


def test_gui_and_cli_change_state_only_through_services() -> None:
    offenders = []
    files = [SRC / "app.py", *sorted((SRC / "gui").glob("*.py"))]
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        rel = str(path.relative_to(SRC.parent))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in WRAPPED:
                for alias in node.names:
                    if alias.name in WRAPPED[node.module] and (rel, alias.name) not in ALLOWED:
                        offenders.append(f"{rel}:{node.lineno} imports {alias.name}")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                target = ast.unparse(node.func)
                if target.endswith(("store.save", "settings.save")) or target in (
                    "Sam3Settings.save",
                    "IlastikSettings.save",
                ):
                    offenders.append(f"{rel}:{node.lineno} calls {target}()")
    assert offenders == [], "\n".join(offenders)

"""The assistant's evaluation set (Phase 28.5): questions with known answers on the synthetic
registry (three runs of the known-answer combination plate, ``synthetic.registry_runs``).

Each item has ground-truth SQL on the curated views. An answer passes when it contains the
expected numbers (allowing rounding and unit conversion: M → µM/nM, fraction → %) or names,
**and** it shows its work (at least one tool call). Done-when: ≥ 90 % pass.

The live evaluation calls the Claude API and costs money: it runs only on request
(``hcs-studio assistant eval`` or ``HCS_ASSISTANT_EVAL=1 pytest tests/test_assistant_eval.py``).
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from hcs_studio.assistant.conversation import Answer, Conversation, Question, _numbers
from hcs_studio.assistant.sql import ReadOnlyDatabase
from hcs_studio.assistant.views import Privacy, Snapshot, build_snapshot

_RUN = (
    "(SELECT u.run_id FROM runs u JOIN plates p ON p.plate_id = u.plate_id WHERE p.barcode = '{}')"
)


def run_of(barcode: str) -> str:
    """SQL for the run ID of a plate barcode (each synthetic plate has one run)."""
    return _RUN.format(barcode)


@dataclass(frozen=True)
class EvalItem:
    id: str
    question: str
    sql: str
    check: Literal["numbers", "names"]
    #: numbers: relative tolerance for a match
    tolerance: float = 0.02


ITEMS: tuple[EvalItem, ...] = (
    EvalItem("plates", "How many plates are in the registry?",
             "SELECT COUNT(*) FROM plates", "numbers"),
    EvalItem("runs", "How many analysis runs are in the registry?",
             "SELECT COUNT(*) FROM runs", "numbers"),
    EvalItem("lines_plate", "Which cell lines are on plate REG-001?",
             "SELECT DISTINCT cell_line FROM wells w JOIN plates p USING (plate_id) "
             "WHERE p.barcode = 'REG-001' AND cell_line <> ''", "names"),
    EvalItem("compounds_plate", "Which compounds were tested on plate REG-002?",
             "SELECT DISTINCT compound_1 FROM wells w JOIN plates p USING (plate_id) "
             "WHERE p.barcode = 'REG-002' AND compound_1 IS NOT NULL", "names"),
    EvalItem("ec50_b", "What is the EC50 of Drug B in MCF7 for the nuclei count on plate "
             "REG-001?",
             f"SELECT ec50_m FROM dose_response WHERE run_id = {run_of('REG-001')} AND "
             "compound = 'Drug B' AND cell_line = 'MCF7' AND readout = 'nuclei_count'",
             "numbers"),
    EvalItem("ec50_mean", "What is the mean EC50 of Drug A in MCF7 (nuclei count) across all "
             "runs?",
             "SELECT AVG(ec50_m) FROM dose_response WHERE compound = 'Drug A' AND "
             "cell_line = 'MCF7' AND readout = 'nuclei_count'", "numbers"),
    EvalItem("most_potent", "Which compound is the most potent (lowest EC50) in MCF7 for the "
             "nuclei count on plate REG-001?",
             f"SELECT compound FROM dose_response WHERE run_id = {run_of('REG-001')} AND "
             "cell_line = 'MCF7' AND readout = 'nuclei_count' ORDER BY ec50_m LIMIT 1",
             "names"),
    EvalItem("zip_run", "Which drug combinations had a ZIP score above 10 on plate REG-001?",
             "SELECT DISTINCT drug_a, drug_b FROM combination_scores WHERE model = 'zip' AND "
             f"mean > 10 AND run_id = {run_of('REG-001')}", "names"),
    EvalItem("zip_all", "Which combinations had a ZIP score above 10 on every plate?",
             "SELECT drug_a, drug_b FROM combination_scores WHERE model = 'zip' GROUP BY "
             "drug_a, drug_b HAVING MIN(mean) > 10", "names"),
    EvalItem("bliss", "What is the mean Bliss synergy score of Drug A + Drug B in MCF7 on "
             "plate REG-002?",
             "SELECT mean FROM combination_scores WHERE model = 'bliss' AND drug_a = 'Drug A' "
             f"AND drug_b = 'Drug B' AND cell_line = 'MCF7' AND run_id = {run_of('REG-002')}",
             "numbers"),
    EvalItem("zprime", "What is the Z′ factor of plate REG-001 for the nuclei count?",
             f"SELECT z_prime FROM plate_qc WHERE run_id = {run_of('REG-001')} AND "
             "readout = 'nuclei_count'", "numbers"),
    EvalItem("best_zprime", "Which plate has the highest Z′ factor for the nuclei count?",
             "SELECT p.barcode FROM plate_qc q JOIN runs u USING (run_id) JOIN plates p ON "
             "p.plate_id = u.plate_id WHERE q.readout = 'nuclei_count' ORDER BY q.z_prime DESC "
             "LIMIT 1", "names"),
    EvalItem("neg_wells", "How many negative-control wells are on plate REG-001?",
             "SELECT COUNT(*) FROM wells JOIN plates p USING (plate_id) WHERE "
             "p.barcode = 'REG-001' AND role = 'negative_control'", "numbers"),
    EvalItem("combo_wells", "How many wells on plate REG-001 contain two compounds?",
             "SELECT COUNT(*) FROM wells JOIN plates p USING (plate_id) WHERE "
             "p.barcode = 'REG-001' AND n_compounds = 2", "numbers"),
    EvalItem("top_conc", "What is the highest concentration of Drug D on plate REG-001?",
             "SELECT MAX(concentration_1_m) FROM wells JOIN plates p USING (plate_id) WHERE "
             "p.barcode = 'REG-001' AND compound_1 = 'Drug D'", "numbers"),
    EvalItem("vehicle_mean", "What is the mean nuclei count of the negative-control wells on "
             "plate REG-001?",
             "SELECT AVG(r.value) FROM well_results r JOIN runs u USING (run_id) JOIN wells w "
             "ON w.plate_id = u.plate_id AND w.well = r.well JOIN plates p ON p.plate_id = "
             "u.plate_id WHERE p.barcode = 'REG-001' AND w.role = 'negative_control' AND "
             "r.readout = 'nuclei_count'", "numbers"),
    EvalItem("hill_c", "What is the Hill slope of Drug C in MCF7 (nuclei count) on plate "
             "REG-001?",
             f"SELECT hill FROM dose_response WHERE run_id = {run_of('REG-001')} AND "
             "compound = 'Drug C' AND cell_line = 'MCF7' AND readout = 'nuclei_count'",
             "numbers", 0.03),
    EvalItem("hela_fits", "How many dose-response fits are there for HeLa cells in the "
             "registry?",
             "SELECT COUNT(*) FROM dose_response WHERE cell_line = 'HeLa'", "numbers"),
    EvalItem("hela_compounds", "Which compounds have dose-response fits in HeLa?",
             "SELECT DISTINCT compound FROM dose_response WHERE cell_line = 'HeLa'", "names"),
    EvalItem("ratio", "On plate REG-001, how many times higher is the EC50 of Drug A in HeLa "
             "than in MCF7 (nuclei count)?",
             "SELECT h.ec50_m / m.ec50_m FROM dose_response h JOIN dose_response m ON "
             "m.run_id = h.run_id AND m.compound = h.compound AND m.readout = h.readout WHERE "
             f"h.run_id = {run_of('REG-001')} AND h.compound = 'Drug A' AND h.cell_line = "
             "'HeLa' AND m.cell_line = 'MCF7' AND h.readout = 'nuclei_count'", "numbers"),
    EvalItem("cv", "What is the CV of the vehicle wells for the nuclei count on plate REG-003?",
             f"SELECT cv_vehicle_pct FROM plate_qc WHERE run_id = {run_of('REG-003')} AND "
             "readout = 'nuclei_count'", "numbers"),
    EvalItem("readouts", "Which readouts were measured in the run of plate REG-001?",
             "SELECT DISTINCT readout FROM well_results WHERE run_id = "
             f"{run_of('REG-001')}", "names"),
    EvalItem("hsa", "What is the mean HSA score of Drug A + Drug C in MCF7, averaged over all "
             "plates?",
             "SELECT AVG(mean) FROM combination_scores WHERE model = 'hsa' AND drug_a = "
             "'Drug A' AND drug_b = 'Drug C' AND cell_line = 'MCF7'", "numbers"),
    EvalItem("max3", "What is the most synergistic 3×3 window ZIP score of Drug A + Drug B on "
             "plate REG-001?",
             "SELECT max_3x3 FROM combination_scores WHERE model = 'zip' AND drug_a = 'Drug A' "
             f"AND drug_b = 'Drug B' AND run_id = {run_of('REG-001')}", "numbers"),
    EvalItem("abs_ic50", "What is the absolute IC50 of Drug D in HeLa (nuclei count) on plate "
             "REG-002?",
             f"SELECT abs_ic50_m FROM dose_response WHERE run_id = {run_of('REG-002')} AND "
             "compound = 'Drug D' AND cell_line = 'HeLa' AND readout = 'nuclei_count'",
             "numbers"),
    EvalItem("date", "On which date was plate REG-002 analysed?",
             f"SELECT substr(started, 1, 10) FROM runs WHERE run_id = {run_of('REG-002')}",
             "names"),
    EvalItem("auc", "What is the area under the curve of Drug B in MCF7 (nuclei count) on "
             "plate REG-003?",
             f"SELECT auc FROM dose_response WHERE run_id = {run_of('REG-003')} AND "
             "compound = 'Drug B' AND cell_line = 'MCF7' AND readout = 'nuclei_count'",
             "numbers"),
    EvalItem("levels", "How many different concentrations of Drug D are on plate REG-001?",
             "SELECT COUNT(DISTINCT concentration_1_m) FROM wells JOIN plates p USING "
             "(plate_id) WHERE p.barcode = 'REG-001' AND compound_1 = 'Drug D'", "numbers"),
    EvalItem("ci", "What is the 95 % confidence interval of the EC50 of Drug B in MCF7 (nuclei "
             "count) on plate REG-001?",
             "SELECT ec50_ci_low_m, ec50_ci_high_m FROM dose_response WHERE run_id = "
             f"{run_of('REG-001')} AND compound = 'Drug B' AND cell_line = 'MCF7' AND "
             "readout = 'nuclei_count'", "numbers", 0.03),
    EvalItem("lowest_run", "On which plate did Drug A in MCF7 have the lowest EC50 (nuclei "
             "count)?",
             "SELECT p.barcode FROM dose_response d JOIN runs u USING (run_id) JOIN plates p "
             "ON p.plate_id = u.plate_id WHERE d.compound = 'Drug A' AND d.cell_line = 'MCF7' "
             "AND d.readout = 'nuclei_count' ORDER BY d.ec50_m LIMIT 1", "names"),
)  # fmt: skip

#: unit factors an answer may use for a value (M → mM/µM/nM/pM, fraction → %)
FACTORS = (1.0, 1e3, 1e6, 1e9, 1e12, 100.0, 0.01)


def expected(item: EvalItem, db: ReadOnlyDatabase) -> list[Any]:
    """The ground truth of ``item`` (flattened values of the SQL result)."""
    result = db.query(item.sql, max_rows=50)
    return [v for row in result.rows for v in row if v is not None]


def number_found(value: float, answer: str, tolerance: float) -> bool:
    for text in _numbers(answer):
        try:
            found = float(text.replace(",", ""))
        except ValueError:
            continue
        for factor in FACTORS:
            target = value * factor
            digits = len(text.replace("-", "").replace(".", "").lstrip("0")) or 1
            if math.isclose(found, target, rel_tol=tolerance, abs_tol=1e-12):
                return True
            # rounded to the digits shown (e.g. 0.3 for 0.297)
            if target and math.isclose(float(f"{target:.{digits}g}"), found, rel_tol=1e-9):
                return True
    return False


def grade(item: EvalItem, truth: list[Any], answer: Answer) -> tuple[bool, str]:
    text = answer.display_text or answer.text
    if not answer.steps:
        return False, "no tool call (the answer shows no work)"
    if not truth:
        return False, "the ground truth is empty"
    if item.check == "numbers":
        missing = [v for v in truth if not number_found(float(v), text, item.tolerance)]
        return (not missing, f"missing {missing}" if missing else "ok")
    names = [str(v) for v in truth]
    missing_names = [n for n in names if n.lower() not in text.lower()]
    return (not missing_names, f"missing {missing_names}" if missing_names else "ok")


@dataclass
class EvalResult:
    id: str
    question: str
    expected: list[Any]
    answer: str
    passed: bool
    note: str
    tools: list[str]
    sql: list[str]
    tokens: dict[str, int]
    seconds: float


@dataclass
class EvalReport:
    model: str
    results: list[EvalResult] = field(default_factory=list)

    @property
    def score(self) -> float:
        return sum(r.passed for r in self.results) / len(self.results) if self.results else 0.0

    @property
    def all_show_work(self) -> bool:
        return all(r.tools for r in self.results)

    def to_json(self) -> dict[str, Any]:
        tokens: dict[str, int] = {}
        for r in self.results:
            for k, v in r.tokens.items():
                tokens[k] = tokens.get(k, 0) + v
        return {
            "model": self.model,
            "score": self.score,
            "passed": sum(r.passed for r in self.results),
            "items": len(self.results),
            "all_show_work": self.all_show_work,
            "tokens": tokens,
            "results": [asdict(r) for r in self.results],
        }

    def write(self, path: Path) -> Path:
        path = Path(path)
        path.write_text(json.dumps(self.to_json(), indent=2, default=str), encoding="utf-8")
        return path


def build_eval_registry(folder: Path, n_runs: int = 3) -> list[Path]:
    """The synthetic registry of the evaluation (in the current ``HCS_STUDIO_HOME``)."""
    from hcs_studio.db import session_scope
    from hcs_studio.db.scanner import scan
    from hcs_studio.synthetic import registry_runs

    runs = registry_runs(Path(folder) / "runs", n_runs=n_runs)
    with session_scope() as session:
        scan(session, [Path(folder) / "runs"])
    return runs


def run_evaluation(
    client: Any,
    settings: Any,
    workdir: Path,
    *,
    items: tuple[EvalItem, ...] = ITEMS,
    strong: bool = False,
    log: Callable[[str], None] = print,
) -> EvalReport:
    """Ask every item in a fresh conversation (no usage counted, no audit events) and grade
    the answers against the ground truth."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    snapshot: Snapshot = build_snapshot(workdir / "eval_views.sqlite", Privacy())
    truth_db = ReadOnlyDatabase(snapshot.path, max_rows=50, timeout_s=30)
    report = EvalReport(model=settings.strong_model if strong else settings.model)
    try:
        for item in items:
            truth = expected(item, truth_db)
            conversation = Conversation(client, snapshot, settings, count_usage=False)
            started = time.monotonic()
            try:
                answer = conversation.ask(Question(item.question, strong=strong))
            except Exception as exc:  # an API error fails this item, not the evaluation
                answer = Answer(conversation.id, "error", message=f"{type(exc).__name__}: {exc}")
            finally:
                conversation.close()
            passed, note = grade(item, truth, answer)
            if answer.status != "answered":
                passed, note = False, f"{answer.status}: {answer.message}"
            result = EvalResult(
                id=item.id,
                question=item.question,
                expected=truth,
                answer=answer.display_text or answer.text,
                passed=passed,
                note=note,
                tools=[s.tool for s in answer.steps],
                sql=[q for s in answer.steps for q in s.sql],
                tokens=answer.usage,
                seconds=round(time.monotonic() - started, 1),
            )
            report.results.append(result)
            log(f"{'PASS' if passed else 'FAIL'} {item.id}: {note}")
    finally:
        truth_db.close()
    return report

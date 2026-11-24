"""The assistant's tools (Phase 28.2). All read-only: they read the curated snapshot (and, for
the manifest, QC and thumbnail tools, scrubbed parts of run folders), or return UI actions and
proposals that the user applies. **No tool changes pipelines, layouts, exclusions or
approvals.**

Each call returns a :class:`ToolOutcome`: the content sent back to Claude, plus what the UI
needs to "show work" (SQL, a summary) and to act (highlights, charts, proposals).
"""

from __future__ import annotations

import base64
import io
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hcs_studio.assistant.sql import QueryRejected, QueryResult, QueryTimeout, ReadOnlyDatabase
from hcs_studio.assistant.views import Snapshot, schema_json, scrub

#: tools that send image pixels; never offered unless enabled for the question
IMAGE_TOOLS = ("get_image_thumbnail",)
MAX_CHART_ROWS = 5000
THUMBNAIL_PX = 256


class ToolError(ValueError):
    """A tool call that cannot run (bad input, missing data); sent back as an error result."""


class NeedsConfirmation(Exception):
    """The call sends image data and the user has not confirmed it yet."""

    def __init__(self, request: dict[str, Any]) -> None:
        super().__init__("confirmation needed")
        self.request = request


@dataclass
class ToolOutcome:
    content: str | list[dict[str, Any]]
    summary: str = ""
    sql: list[str] = field(default_factory=list)
    ui: list[dict[str, Any]] = field(default_factory=list)
    is_error: bool = False


def _nullable(kind: str, description: str) -> dict[str, Any]:
    return {"type": [kind, "null"], "description": description}


def _schema(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "get_schema",
        "description": "The curated views you can query with run_query: names, columns, types "
        "and meanings. The same documentation is in the system prompt; call this only to check.",
        "input_schema": _schema({}),
    },
    {
        "name": "run_query",
        "description": "Run one read-only SQLite SELECT on the curated views (no other tables). "
        "Results are capped (see row_count/truncated); aggregate in SQL instead of reading many "
        "rows. Every number in your answer must come from a tool result.",
        "input_schema": _schema(
            {
                "sql": {"type": "string", "description": "one SELECT (or WITH … SELECT)"},
                "purpose": {"type": "string", "description": "what the query answers, briefly"},
            }
        ),
    },
    {
        "name": "get_plate_layout",
        "description": "A plate's layout: counts by role, cell lines, compounds with their "
        "concentration ranges, and every well.",
        "input_schema": _schema(
            {"plate_id": {"type": "integer", "description": "plates.plate_id"}}
        ),
    },
    {
        "name": "get_run_manifest",
        "description": "How a run was made: software version, pipeline steps with parameters, "
        "models, plate format and channels, status, image-QC summary (paths and user names "
        "removed).",
        "input_schema": _schema({"run_id": {"type": "integer", "description": "runs.run_id"}}),
    },
    {
        "name": "get_dose_response",
        "description": "Dose-response fits filtered by run, compound, cell line and readout; "
        "optionally with the single-agent points (concentration, well value).",
        "input_schema": _schema(
            {
                "run_id": _nullable("integer", "runs.run_id, or null for all runs"),
                "compound": _nullable("string", "compound name/code, or null"),
                "cell_line": _nullable("string", "cell line, or null"),
                "readout": _nullable("string", "readout ID, or null"),
                "include_points": {"type": "boolean", "description": "add the data points"},
            }
        ),
    },
    {
        "name": "get_combination_scores",
        "description": "Synergy scores filtered by run, cell line, model, drug and a minimum "
        "mean score.",
        "input_schema": _schema(
            {
                "run_id": _nullable("integer", "runs.run_id, or null"),
                "cell_line": _nullable("string", "cell line, or null"),
                "model": _nullable("string", "bliss | hsa | loewe | zip, or null"),
                "drug": _nullable("string", "a drug in the pair, or null"),
                "min_mean": _nullable("number", "only scores with mean ≥ this, or null"),
            }
        ),
    },
    {
        "name": "get_qc_summary",
        "description": "A run's QC: plate QC per readout (Z′, CVs, …), wells by QC flag, and "
        "the image-QC summary.",
        "input_schema": _schema({"run_id": {"type": "integer", "description": "runs.run_id"}}),
    },
    {
        "name": "list_readouts",
        "description": "Readouts with display names, units and direction, for one run or all.",
        "input_schema": _schema({"run_id": _nullable("integer", "runs.run_id, or null")}),
    },
    {
        "name": "make_chart",
        "description": "Draw a chart in HCS Studio from a read-only query. The data stays on "
        "the user's computer; you get back only the number of points.",
        "input_schema": _schema(
            {
                "title": {"type": "string", "description": "chart title"},
                "kind": {
                    "type": "string",
                    "enum": ["scatter", "line", "bar", "dose_response"],
                    "description": "chart type",
                },
                "sql": {"type": "string", "description": "SELECT giving the columns below"},
                "x": {"type": "string", "description": "column for the x axis"},
                "y": {"type": "string", "description": "column for the y axis"},
                "group": _nullable("string", "column for colour groups, or null"),
                "log_x": {"type": "boolean", "description": "logarithmic x axis"},
            }
        ),
    },
    {
        "name": "highlight_wells",
        "description": "Select wells in the user's plate view (no data changes).",
        "input_schema": _schema(
            {
                "run_id": _nullable("integer", "runs.run_id, or null"),
                "plate_id": _nullable("integer", "plates.plate_id, or null"),
                "wells": {"type": "array", "items": {"type": "string"}, "description": "wells"},
                "note": {"type": "string", "description": "why, shown to the user"},
            }
        ),
    },
    {
        "name": "open_comparison",
        "description": "Open the comparison view for wells of a run (no data changes).",
        "input_schema": _schema(
            {
                "run_id": {"type": "integer", "description": "runs.run_id"},
                "wells": {"type": "array", "items": {"type": "string"}, "description": "wells"},
            }
        ),
    },
    {
        "name": "propose_pipeline_change",
        "description": "Propose parameter changes to a run's pipeline. The user reviews and "
        "applies them; you cannot change pipelines yourself.",
        "input_schema": _schema(
            {
                "run_id": {"type": "integer", "description": "runs.run_id"},
                "summary": {"type": "string", "description": "what the change is for"},
                "changes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "step": {"type": "integer", "description": "step number, from 1"},
                            "parameter": {"type": "string", "description": "parameter name"},
                            "new_value": {"type": "string", "description": "value as JSON"},
                            "reason": {"type": "string", "description": "why"},
                        },
                        "required": ["step", "parameter", "new_value", "reason"],
                        "additionalProperties": False,
                    },
                },
            }
        ),
    },
    {
        "name": "draft_methods_text",
        "description": "Draft a methods paragraph for a run from its manifest, for the user to "
        "edit. Returns the paragraphs and the manifest facts each one uses.",
        "input_schema": _schema(
            {
                "run_id": {"type": "integer", "description": "runs.run_id"},
                "style": {"type": "string", "enum": ["short", "full"], "description": "length"},
            }
        ),
    },
    {
        "name": "get_image_thumbnail",
        "description": "A downsampled 8-bit preview of one field (only offered when the user "
        "enabled images for this question; the user confirms each image).",
        "input_schema": _schema(
            {
                "run_id": {"type": "integer", "description": "runs.run_id"},
                "well": {"type": "string", "description": "well, e.g. B07"},
                "field": {"type": "integer", "description": "field number, from 1"},
            }
        ),
    },
]

TOOL_NAMES = tuple(t["name"] for t in TOOL_SPECS)


def tool_definitions(images: bool = False, strict: bool = True) -> list[dict[str, Any]]:
    """The tool list sent to Claude, in a fixed order (stable for prompt caching)."""
    out = []
    for spec in TOOL_SPECS:
        if spec["name"] in IMAGE_TOOLS and not images:
            continue
        out.append({**spec, "strict": True} if strict else dict(spec))
    return out


def _json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), default=str, ensure_ascii=False)


def _rows_as_dicts(result: QueryResult) -> list[dict[str, Any]]:
    return [dict(zip(result.columns, r, strict=True)) for r in result.rows]


def _check_types(name: str, data: dict[str, Any]) -> dict[str, Any]:
    """Validate a call's input against the tool's schema (strict tool use is not available
    everywhere, so never trust it)."""
    spec = next((t for t in TOOL_SPECS if t["name"] == name), None)
    if spec is None:
        raise ToolError(f"Unknown tool {name!r}")
    if not isinstance(data, dict):
        raise ToolError("The input must be an object")
    properties = spec["input_schema"]["properties"]
    unknown = set(data) - set(properties)
    if unknown:
        raise ToolError(f"Unknown input fields: {', '.join(sorted(unknown))}")
    out: dict[str, Any] = {}
    for key, prop in properties.items():
        value = data.get(key)
        kinds = prop["type"] if isinstance(prop["type"], list) else [prop["type"]]
        if value is None:
            if "null" not in kinds:
                raise ToolError(f"{key} is required")
            out[key] = None
            continue
        ok = any(
            (k == "integer" and isinstance(value, int) and not isinstance(value, bool))
            or (k == "number" and isinstance(value, (int, float)) and not isinstance(value, bool))
            or (k == "string" and isinstance(value, str))
            or (k == "boolean" and isinstance(value, bool))
            or (k == "array" and isinstance(value, list))
            for k in kinds
        )
        if not ok:
            raise ToolError(f"{key} must be {' or '.join(kinds)}")
        if "enum" in prop and value not in prop["enum"]:
            raise ToolError(f"{key} must be one of {', '.join(prop['enum'])}")
        out[key] = value
    return out


class ToolBox:
    """Runs tool calls for one conversation against one snapshot."""

    def __init__(
        self,
        snapshot: Snapshot,
        *,
        max_rows: int = 200,
        timeout_s: float = 5.0,
        drafter: Callable[[dict[str, Any], str], dict[str, Any]] | None = None,
    ) -> None:
        self.snapshot = snapshot
        self.db = ReadOnlyDatabase(snapshot.path, max_rows=max_rows, timeout_s=timeout_s)
        self.chart_db = ReadOnlyDatabase(
            snapshot.path, max_rows=MAX_CHART_ROWS, timeout_s=timeout_s
        )
        self.drafter = drafter
        self.images_enabled = False
        #: thumbnails the user confirmed: (run_id, well, field)
        self.confirmed: set[tuple[int, str, int]] = set()

    def close(self) -> None:
        self.db.close()
        self.chart_db.close()

    # --- dispatch -------------------------------------------------------------------------

    def run(self, name: str, data: dict[str, Any]) -> ToolOutcome:
        try:
            args = _check_types(name, data)
            if name in IMAGE_TOOLS and not self.images_enabled:
                raise ToolError("Images are not enabled for this question")
            handler = getattr(self, f"_t_{name}")
            outcome: ToolOutcome = handler(**args)
            return outcome
        except NeedsConfirmation:
            raise
        except (ToolError, QueryRejected, QueryTimeout) as exc:
            return ToolOutcome(f"Error: {exc}", summary=f"error: {exc}", is_error=True)

    def _query(self, sql: str, params: tuple[Any, ...] = ()) -> QueryResult:
        """Internal queries of the tools (parameterised, same read-only connection)."""
        if not params:
            return self.db.query(sql, max_rows=MAX_CHART_ROWS)
        cursor = self.db.db.execute(sql, params)
        columns = [d[0] for d in cursor.description or []]
        rows = [list(r) for r in cursor.fetchall()]
        return QueryResult(sql, columns, rows, False, 0.0)

    def _manifest(self, run_id: int) -> dict[str, Any]:
        folder = self._folder(run_id)
        try:
            return dict(json.loads((folder / "run_manifest.json").read_text(encoding="utf-8")))
        except (OSError, ValueError) as exc:
            raise ToolError(f"The manifest of run {run_id} is not readable") from exc

    def _folder(self, run_id: int) -> Path:
        rooted = self.snapshot.run_folders.get(int(run_id))
        if not rooted:
            raise ToolError(f"No run {run_id}")
        from hcs_studio.paths import resolve

        try:
            return resolve(rooted)
        except Exception as exc:
            raise ToolError(f"Run {run_id}'s folder is not reachable from here") from exc

    # --- tools ----------------------------------------------------------------------------

    def _t_get_schema(self) -> ToolOutcome:
        return ToolOutcome(_json(schema_json(self.snapshot.privacy)), summary="schema")

    def _t_run_query(self, sql: str, purpose: str) -> ToolOutcome:
        result = self.db.query(sql)
        note = f"{len(result.rows)} rows" + (" (truncated)" if result.truncated else "")
        return ToolOutcome(_json(result.to_json()), summary=f"{purpose}: {note}", sql=[result.sql])

    def _t_get_plate_layout(self, plate_id: int) -> ToolOutcome:
        sql = (
            "SELECT well, role, cell_line, compound_1, concentration_1_m, compound_2, "
            "concentration_2_m FROM wells WHERE plate_id = ? ORDER BY well"
        )
        result = self._query(sql, (plate_id,))
        if not result.rows:
            raise ToolError(f"No layout for plate {plate_id}")
        wells = _rows_as_dicts(result)
        roles: dict[str, int] = {}
        compounds: dict[str, list[float]] = {}
        for w in wells:
            roles[w["role"]] = roles.get(w["role"], 0) + 1
            for name, conc in (
                (w["compound_1"], w["concentration_1_m"]),
                (w["compound_2"], w["concentration_2_m"]),
            ):
                if name:
                    compounds.setdefault(name, []).append(conc or 0.0)
        content = {
            "plate_id": plate_id,
            "wells_annotated": len(wells),
            "roles": roles,
            "cell_lines": sorted({w["cell_line"] for w in wells if w["cell_line"]}),
            "compounds": {
                n: {"min_m": min(c), "max_m": max(c), "levels": len(set(c))}
                for n, c in sorted(compounds.items())
            },
            "wells": [
                [
                    w["well"],
                    w["role"],
                    w["cell_line"],
                    w["compound_1"],
                    w["concentration_1_m"],
                    w["compound_2"],
                    w["concentration_2_m"],
                ]
                for w in wells
            ],
            "well_columns": [
                "well",
                "role",
                "cell_line",
                "compound_1",
                "concentration_1_m",
                "compound_2",
                "concentration_2_m",
            ],
        }
        return ToolOutcome(_json(content), summary=f"layout of plate {plate_id}", sql=[sql])

    def _t_get_run_manifest(self, run_id: int) -> ToolOutcome:
        manifest = self._manifest(run_id)
        plate = manifest.get("plate") or {}
        metadata = plate.get("metadata") or {}
        steps = (manifest.get("pipeline") or {}).get("steps") or []
        curated = {
            "run_id": run_id,
            "status": manifest.get("status"),
            "software_version": (manifest.get("software") or {}).get("version"),
            "started": manifest.get("started"),
            "finished": manifest.get("finished"),
            "plate": {
                "barcode": plate.get("barcode"),
                "vendor": plate.get("vendor"),
                "rows": plate.get("rows"),
                "cols": plate.get("cols"),
                "timepoints": plate.get("timepoints"),
                "channels": metadata.get("channels"),
                "objective": metadata.get("objective"),
                "pixel_size_um": metadata.get("pixel_size_um"),
                "z_step_um": metadata.get("z_step_um"),
            },
            "pipeline": [
                {"step": i + 1, "module": s.get("module"), "params": s.get("params") or {}}
                for i, s in enumerate(steps)
                if isinstance(s, dict)
            ],
            "models": manifest.get("models") or [],
            "wells_processed": len(manifest.get("wells") or []),
            "failed_wells": len(manifest.get("failed_wells") or {}),
            "image_qc": manifest.get("image_qc"),
            "layout": manifest.get("layout"),
            "run_version": manifest.get("run_version"),
        }
        content = scrub(curated, self.snapshot.pseudonyms)
        return ToolOutcome(_json(content), summary=f"manifest of run {run_id}")

    def _t_get_dose_response(
        self,
        run_id: int | None,
        compound: str | None,
        cell_line: str | None,
        readout: str | None,
        include_points: bool,
    ) -> ToolOutcome:
        where: list[str] = []
        params: list[Any] = []
        for column, value in (
            ("run_id", run_id),
            ("compound", compound),
            ("cell_line", cell_line),
            ("readout", readout),
        ):
            if value is not None:
                where.append(f"{column} = ?")
                params.append(value)
        sql = "SELECT * FROM dose_response" + (" WHERE " + " AND ".join(where) if where else "")
        sql += " ORDER BY run_id, compound, cell_line"
        fits = self._query(sql, tuple(params))
        content: dict[str, Any] = {"fits": _rows_as_dicts(fits)[:200], "n_fits": len(fits.rows)}
        sqls = [sql]
        if include_points and compound is not None:
            point_sql = (
                "SELECT r.run_id, w.cell_line, w.concentration_1_m AS concentration_m, "
                "r.well, r.readout, r.value FROM well_results r JOIN runs u ON u.run_id = r.run_id "
                "JOIN wells w ON w.plate_id = u.plate_id AND w.well = r.well "
                "WHERE w.n_compounds = 1 AND w.compound_1 = ?"
            )
            point_params: list[Any] = [compound]
            for column, value in (
                ("r.run_id", run_id),
                ("w.cell_line", cell_line),
                ("r.readout", readout),
            ):
                if value is not None:
                    point_sql += f" AND {column} = ?"
                    point_params.append(value)
            point_sql += " ORDER BY r.run_id, w.cell_line, concentration_m"
            points = self._query(point_sql, tuple(point_params))
            content["points"] = {"columns": points.columns, "rows": points.rows[:1000]}
            sqls.append(point_sql)
        return ToolOutcome(_json(content), summary=f"{len(fits.rows)} fits", sql=sqls)

    def _t_get_combination_scores(
        self,
        run_id: int | None,
        cell_line: str | None,
        model: str | None,
        drug: str | None,
        min_mean: float | None,
    ) -> ToolOutcome:
        where: list[str] = []
        params: list[Any] = []
        for column, value in (("run_id", run_id), ("cell_line", cell_line), ("model", model)):
            if value is not None:
                where.append(f"{column} = ?")
                params.append(value)
        if drug is not None:
            where.append("(drug_a = ? OR drug_b = ?)")
            params += [drug, drug]
        if min_mean is not None:
            where.append("mean >= ?")
            params.append(min_mean)
        sql = "SELECT * FROM combination_scores" + (
            " WHERE " + " AND ".join(where) if where else ""
        )
        sql += " ORDER BY mean DESC"
        result = self._query(sql, tuple(params))
        content = {"scores": _rows_as_dicts(result)[:300], "n_scores": len(result.rows)}
        return ToolOutcome(_json(content), summary=f"{len(result.rows)} scores", sql=[sql])

    def _t_get_qc_summary(self, run_id: int) -> ToolOutcome:
        qc_sql = "SELECT * FROM plate_qc WHERE run_id = ?"
        flags_sql = (
            "SELECT qc_flag, COUNT(DISTINCT well) AS wells FROM well_results WHERE run_id = ? "
            "GROUP BY qc_flag"
        )
        qc = self._query(qc_sql, (run_id,))
        flags = self._query(flags_sql, (run_id,))
        image_qc: Any = None
        try:
            path = self._folder(run_id) / "qc" / "plate_qc.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            keys = (
                "passed",
                "reasons",
                "fields",
                "excluded_fields",
                "wells",
                "failed_wells",
                "flag_counts",
                "compound_flags",
            )
            image_qc = scrub({k: data.get(k) for k in keys}, self.snapshot.pseudonyms)
        except (OSError, ValueError, ToolError):
            image_qc = None
        content = {
            "plate_qc": _rows_as_dicts(qc),
            "wells_by_qc_flag": {(r[0] or "ok"): r[1] for r in flags.rows},
            "image_qc": image_qc,
        }
        return ToolOutcome(_json(content), summary=f"QC of run {run_id}", sql=[qc_sql, flags_sql])

    def _t_list_readouts(self, run_id: int | None) -> ToolOutcome:
        if run_id is None:
            sql = "SELECT * FROM readouts ORDER BY readout"
            result = self._query(sql)
        else:
            sql = (
                "SELECT * FROM readouts WHERE readout IN (SELECT DISTINCT readout FROM "
                "well_results WHERE run_id = ?) ORDER BY readout"
            )
            result = self._query(sql, (run_id,))
        return ToolOutcome(
            _json({"readouts": _rows_as_dicts(result)}),
            summary=f"{len(result.rows)} readouts",
            sql=[sql],
        )

    def _t_make_chart(
        self,
        title: str,
        kind: str,
        sql: str,
        x: str,
        y: str,
        group: str | None,
        log_x: bool,
    ) -> ToolOutcome:
        result = self.chart_db.query(sql)
        missing = [c for c in (x, y, group) if c and c not in result.columns]
        if missing:
            raise ToolError(f"The query has no column {', '.join(missing)}")
        chart = {
            "action": "chart",
            "title": title,
            "kind": kind,
            "x": x,
            "y": y,
            "group": group,
            "log_x": log_x,
            "columns": result.columns,
            "rows": result.rows,
            "sql": result.sql,
        }
        content = {"chart": title, "points": len(result.rows), "truncated": result.truncated}
        return ToolOutcome(
            _json(content),
            summary=f"chart '{title}' ({len(result.rows)} points)",
            sql=[result.sql],
            ui=[chart],
        )

    def _t_highlight_wells(
        self, run_id: int | None, plate_id: int | None, wells: list[str], note: str
    ) -> ToolOutcome:
        clean = sorted({str(w).strip().upper() for w in wells if str(w).strip()})
        action = {
            "action": "highlight_wells",
            "run_id": run_id,
            "plate_id": plate_id,
            "wells": clean,
            "note": note,
        }
        return ToolOutcome(
            _json({"highlighted": len(clean)}), summary=f"highlight {len(clean)} wells", ui=[action]
        )

    def _t_open_comparison(self, run_id: int, wells: list[str]) -> ToolOutcome:
        self._folder(run_id)  # the run exists
        clean = sorted({str(w).strip().upper() for w in wells if str(w).strip()})
        action = {"action": "open_comparison", "run_id": run_id, "wells": clean}
        return ToolOutcome(_json({"opened": len(clean)}), summary="comparison view", ui=[action])

    def _t_propose_pipeline_change(
        self, run_id: int, summary: str, changes: list[dict[str, Any]]
    ) -> ToolOutcome:
        steps = (self._manifest(run_id).get("pipeline") or {}).get("steps") or []
        from hcs_studio.processing.base import MODULES, load_builtin_modules

        load_builtin_modules()
        diff, problems = [], []
        for change in changes:
            if not isinstance(change, dict):
                problems.append("a change is not an object")
                continue
            step = change.get("step")
            parameter = str(change.get("parameter", ""))
            if not isinstance(step, int) or not 1 <= step <= len(steps):
                problems.append(f"no step {step} (the pipeline has {len(steps)})")
                continue
            module = str(steps[step - 1].get("module", ""))
            params = steps[step - 1].get("params") or {}
            cls = MODULES.get(module) if module in MODULES else None
            fields = getattr(getattr(cls, "Params", None), "model_fields", {}) if cls else {}
            if fields and parameter not in fields:
                problems.append(f"step {step} ({module}) has no parameter {parameter!r}")
                continue
            try:
                new = json.loads(str(change.get("new_value", "")))
            except ValueError:
                new = change.get("new_value")
            diff.append(
                {
                    "step": step,
                    "module": module,
                    "parameter": parameter,
                    "old": params.get(
                        parameter, fields[parameter].default if parameter in fields else None
                    ),
                    "new": new,
                    "reason": change.get("reason", ""),
                }
            )
        if problems and not diff:
            raise ToolError("; ".join(problems))
        proposal = {
            "action": "pipeline_proposal",
            "run_id": run_id,
            "summary": summary,
            "changes": scrub(diff),
            "problems": problems,
        }
        content = {
            "proposal": "shown to the user, who decides whether to apply it",
            "changes": scrub(diff, self.snapshot.pseudonyms),
            "problems": problems,
        }
        return ToolOutcome(_json(content), summary=f"proposed {len(diff)} change(s)", ui=[proposal])

    def _t_draft_methods_text(self, run_id: int, style: str) -> ToolOutcome:
        manifest_outcome = self._t_get_run_manifest(run_id)
        facts = json.loads(str(manifest_outcome.content))
        if self.drafter is None:
            raise ToolError("Methods drafting is not available")
        draft = self.drafter(facts, style)
        action = {"action": "methods_draft", "run_id": run_id, "draft": draft}
        return ToolOutcome(_json(draft), summary=f"methods draft for run {run_id}", ui=[action])

    def thumbnail_request(self, run_id: int, well: str, field: int) -> dict[str, Any]:
        return {"run_id": int(run_id), "well": well.strip().upper(), "field": int(field)}

    def _t_get_image_thumbnail(self, run_id: int, well: str, field: int) -> ToolOutcome:
        request = self.thumbnail_request(run_id, well, field)
        key = (request["run_id"], request["well"], request["field"])
        if key not in self.confirmed:
            raise NeedsConfirmation(request)
        path = self._folder(run_id) / "wells" / request["well"] / "previews"
        path = path / f"f{request['field']:03d}.png"
        if not path.exists():
            raise ToolError(f"No preview of {request['well']} field {request['field']}")
        data = thumbnail_png(path)
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": f"Run {run_id}, well {request['well']}, field "
                f"{request['field']} (8-bit preview, at most {THUMBNAIL_PX} px).",
            },
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": base64.b64encode(data).decode("ascii"),
                },
            },
        ]
        return ToolOutcome(content, summary=f"thumbnail {request['well']} f{request['field']}")


def thumbnail_png(path: Path, size: int = THUMBNAIL_PX) -> bytes:
    """A downsampled 8-bit RGB PNG of a preview (metadata stripped)."""
    from PIL import Image

    with Image.open(path) as image:
        rgb = image.convert("RGB")
        rgb.thumbnail((size, size))
        clean = Image.new("RGB", rgb.size)
        clean.paste(rgb)
    buffer = io.BytesIO()
    clean.save(buffer, format="PNG")
    return buffer.getvalue()

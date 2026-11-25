"""FijiRunner: ilastik through Fiji's ilastik4ij plugin (PROMPT_ILASTIK.md §3.3), and Fiji
macros for collaborators ("Export as Fiji macro", §7).

The runner writes an ImageJ macro and starts Fiji without a window::

    <Fiji launcher> --headless --console -macro <job>.ijm

The macro configures the plugin (ilastik's path, threads, RAM), then per input: ``Import HDF5``
(our ilastik-ready file) → ``Run … Prediction`` → ``Export HDF5`` of the result into the file
HeadlessRunner would have written, and finally a marker file. Fiji's exit code does not report
macro errors, so a missing marker (or output) is the failure signal.

Command names and macro keys were read from the ilastik4ij source (main branch, Feb 2026): the
keys are the commands' parameter names in lower case — ``projectfilename``, ``inputimage``,
``pixelclassificationtype`` (Probabilities | Segmentation), ``autocontextpredictiontype``,
``inputproborsegimage`` + ``secondinputtype`` + ``objectexportsource`` (objects),
``boundarypredictionimage`` (multicut); Import HDF5 ``select datasetname axisorder``; Export HDF5
``input exportpath datasetname compressionlevel`` (written as ``tzyxc`` with vigra axistags);
Configure ``executablefile numthreads maxrammb``. Some published example macros use ``input=``
instead of ``inputimage=``: the source is what counts.

What the plugin cannot do, and the runner therefore refuses (the step then runs headless and
says so in its log): Object Classification from a label image (``--label_image``: the plugin
offers only probabilities or a binary segmentation), Tracking exports other than the lineage
image (HCS Studio needs Object-Identities + the CSV table), autocontext stage-1 / all-stage
exports, Uncertainty/Features/Labels, and output formats other than HDF5. The plugin starts
ilastik once per image (no well batching), so it is slower than HeadlessRunner.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

from hcs_studio.segmentation.ilastik_backend.job import IlastikError, IlastikJob, IlastikResult
from hcs_studio.segmentation.ilastik_backend.project import file_sha256
from hcs_studio.segmentation.ilastik_backend.settings import ilastik_command

LOG_TAIL_LINES = 40
#: ilastik4ij's own default when no limit is set (its option has no "unlimited")
DEFAULT_RAM_MB = 4096
DONE_TEXT = "hcs-studio macro finished"

#: (workflow, export source) → (plugin command, its output-type key, value)
COMMANDS: dict[tuple[str, str], tuple[str, str, str]] = {
    ("pixel", "Probabilities"): (
        "Run Pixel Classification Prediction",
        "pixelclassificationtype",
        "Probabilities",
    ),
    ("pixel", "Simple Segmentation"): (
        "Run Pixel Classification Prediction",
        "pixelclassificationtype",
        "Segmentation",
    ),
    ("autocontext", "Probabilities Stage 2"): (
        "Run Autocontext Prediction",
        "autocontextpredictiontype",
        "Probabilities",
    ),
    ("autocontext", "Simple Segmentation Stage 2"): (
        "Run Autocontext Prediction",
        "autocontextpredictiontype",
        "Segmentation",
    ),
    ("objects", "Object Predictions"): (
        "Run Object Classification Prediction",
        "objectexportsource",
        "Object Predictions",
    ),
    ("objects", "Object Probabilities"): (
        "Run Object Classification Prediction",
        "objectexportsource",
        "Object Probabilities",
    ),
    ("objects", "Object Identities"): (
        "Run Object Classification Prediction",
        "objectexportsource",
        "Object Identities",
    ),
    ("multicut", "Multicut Segmentation"): ("Run Multicut", "", ""),
    ("tracking", "Tracking-Result"): ("Run Tracking", "", ""),
}

#: second input role (ilastik flag) → the plugin's image key and its input type
SECOND_INPUTS: dict[str, dict[str, tuple[str, str]]] = {
    "objects": {
        "prediction_maps": ("inputproborsegimage", "Probabilities"),
        "segmentation_image": ("inputproborsegimage", "Segmentation"),
    },
    "tracking": {
        "prediction_maps": ("inputproborsegimage", "Probabilities"),
        "segmentation_image": ("inputproborsegimage", "Segmentation"),
    },
    "multicut": {"probabilities": ("boundarypredictionimage", "")},
}


def macro_path(path: str | Path) -> str:
    """A path for a macro string: forward slashes (Java accepts them on Windows too), so no
    escaping is needed. ``]`` and ``"`` would break ImageJ's argument parser."""
    text = Path(path).as_posix()
    if any(c in text for c in '[]"'):
        raise IlastikError(f"Fiji macros cannot use paths containing [ ] or quotes: {text}")
    return text


def unsupported(job: IlastikJob) -> str | None:
    """Why the plugin cannot run this job (None = it can)."""
    if (job.workflow, job.export_source) not in COMMANDS:
        return f"the ilastik4ij plugin has no {job.workflow} export {job.export_source!r}"
    if job.output_format != "hdf5" or job.export_plugin:
        return "the ilastik4ij plugin exports HDF5 images only"
    allowed = SECOND_INPUTS.get(job.workflow, {})
    for role in job.roles:
        if role not in allowed:
            return f"the ilastik4ij plugin has no {job.workflow} input --{role}"
    if len(job.roles) > 1:
        return "the ilastik4ij plugin takes one second input"
    if job.workflow in SECOND_INPUTS and not job.roles:
        return f"the ilastik4ij {job.workflow} command needs a second input"
    return None


def _split(spec: str) -> tuple[str, str]:
    """``<file>.h5/<dataset>`` → (file, dataset)."""
    file, sep, dataset = spec.rpartition(".h5/")
    if not sep:
        return spec, "data"
    return file + ".h5", dataset


def _axes_of(spec: str) -> str:
    from hcs_studio.segmentation.ilastik_backend.h5io import read_axes

    file, dataset = _split(spec)
    axes = read_axes(Path(file), dataset)
    if not axes:
        raise IlastikError(f"{file} has no axistags: write inputs with h5io.write_h5")
    return axes


def configure_line(ilastik: str, threads: int, ram_mb: int) -> str:
    options = (
        f"executablefile=[{macro_path(ilastik)}] numthreads={threads} "
        f"maxrammb={ram_mb or DEFAULT_RAM_MB}"
    )
    return f'run("Configure ilastik executable location", "{options}");'


def _command_line(
    job: IlastikJob, raw_title: str, second: tuple[str, str] | None, second_title: str
) -> str:
    command, key, value = COMMANDS[(job.workflow, job.export_source)]
    parts = [f"projectfilename=[{macro_path(job.project)}]", f"inputimage=[{raw_title}]"]
    if second is not None:
        image_key, input_type = second
        parts.append(f"{image_key}=[{second_title}]")
        if input_type:
            parts.append(f"secondinputtype=[{input_type}]")
    if key:
        parts.append(f"{key}=[{value}]")
    return f'run("{command}", "{" ".join(parts)}");'


def job_macro(
    job: IlastikJob, ilastik: str, *, configure: bool = True, done_file: Path | None = None
) -> str:
    """A self-contained macro for one job: literal paths, one block per input, no variables."""
    reason = unsupported(job)
    if reason:
        raise IlastikError(reason)
    role = next(iter(job.roles), None)
    second = SECOND_INPUTS[job.workflow][role] if role else None
    lines = [
        "// Written by HCS Studio for the ilastik4ij plugin (Fiji → Plugins → ilastik).",
        f"// Project: {macro_path(job.project)}",
        f"// Export: {job.export_source}; {len(job.raw_data)} image(s)",
        "setBatchMode(true);",
    ]
    if configure:
        lines.append(configure_line(ilastik, job.threads, job.ram_mb))
    outputs = job.expected_outputs()
    for i, spec in enumerate(job.raw_data):
        file, dataset = _split(spec)
        raw_title = f"hcs_raw_{i}"
        lines += [
            f"// image {i + 1} of {len(job.raw_data)}",
            f'run("Import HDF5", "select=[{macro_path(file)}] datasetname=[/{dataset}] '
            f'axisorder=[{_axes_of(spec)}]");',
            f'rename("{raw_title}");',
        ]
        second_title = f"hcs_second_{i}"
        if role:
            other = job.roles[role][i]
            other_file, other_dataset = _split(other)
            lines += [
                f'run("Import HDF5", "select=[{macro_path(other_file)}] '
                f'datasetname=[/{other_dataset}] axisorder=[{_axes_of(other)}]");',
                f'rename("{second_title}");',
            ]
        lines += [
            _command_line(job, raw_title, second, second_title),
            f'rename("hcs_out_{i}");',
            f'run("Export HDF5", "input=[hcs_out_{i}] exportpath=[{macro_path(outputs[i])}] '
            'datasetname=[exported_data] compressionlevel=[0]");',
            'close("*");',
            f'print("HCS Studio: image {i + 1} of {len(job.raw_data)} done");',
        ]
    if done_file is not None:
        lines.append(f'File.saveString("{DONE_TEXT}", "{macro_path(done_file)}");')
    return "\n".join(lines) + "\n"


class FijiRunner:
    name = "fiji"

    def __init__(self, launcher: str | Path, ilastik: str | Path, *, configure: bool = True):
        self.launcher = str(launcher)
        self.ilastik = str(ilastik)
        self.configure = configure

    @staticmethod
    def check_ilastik(ilastik: str | Path) -> str | None:
        """The plugin starts ilastik itself: it needs the program, not a Python interpreter."""
        if len(ilastik_command(ilastik)) > 1:
            return (
                "the Fiji runner needs the ilastik program (ilastik.exe / run_ilastik.sh), not "
                f"the Python of an ilastik environment ({ilastik})"
            )
        return None

    def unsupported(self, job: IlastikJob) -> str | None:
        return self.check_ilastik(self.ilastik) or unsupported(job)

    def command(self, macro: Path) -> list[str]:
        return [self.launcher, "--headless", "--console", "-macro", str(macro)]

    def run(self, job: IlastikJob, log_path: Path | None = None) -> IlastikResult:
        reason = self.unsupported(job)
        if reason:
            raise IlastikError(f"The Fiji runner cannot run this step: {reason}")
        folder = Path(job.output_folder)
        folder.mkdir(parents=True, exist_ok=True)
        done = folder / f".fiji_{job.output_suffix}.done"
        done.unlink(missing_ok=True)
        for output in job.expected_outputs():
            Path(output).unlink(missing_ok=True)  # never mistake an old export for a new one
        text = job_macro(job, self.ilastik, configure=self.configure, done_file=done)
        macro = folder / f"fiji_{job.output_suffix}.ijm"
        macro.write_text(text, encoding="utf-8")
        command = self.command(macro)
        before = file_sha256(Path(job.project))
        started = time.monotonic()
        try:
            finished = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=job.timeout_s,
                check=False,
                stdin=subprocess.DEVNULL,
            )
        except FileNotFoundError as exc:
            raise IlastikError(
                f"Fiji could not be started ({self.launcher}): set its folder in Tools → "
                "ilastik settings"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise IlastikError(f"Fiji did not finish within {job.timeout_s:g} s") from exc
        seconds = time.monotonic() - started
        output = (finished.stdout or "") + (finished.stderr or "")
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write("$ " + " ".join(command) + "\n" + text + output + "\n")
        tail = "\n".join(output.strip().splitlines()[-LOG_TAIL_LINES:])
        completed = done.exists() and done.read_text(encoding="utf-8").strip() == DONE_TEXT
        done.unlink(missing_ok=True)
        if finished.returncode != 0 or not completed:
            raise IlastikError(_explain(finished.returncode, completed, tail))
        after = file_sha256(Path(job.project))
        if after != before:
            raise IlastikError(
                f"ilastik changed the project file {job.project} during a Fiji run; results "
                "are not traceable to the recorded classifier"
            )
        missing = [p for p in job.expected_outputs() if not Path(p).exists()]
        if missing:
            raise IlastikError(
                f"Fiji finished but wrote no output for {len(missing)} input(s), e.g. "
                f"{Path(missing[0]).name}\n{tail}"
            )
        return IlastikResult(
            outputs=job.expected_outputs(),
            command=command,
            runner=self.name,
            returncode=finished.returncode,
            seconds=round(seconds, 2),
            log_tail=tail,
            project_sha256_before=before,
            project_sha256_after=after,
            macro=text,
        )


def _explain(code: int, completed: bool, tail: str) -> str:
    if code != 0:
        text = f"Fiji stopped with exit code {code}"
    elif not completed:
        text = "The Fiji macro did not finish (a macro error, or the ilastik4ij plugin failed)"
    else:
        text = "Fiji reported an error"
    if "Unrecognized command" in tail:
        text += (
            ": a command is missing — is the ilastik update site enabled in Fiji (Help → "
            "Update → Manage update sites → ilastik)?"
        )
    return f"{text}\n{tail}" if tail else text


# ------------------------------------------------------------------ macros for people


def folder_macro(
    *,
    project: str | Path,
    workflow: str,
    export_source: str,
    axis_order: str,
    second_role: str | None = None,
    second_suffix: str = "_probabilities",
    ilastik: str | Path | None = None,
    threads: int = -1,
    ram_mb: int = 0,
) -> str:
    """An ``.ijm`` a collaborator runs in Fiji: every ``*.h5`` of a chosen folder through the
    project, results saved next to them as ``<name>_<export>.h5``. Second inputs (objects,
    multicut, tracking) are found by name: ``<name><second_suffix>.h5``.

    Script parameters (``#@``) ask for the folders when run from Fiji's editor; for batch use
    remove them and set the variables (as the plugin's own examples say)."""
    job = IlastikJob(
        project=str(project),
        workflow=workflow,
        raw_data=["x.h5/data"],
        roles={second_role: ["y.h5/data"]} if second_role else {},
        export_source=export_source,
        output_folder=".",
        output_suffix="x",
    )
    reason = unsupported(job)
    if reason:
        raise IlastikError(reason)
    second = SECOND_INPUTS[workflow][second_role] if second_role else None
    suffix = export_source.replace(" ", "_")
    command = _command_line(job, "RAW", second, "SECOND")
    command = command.replace("[RAW]", '[" + raw + "]').replace("[SECOND]", '[" + other + "]')
    command = command.replace(
        f"[{macro_path(project)}]", '[" + project + "]'
    )  # the project is a variable too
    lines = [
        "// Written by HCS Studio: runs an ilastik project on every .h5 file of a folder with",
        "// the ilastik4ij plugin (Fiji → Help → Update → Manage update sites → ilastik).",
        f"// Workflow: {workflow}; export: {export_source}; axis order: {axis_order}",
        '#@ File (label = "ilastik project", style = "file") projectFile',
        '#@ File (label = "Folder of .h5 inputs", style = "directory") inputFolder',
        '#@ File (label = "Output folder", style = "directory") outputFolder',
        "project = projectFile.getAbsolutePath();",
        f'// project = "{macro_path(project)}";',
        f'axisOrder = "{axis_order}";',
        'datasetName = "/data";',
        f'suffix = "_{suffix}";',
    ]
    if second_role:
        lines.append(f'secondSuffix = "{second_suffix}";')
    if ilastik:
        lines.append(configure_line(str(ilastik), threads, ram_mb))
    else:
        lines.append(
            "// ilastik's location, threads and RAM: Plugins → ilastik → Configure ilastik "
            "executable location"
        )
    lines += [
        "setBatchMode(true);",
        "files = getFileList(inputFolder);",
        "for (i = 0; i < files.length; i++) {",
        "    name = files[i];",
        '    if (!endsWith(name, ".h5") || indexOf(name, suffix) >= 0) continue;',
    ]
    if second_role:
        lines.append("    if (indexOf(name, secondSuffix) >= 0) continue;")
    lines += [
        "    stem = substring(name, 0, lengthOf(name) - 3);",
        '    run("Import HDF5", "select=[" + inputFolder + File.separator + name + "] '
        'datasetname=[" + datasetName + "] axisorder=[" + axisOrder + "]");',
        "    raw = getTitle();",
    ]
    if second_role:
        lines += [
            '    run("Import HDF5", "select=[" + inputFolder + File.separator + stem + '
            'secondSuffix + ".h5] datasetname=[" + datasetName + "] axisorder=[" + axisOrder '
            '+ "]");',
            "    other = getTitle();",
        ]
    lines += [
        "    " + command,
        "    result = getTitle();",
        '    run("Export HDF5", "input=[" + result + "] exportpath=[" + outputFolder + '
        'File.separator + stem + suffix + ".h5] datasetname=[exported_data] '
        'compressionlevel=[0]");',
        '    close("*");',
        '    print("done: " + name);',
        "}",
    ]
    return "\n".join(lines) + "\n"


#: friendly export names per workflow (first = default) → ilastik export source
EXPORT_CHOICES: dict[str, dict[str, str]] = {
    "pixel": {"probabilities": "Probabilities", "segmentation": "Simple Segmentation"},
    "autocontext": {
        "probabilities": "Probabilities Stage 2",
        "segmentation": "Simple Segmentation Stage 2",
    },
    "objects": {
        "predictions": "Object Predictions",
        "probabilities": "Object Probabilities",
        "identities": "Object Identities",
    },
    "multicut": {"segmentation": "Multicut Segmentation"},
    "tracking": {"lineage": "Tracking-Result"},
}
#: second input per workflow: friendly name → ilastik role (first = default)
SECOND_CHOICES: dict[str, dict[str, str]] = {
    "objects": {"probabilities": "prediction_maps", "segmentation": "segmentation_image"},
    "tracking": {"probabilities": "prediction_maps", "segmentation": "segmentation_image"},
    "multicut": {"probabilities": "probabilities"},
}


def project_macro(
    project: str | Path,
    *,
    export: str | None = None,
    second: str | None = None,
    axis_order: str | None = None,
    ilastik: str | Path | None = None,
    threads: int = -1,
    ram_mb: int = 0,
) -> str:
    """``folder_macro`` for a project, with its workflow read from the ``.ilp``; ``axis_order``
    defaults to the project's training axes (else ``yxc``)."""
    from hcs_studio.segmentation.ilastik_backend.project import read_project

    info = read_project(Path(project))
    choices = EXPORT_CHOICES.get(info.workflow)
    if not choices:
        raise IlastikError(f"No Fiji macro for {info.workflow_name or info.workflow} projects")
    export = export or next(iter(choices))
    if export not in choices:
        raise IlastikError(f"{info.workflow} exports: {', '.join(choices)} (not {export!r})")
    role = None
    if info.workflow in SECOND_CHOICES:
        roles = SECOND_CHOICES[info.workflow]
        second = second or next(iter(roles))
        if second not in roles:
            raise IlastikError(f"second input: {', '.join(roles)} (not {second!r})")
        role = roles[second]
    return folder_macro(
        project=info.path,
        workflow=info.workflow,
        export_source=choices[export],
        axis_order=axis_order or info.axes or "yxc",
        second_role=role,
        second_suffix=f"_{second}" if second else "_probabilities",
        ilastik=ilastik,
        threads=threads,
        ram_mb=ram_mb,
    )

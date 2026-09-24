"""Evidence Intake: admit and extract benchmark-visible facts."""

from __future__ import annotations

import ast
import json
import re
import tokenize
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Iterable

from .core.errors import BundleError, ExtractionError
from .core.model import (
    ArtifactKind,
    RepoArtifact,
    SymbolSpan,
    TaskDocument,
    TraceEvent,
    VisibleBundle,
)
from .core.syntax import parse_python
from .language_support import (
    generic_source_drafts,
    generic_symbol_spans,
    is_supported_source,
)


# ---------------------------------------------------------------------------
# Production artifact admission
# ---------------------------------------------------------------------------

SOURCE_EXTENSIONS = {
    ".py", ".pyw", ".js", ".mjs", ".cjs", ".jsx", ".ts", ".tsx",
    ".java", ".go", ".rs", ".rb", ".php", ".c", ".h", ".cc",
    ".cpp", ".hpp", ".cs", ".swift", ".kt", ".kts", ".scala",
    ".lua", ".r", ".ex", ".exs", ".erl", ".hrl",
}
TEMPLATE_EXTENSIONS = {".j2", ".jinja", ".jinja2", ".tmpl", ".tpl", ".hbs", ".mustache"}
EXECUTABLE_EXTENSIONS = {".sh", ".bash", ".zsh", ".fish", ".ps1", ".bat", ".cmd"}
CONFIG_EXTENSIONS = {".json", ".toml", ".yaml", ".yml", ".ini", ".cfg", ".conf"}
DOCUMENT_EXTENSIONS = {".md", ".markdown", ".rst", ".rest", ".adoc", ".txt"}
BINARY_EXTENSIONS = {
    ".7z", ".avi", ".bmp", ".bz2", ".class", ".db", ".dll", ".dylib",
    ".eot", ".exe", ".gif", ".gz", ".ico", ".jar", ".jpeg", ".jpg",
    ".lock", ".m4a", ".mov", ".mp3", ".mp4", ".o", ".parquet", ".pdf",
    ".png", ".pyc", ".so", ".tar", ".ttf", ".webp", ".woff", ".woff2",
    ".xz", ".zip",
}

EXCLUDED_COMPONENTS = {
    ".git", ".github", ".circleci", ".gitlab", ".travis", ".ci", "ci",
    ".buildkite", ".azure-pipelines", ".kokoro", "kokoro", "appveyor",
    ".idea", ".vscode",
    "__pycache__", "test", "tests", "testing", "__tests__", "spec", "specs",
    "docs", "doc", "documentation", "docs_src", "docs-source", "docs_build",
    "docs-build", "benchmarks", "benchmark", "vendor",
    "gold", "golden", "patch", "patches",
    "third_party", "node_modules",
    "dist", "build", "coverage", "htmlcov",
    "testdata", "test_data", "test-data", "test_resources", "test-resources",
    "test_handcalcs", "test_module", "utest", "stresstest", "locking_tests",
    "language_agnostic_test",
}
EXCLUDED_SOURCE_NAMES = {
    "conftest.py", "noxfile.py", "pytest.ini", "runtests", "runtests.py",
    "run_tests", "run_tests.py", "run_tests.sh", "test", "test.py", "test.sh",
    "test.ps1", "install_test_deps.sh", "docusaurus.config.js",
    "docusaurus.config.ts", "sidebars.js", "sidebars.ts", "gold.py",
}
EXCLUDED_CONFIG_NAMES = {
    ".editorconfig", ".pre-commit-config.yaml", ".pre-commit-config.yml",
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "cargo.lock",
    "composer.lock", "go.sum", "pipfile.lock",
    ".travis.yml", ".travis.yaml", ".gitlab-ci.yml", ".gitlab-ci.yaml",
    "appveyor.yml", "appveyor.yaml", "azure-pipelines.yml",
    "azure-pipelines.yaml", "circle.yml", "codecov.yml", ".codecov.yml",
    "mkdocs.yml", "mkdocs.yaml",
}
TEST_FILE_PATTERN = re.compile(
    r"(?:^test_.+|^_?testing|.+(?:_|-)tests?|.+\.test|.+\.spec)(?:\.[^.]+)?$",
    re.IGNORECASE,
)
TEST_DEPENDENCY_PATTERN = re.compile(
    r"^requirements[-_.]tests?(?:\.[^.]+)?$", re.IGNORECASE
)
TEMPLATE_MARKER = re.compile(r"(?:\{\{|\{%|<%|\$\{[^}]+\})")
DOC_SITE_MARKERS = {"docusaurus.config.js", "docusaurus.config.ts"}


def documentation_site_roots(paths: Iterable[str]) -> set[str]:
    """Find repository subtrees that are deterministically documentation sites."""

    roots: set[str] = set()
    for path in paths:
        pure = PurePosixPath(path)
        if pure.name.casefold() in DOC_SITE_MARKERS and pure.parent != PurePosixPath("."):
            roots.add(pure.parent.as_posix())
    return roots


def classify_repository_file(path: str, prefix: bytes) -> ArtifactKind | None:
    """Return a production kind, or ``None`` for deterministically excluded input."""

    pure = PurePosixPath(path)
    lowered_parts = tuple(part.casefold() for part in pure.parts)
    name = pure.name.casefold()
    suffix = pure.suffix.casefold()

    if not pure.parts or any(part in EXCLUDED_COMPONENTS for part in lowered_parts[:-1]):
        return None
    if name in EXCLUDED_SOURCE_NAMES or name in EXCLUDED_CONFIG_NAMES:
        return None
    if TEST_FILE_PATTERN.fullmatch(name) or TEST_DEPENDENCY_PATTERN.fullmatch(name):
        return None
    dependency_manifest = (
        name.startswith("requirements") and suffix in {"", ".in", ".txt"}
    ) or "requirements" in lowered_parts[:-1]
    if suffix in BINARY_EXTENSIONS or (
        suffix in DOCUMENT_EXTENSIONS and not dependency_manifest
    ):
        return None
    if b"\x00" in prefix:
        return None

    if suffix in SOURCE_EXTENSIONS:
        return "source"
    if suffix in TEMPLATE_EXTENSIONS:
        return "runtime_template"
    if suffix == ".html":
        decoded = prefix.decode("utf-8", errors="ignore")
        if "templates" in lowered_parts or "views" in lowered_parts or TEMPLATE_MARKER.search(decoded):
            return "runtime_template"
        return None
    if suffix in EXECUTABLE_EXTENSIONS:
        return "executable"
    if not suffix and prefix.startswith(b"#!"):
        return "executable"
    if dependency_manifest or suffix in CONFIG_EXTENSIONS:
        return "configuration"
    return None


# ---------------------------------------------------------------------------
# Visible bundle loading
# ---------------------------------------------------------------------------

BENCHMARKS = {"specgap", "silentswap", "feedbacktrace"}
TASK_DOCUMENTS = {
    "specgap": "documents/3_document_after.md",
    "silentswap": "documents/original_document.md",
}
TRACE_TYPES = {"system", "user_prompt", "assistant_response", "tool_exchange"}


def load_visible_bundle(root: str | Path) -> VisibleBundle:
    bundle_root = Path(root).absolute()
    if not bundle_root.is_dir() or bundle_root.is_symlink():
        raise BundleError(f"bundle root must be a real directory: {bundle_root}")
    bundle_root = bundle_root.resolve(strict=True)
    manifest_path = bundle_root / "input_manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise BundleError("bundle must contain a regular input_manifest.json")
    manifest = _json_object(manifest_path, "input manifest")

    input_id = manifest.get("input_id")
    benchmark = manifest.get("benchmark")
    visible_files = manifest.get("visible_files")
    if not isinstance(input_id, str) or not input_id or input_id != bundle_root.name:
        raise BundleError("manifest input_id must equal the bundle directory name")
    if benchmark not in BENCHMARKS:
        raise BundleError(f"unsupported benchmark: {benchmark!r}")
    if not isinstance(visible_files, list) or not visible_files:
        raise BundleError("manifest visible_files must be a non-empty array")
    if any(not isinstance(value, str) or not value for value in visible_files):
        raise BundleError("visible_files entries must be non-empty strings")
    if len(visible_files) != len(set(visible_files)):
        raise BundleError("visible_files must not contain duplicates")

    resolved = {
        value: _resolve_visible_file(bundle_root, value) for value in visible_files
    }
    _require_exact_materialization(bundle_root, set(visible_files))

    if benchmark == "feedbacktrace":
        return _load_feedbacktrace(bundle_root, input_id, manifest, resolved)
    return _load_repository_benchmark(
        bundle_root, input_id, benchmark, manifest, resolved
    )


def _load_repository_benchmark(
    root: Path,
    input_id: str,
    benchmark: str,
    manifest: dict[str, Any],
    files: dict[str, Path],
) -> VisibleBundle:
    allowed_keys = {"input_id", "benchmark", "visible_files"}
    _reject_unknown_keys(manifest, allowed_keys, "input manifest")
    expected_document = TASK_DOCUMENTS[benchmark]
    document_paths = sorted(path for path in files if path.startswith("documents/"))
    if document_paths != [expected_document]:
        raise BundleError(f"{benchmark} requires exactly {expected_document}")
    invalid_roots = sorted(
        path for path in files if PurePosixPath(path).parts[0] not in {"documents", "repository"}
    )
    if invalid_roots:
        raise BundleError(f"unsupported visible root: {invalid_roots[0]}")

    document = TaskDocument(
        path=PurePosixPath(expected_document).name,
        content=_read_utf8(files[expected_document]),
    )
    repository_entries: list[tuple[str, Path]] = []
    for visible_path in sorted(files):
        pure = PurePosixPath(visible_path)
        if pure.parts[0] != "repository":
            continue
        repo_path = pure.relative_to("repository").as_posix()
        repository_entries.append((repo_path, files[visible_path]))
    repository_count = len(repository_entries)
    excluded_documentation_roots = documentation_site_roots(
        path for path, _ in repository_entries
    )
    artifacts: list[RepoArtifact] = []
    for repo_path, source_path in repository_entries:
        if any(
            repo_path == root or repo_path.startswith(f"{root}/")
            for root in excluded_documentation_roots
        ):
            continue
        prefix = source_path.read_bytes()[:8192]
        kind = classify_repository_file(repo_path, prefix)
        if kind is None:
            continue
        content = _read_source_text(source_path, repo_path)
        artifacts.append(
            RepoArtifact(
                path=repo_path,
                absolute_path=source_path,
                kind=kind,
                content=content,
            )
        )
    if repository_count == 0:
        raise BundleError(f"{benchmark} bundle has no repository files")
    if not artifacts:
        raise BundleError(f"{benchmark} bundle has no production artifacts")
    return VisibleBundle(
        root=root,
        input_id=input_id,
        benchmark=benchmark,  # type: ignore[arg-type]
        task_document=document,
        repo_artifacts=tuple(sorted(artifacts, key=lambda item: item.path)),
        trace_events=(),
        cutoff_turn=None,
        visible_repository_files=repository_count,
        excluded_repository_files=repository_count - len(artifacts),
    )


def _load_feedbacktrace(
    root: Path,
    input_id: str,
    manifest: dict[str, Any],
    files: dict[str, Path],
) -> VisibleBundle:
    allowed_keys = {"input_id", "benchmark", "visible_files", "cutoff_turn"}
    _reject_unknown_keys(manifest, allowed_keys, "input manifest")
    if list(files) != ["trace/model_input.json"]:
        raise BundleError("FeedbackTrace accepts only trace/model_input.json")
    cutoff = manifest.get("cutoff_turn")
    if not isinstance(cutoff, int) or isinstance(cutoff, bool) or cutoff <= 0:
        raise BundleError("FeedbackTrace cutoff_turn must be a positive integer")
    payload = _json_object(files["trace/model_input.json"], "trace input")
    if payload.get("input_id") != input_id:
        raise BundleError("trace input_id does not match the manifest")
    _reject_unknown_keys(payload, {"input_id", "events"}, "trace input")
    raw_events = payload.get("events")
    if not isinstance(raw_events, list) or not raw_events:
        raise BundleError("trace events must be a non-empty array")
    events: list[TraceEvent] = []
    previous_turn = -1
    seen_evidence: set[str] = set()
    for order, raw in enumerate(raw_events):
        if not isinstance(raw, dict):
            raise BundleError(f"trace event {order} must be an object")
        _reject_unknown_keys(
            raw,
            {"event_type", "turn_number", "content", "evidence_id", "tool_name"},
            f"trace event {order}",
        )
        event_type = raw.get("event_type")
        turn = raw.get("turn_number")
        content = raw.get("content")
        if event_type not in TRACE_TYPES:
            raise BundleError(f"trace event {order} has an invalid event_type")
        if not isinstance(turn, int) or isinstance(turn, bool) or turn < previous_turn:
            raise BundleError("trace turns must be non-negative and non-decreasing")
        if turn >= cutoff:
            raise BundleError(f"trace event {order} is at or after cutoff_turn")
        if not isinstance(content, str):
            raise BundleError(f"trace event {order} content must be a string")
        evidence_id = raw.get("evidence_id")
        tool_name = raw.get("tool_name")
        if evidence_id is not None and (not isinstance(evidence_id, str) or not evidence_id):
            raise BundleError(f"trace event {order} evidence_id must be non-empty")
        if tool_name is not None and (not isinstance(tool_name, str) or not tool_name):
            raise BundleError(f"trace event {order} tool_name must be non-empty")
        if event_type == "tool_exchange" and tool_name is None:
            raise BundleError("tool_exchange must carry tool_name")
        if event_type != "tool_exchange" and tool_name is not None:
            raise BundleError("only tool_exchange may carry tool_name")
        if evidence_id is not None:
            if evidence_id in seen_evidence:
                raise BundleError(f"duplicate trace evidence_id: {evidence_id}")
            seen_evidence.add(evidence_id)
        events.append(
            TraceEvent(
                event_type=event_type,
                turn_number=turn,
                content=content,
                evidence_id=evidence_id,
                tool_name=tool_name,
                input_order=order,
            )
        )
        previous_turn = turn
    return VisibleBundle(
        root=root,
        input_id=input_id,
        benchmark="feedbacktrace",
        task_document=None,
        repo_artifacts=(),
        trace_events=tuple(events),
        cutoff_turn=cutoff,
        visible_repository_files=0,
        excluded_repository_files=0,
    )


def _resolve_visible_file(root: Path, value: str) -> Path:
    if "\\" in value:
        raise BundleError(f"visible path must use forward slashes: {value!r}")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if posix.is_absolute() or windows.is_absolute() or windows.drive:
        raise BundleError(f"absolute visible path is forbidden: {value!r}")
    if not posix.parts or any(part in {"", ".", ".."} for part in posix.parts):
        raise BundleError(f"non-canonical visible path: {value!r}")
    if posix.as_posix() != value:
        raise BundleError(f"non-canonical visible path: {value!r}")
    candidate = root
    for part in posix.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise BundleError(f"symbolic links are forbidden: {value!r}")
    if not candidate.is_file():
        raise BundleError(f"allowlisted file does not exist: {value!r}")
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise BundleError(f"visible path escapes bundle: {value!r}") from error
    return resolved


def _require_exact_materialization(root: Path, visible_files: set[str]) -> None:
    materialized: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise BundleError("symbolic links are forbidden inside a visible bundle")
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            if relative != "input_manifest.json":
                materialized.add(relative)
    if materialized != visible_files:
        missing = sorted(visible_files - materialized)
        hidden = sorted(materialized - visible_files)
        raise BundleError(
            f"manifest/materialization mismatch: missing={missing[:1]} hidden={hidden[:1]}"
        )


def _read_source_text(path: Path, repo_path: str) -> str:
    try:
        if PurePosixPath(repo_path).suffix.casefold() in {".py", ".pyw"}:
            with tokenize.open(path) as handle:
                return handle.read()
        return path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError, SyntaxError) as error:
        raise BundleError(f"cannot decode production artifact {repo_path}: {error}") from error


def _read_utf8(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise BundleError(f"cannot read UTF-8 text {path.name}: {error}") from error


def _json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise BundleError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise BundleError(f"{label} must be a JSON object")
    return value


def _reject_unknown_keys(value: dict[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise BundleError(f"{label} has unsupported fields: {', '.join(unknown)}")


# ---------------------------------------------------------------------------
# Evidence extraction
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class _Draft:
    path: str
    symbol: str
    line_start: int | None
    line_end: int | None
    content: str
    turn: int | None = None
    event_type: str | None = None
    tool_name: str | None = None
    original_evidence_id: str | None = None
    event_index: int | None = None


def build_evidence(bundle: VisibleBundle) -> list[dict[str, Any]]:
    """Build the module-one Evidence set.

    The output follows the Evidence schema defined by ``EBG.md``.
    """

    drafts: list[_Draft] = []
    source_type = "trace" if bundle.benchmark == "feedbacktrace" else "code"
    if source_type == "trace":
        for event in bundle.trace_events:
            drafts.append(
                _Draft(
                    path="",
                    symbol="",
                    line_start=None,
                    line_end=None,
                    content=event.content,
                    turn=event.turn_number,
                    event_type=event.event_type,
                    tool_name=event.tool_name,
                    original_evidence_id=event.evidence_id,
                    event_index=event.input_order,
                )
            )
    else:
        for artifact in bundle.repo_artifacts:
            drafts.extend(_extract_artifact(artifact))
        drafts.sort(key=_code_draft_key)

    if not drafts:
        raise ExtractionError("Evidence Intake produced no facts")
    nodes: list[dict[str, Any]] = []
    locator_keys: set[tuple[Any, ...]] = set()
    for index, draft in enumerate(drafts, start=1):
        if source_type == "code":
            locator: dict[str, Any] = {
                "path": draft.path,
                "symbol": draft.symbol,
                "line_start": draft.line_start,
                "line_end": draft.line_end,
            }
        else:
            locator = {
                "turn": draft.turn,
                "event_index": draft.event_index,
                "event_type": draft.event_type,
                "tool_name": draft.tool_name,
                "original_evidence_id": draft.original_evidence_id,
            }
        locator_key = (source_type, *locator.values())
        if locator_key in locator_keys:
            raise ExtractionError(f"ambiguous duplicate Evidence locator: {locator_key}")
        locator_keys.add(locator_key)
        nodes.append(
            {
                "evidence_id": f"E{index:06d}",
                "source_type": source_type,
                "locator": locator,
                "content": draft.content,
            }
        )
    validate_evidence(bundle, nodes)
    return nodes


def validate_evidence(
    bundle: VisibleBundle, nodes: list[dict[str, Any]]
) -> None:
    expected_ids = [f"E{index:06d}" for index in range(1, len(nodes) + 1)]
    if [node.get("evidence_id") for node in nodes] != expected_ids:
        raise ExtractionError("Evidence IDs must be contiguous and deterministic")
    expected_type = "trace" if bundle.benchmark == "feedbacktrace" else "code"
    allowed_paths = {artifact.path for artifact in bundle.repo_artifacts}
    artifacts = {artifact.path: artifact for artifact in bundle.repo_artifacts}
    if expected_type == "trace" and len(nodes) != len(bundle.trace_events):
        raise ExtractionError("Trace events were lost or added")
    trace_evidence: list[str] = []
    previous_trace_key = (-1, -1)
    for index, node in enumerate(nodes):
        if set(node) != {"evidence_id", "source_type", "locator", "content"}:
            raise ExtractionError("Evidence has an unexpected field")
        if node["source_type"] != expected_type:
            raise ExtractionError("Evidence source_type is inconsistent")
        locator = node["locator"]
        if not isinstance(node["content"], str):
            raise ExtractionError("Evidence content must be a string")
        if expected_type == "code":
            if set(locator) != {"path", "symbol", "line_start", "line_end"}:
                raise ExtractionError("Code Evidence locator has an unexpected field")
            if locator["path"] not in allowed_paths:
                raise ExtractionError("Code Evidence points outside production artifacts")
            if not isinstance(locator["symbol"], str) or not locator["symbol"]:
                raise ExtractionError("Code Evidence requires one concrete symbol")
            start, end = locator["line_start"], locator["line_end"]
            if not isinstance(start, int) or not isinstance(end, int) or not 1 <= start <= end:
                raise ExtractionError("Code Evidence requires a valid line range")
            artifact = artifacts[locator["path"]]
            source = "".join(artifact.content.splitlines(keepends=True)[start - 1 : end])
            if node["content"] != source:
                raise ExtractionError("Code Evidence content is not the located source substring")
            if not node["content"]:
                raise ExtractionError("Code Evidence cannot be empty")
        else:
            if set(locator) != {
                "turn", "event_index", "event_type", "tool_name", "original_evidence_id"
            }:
                raise ExtractionError("Trace Evidence locator has an unexpected field")
            turn = locator["turn"]
            order = locator["event_index"]
            if not isinstance(turn, int) or turn < 0:
                raise ExtractionError("Trace Evidence requires a non-negative turn")
            if not isinstance(order, int) or order < 0:
                raise ExtractionError("Trace Evidence requires a non-negative event_index")
            if (turn, order) < previous_trace_key:
                raise ExtractionError("Trace Evidence changed input order")
            previous_trace_key = (turn, order)
            event = bundle.trace_events[index]
            if (
                order != event.input_order
                or locator["event_type"] != event.event_type
                or locator["tool_name"] != event.tool_name
                or locator["original_evidence_id"] != event.evidence_id
                or node["content"] != event.content
            ):
                raise ExtractionError("Trace Evidence does not reproduce its input event")
            if locator["original_evidence_id"] is not None:
                trace_evidence.append(locator["original_evidence_id"])
    if expected_type == "trace":
        expected = [
            event.evidence_id
            for event in bundle.trace_events
            if event.evidence_id is not None
        ]
        if trace_evidence != expected or len(nodes) != len(bundle.trace_events):
            raise ExtractionError("Trace events or original evidence IDs were lost or reordered")


def source_symbol_spans(
    bundle: VisibleBundle,
) -> dict[tuple[str, str], list[SymbolSpan]]:
    """Return every concrete definition range used by Graph Assembly."""

    spans: dict[tuple[str, str], list[SymbolSpan]] = {}
    for artifact in bundle.repo_artifacts:
        lines = artifact.content.splitlines(keepends=True)
        if not lines:
            continue
        if _is_python(artifact):
            try:
                tree = parse_python(artifact.content, artifact.path)
            except SyntaxError:
                tree = None
            if tree is not None:
                _append_symbol_span(
                    spans,
                    SymbolSpan(artifact.path, "<module>", 1, len(lines)),
                )
                _collect_python_spans(artifact.path, tree.body, None, spans)
                continue
        if is_supported_source(artifact) and artifact.kind in {"source", "executable"}:
            _append_symbol_span(
                spans,
                SymbolSpan(artifact.path, "<module>", 1, len(lines)),
            )
            for span in generic_symbol_spans(artifact.path, artifact.content):
                _append_symbol_span(spans, span)
            continue
        if artifact.kind in {"source", "executable"}:
            _append_symbol_span(
                spans,
                SymbolSpan(artifact.path, "<module>", 1, len(lines)),
            )
            continue
        if artifact.kind == "runtime_template":
            _append_symbol_span(
                spans,
                SymbolSpan(artifact.path, "<module>", 1, len(lines)),
            )
            for span in _template_symbol_spans(artifact):
                _append_symbol_span(spans, span)
            continue
        drafts = _extract_artifact(artifact)
        for draft in drafts:
            if draft.line_start is None or draft.line_end is None:
                raise ExtractionError(
                    f"repository symbol lacks a concrete range: {artifact.path}"
                )
            _append_symbol_span(
                spans,
                SymbolSpan(
                    artifact.path,
                    draft.symbol,
                    draft.line_start,
                    draft.line_end,
                ),
            )
    return spans


def _extract_artifact(artifact: RepoArtifact) -> list[_Draft]:
    if _is_python(artifact):
        try:
            tree = parse_python(artifact.content, artifact.path)
        except SyntaxError:
            return _whole_artifact_evidence(artifact)
        lines = artifact.content.splitlines(keepends=True)
        drafts: list[_Draft] = []
        _extract_python_body(artifact.path, lines, tree.body, None, drafts)
        return _deduplicate_drafts(drafts)
    if is_supported_source(artifact) and artifact.kind in {"source", "executable"}:
        return _deduplicate_drafts(
            [_Draft(artifact.path, symbol, start, end, content)
             for symbol, start, end, content in generic_source_drafts(artifact)]
        )
    if artifact.kind == "configuration":
        return _extract_config_blocks(artifact)
    if artifact.kind == "runtime_template":
        return _extract_template_blocks(artifact)
    return _whole_artifact_evidence(artifact)


def _extract_python_body(
    path: str,
    lines: list[str],
    body: list[ast.stmt],
    scope: str | None,
    drafts: list[_Draft],
) -> None:
    for statement in body:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if _is_overload_definition(statement):
                continue
            symbol = f"{scope}.{statement.name}" if scope else statement.name
            for decorator in statement.decorator_list:
                _append_line_draft(
                    path,
                    symbol,
                    lines,
                    int(decorator.lineno),
                    int(getattr(decorator, "end_lineno", decorator.lineno)),
                    drafts,
                )
            _append_line_draft(
                path,
                symbol,
                lines,
                int(statement.lineno),
                _header_end(statement, statement.body),
                drafts,
            )
            _extract_python_body(path, lines, statement.body, symbol, drafts)
            continue
        if isinstance(statement, ast.ClassDef):
            symbol = f"{scope}.{statement.name}" if scope else statement.name
            for decorator in statement.decorator_list:
                _append_line_draft(
                    path,
                    symbol,
                    lines,
                    int(decorator.lineno),
                    int(getattr(decorator, "end_lineno", decorator.lineno)),
                    drafts,
                )
            _append_line_draft(
                path,
                symbol,
                lines,
                int(statement.lineno),
                _header_end(statement, statement.body),
                drafts,
            )
            _extract_python_body(path, lines, statement.body, symbol, drafts)
            continue
        current_symbol = scope or "<module>"
        if isinstance(statement, ast.If):
            _append_compound_header(path, current_symbol, lines, statement, statement.body, drafts)
            _extract_python_body(path, lines, statement.body, scope, drafts)
            _append_clause_gap(path, current_symbol, lines, statement.body, statement.orelse, drafts)
            _extract_python_body(path, lines, statement.orelse, scope, drafts)
        elif isinstance(statement, (ast.For, ast.AsyncFor, ast.While)):
            _append_compound_header(path, current_symbol, lines, statement, statement.body, drafts)
            _extract_python_body(path, lines, statement.body, scope, drafts)
            _append_clause_gap(path, current_symbol, lines, statement.body, statement.orelse, drafts)
            _extract_python_body(path, lines, statement.orelse, scope, drafts)
        elif isinstance(statement, (ast.With, ast.AsyncWith)):
            _append_compound_header(path, current_symbol, lines, statement, statement.body, drafts)
            _extract_python_body(path, lines, statement.body, scope, drafts)
        elif isinstance(statement, (ast.Try, ast.TryStar)):
            _append_compound_header(path, current_symbol, lines, statement, statement.body, drafts)
            _extract_python_body(path, lines, statement.body, scope, drafts)
            previous_body: list[ast.stmt] = statement.body
            for handler in statement.handlers:
                _append_clause_gap(path, current_symbol, lines, previous_body, handler.body, drafts)
                _extract_python_body(path, lines, handler.body, scope, drafts)
                previous_body = handler.body
            _append_clause_gap(path, current_symbol, lines, previous_body, statement.orelse, drafts)
            _extract_python_body(path, lines, statement.orelse, scope, drafts)
            final_predecessor = statement.orelse or previous_body
            _append_clause_gap(path, current_symbol, lines, final_predecessor, statement.finalbody, drafts)
            _extract_python_body(path, lines, statement.finalbody, scope, drafts)
        elif isinstance(statement, ast.Match):
            first_case_body = statement.cases[0].body if statement.cases else []
            _append_compound_header(path, current_symbol, lines, statement, first_case_body, drafts)
            for case in statement.cases:
                if case.body:
                    case_start = getattr(case.pattern, "lineno", case.body[0].lineno)
                    case_end = max(case_start, case.body[0].lineno - 1)
                    _append_line_draft(path, current_symbol, lines, case_start, case_end, drafts)
                    _extract_python_body(path, lines, case.body, scope, drafts)
        else:
            _append_line_draft(
                path,
                current_symbol,
                lines,
                _node_start(statement),
                int(getattr(statement, "end_lineno", statement.lineno)),
                drafts,
            )


def _append_compound_header(
    path: str,
    symbol: str,
    lines: list[str],
    node: ast.stmt,
    body: list[ast.stmt],
    drafts: list[_Draft],
) -> None:
    _append_line_draft(path, symbol, lines, _node_start(node), _header_end(node, body), drafts)


def _append_clause_gap(
    path: str,
    symbol: str,
    lines: list[str],
    previous: list[ast.stmt],
    following: list[ast.stmt],
    drafts: list[_Draft],
) -> None:
    if not previous or not following:
        return
    start = int(getattr(previous[-1], "end_lineno", previous[-1].lineno)) + 1
    end = int(following[0].lineno) - 1
    if start <= end and any(lines[index - 1].strip() for index in range(start, end + 1)):
        _append_line_draft(path, symbol, lines, start, end, drafts)


def _append_line_draft(
    path: str,
    symbol: str,
    lines: list[str],
    start: int,
    end: int,
    drafts: list[_Draft],
) -> None:
    if not lines:
        return
    start = max(1, min(start, len(lines)))
    end = max(start, min(end, len(lines)))
    content = "".join(lines[start - 1 : end])
    if content:
        drafts.append(_Draft(path, symbol, start, end, content))


def _extract_config_blocks(artifact: RepoArtifact) -> list[_Draft]:
    """Extract complete configuration items without section-sized overreach."""

    lines = artifact.content.splitlines(keepends=True)
    if not lines:
        return []
    suffix = PurePosixPath(artifact.path).suffix.casefold()
    name = PurePosixPath(artifact.path).name.casefold()
    starts: list[tuple[int, str, int]] = []
    boundaries: list[int] = []
    if suffix in {".ini", ".cfg", ".conf"}:
        section = ""
        for index, line in enumerate(lines, start=1):
            section_match = re.match(r"^\s*\[([^]]+)]", line)
            if section_match:
                section = section_match.group(1).strip()
                boundaries.append(index)
                continue
            key_match = re.match(r"^(\s*)([^#;\s][^=:#]*?)\s*(?:=|:)", line)
            if key_match:
                key = key_match.group(2).strip()
                starts.append((index, f"{section}.{key}" if section else key, len(key_match.group(1))))
    elif suffix == ".json":
        for index, line in enumerate(lines, start=1):
            match = re.match(r'^(\s*)"([^"\\]+)"\s*:', line)
            if match:
                starts.append((index, match.group(2), len(match.group(1))))
        if starts:
            top_indent = min(indent for _, _, indent in starts)
            starts = [item for item in starts if item[2] == top_indent]
    elif suffix == ".toml":
        section = ""
        for index, line in enumerate(lines, start=1):
            section_match = re.match(r"^\s*\[\[?([^]]+)]?]", line)
            if section_match:
                section = section_match.group(1).strip()
                boundaries.append(index)
                continue
            key_match = re.match(r"^(\s*)([A-Za-z0-9_.-]+)\s*=", line)
            if key_match:
                key = key_match.group(2)
                starts.append((index, f"{section}.{key}" if section else key, len(key_match.group(1))))
    elif (
        name.startswith("requirements")
        or "requirements" in (part.casefold() for part in PurePosixPath(artifact.path).parts[:-1])
    ):
        for index, line in enumerate(lines, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("--hash="):
                continue
            key = re.split(r"[<>=!~\[;\s]", stripped, maxsplit=1)[0] or "<module>"
            starts.append((index, key, 0))
    else:
        parents: list[tuple[int, str]] = []
        for index, line in enumerate(lines, start=1):
            match = re.match(r"^(\s*)(?!#)([A-Za-z0-9_.-]+)\s*:\s*(.*)$", line)
            if match:
                indent = len(match.group(1))
                while parents and parents[-1][0] >= indent:
                    parents.pop()
                key = match.group(2)
                value = match.group(3).strip()
                qualified = ".".join([*(item[1] for item in parents), key])
                if value:
                    starts.append((index, qualified, indent))
                else:
                    parents.append((indent, key))
    if not starts:
        return _whole_artifact_evidence(artifact)
    drafts: list[_Draft] = []
    for index, (start, key, indent) in enumerate(starts):
        end = len(lines)
        next_boundary = next((line for line in boundaries if line > start), None)
        if next_boundary is not None:
            end = next_boundary - 1
        for next_start, _, next_indent in starts[index + 1 :]:
            if next_indent <= indent:
                end = min(end, next_start - 1)
                break
        while end >= start:
            trailing = lines[end - 1]
            trailing_indent = len(trailing) - len(trailing.lstrip())
            if trailing.strip() in {"}", "]", "},", "],"} and trailing_indent < indent:
                end -= 1
                continue
            break
        while end >= start and not lines[end - 1].strip():
            end -= 1
        _append_line_draft(artifact.path, key or "<module>", lines, start, end, drafts)
    return _deduplicate_drafts(drafts)


def _extract_template_blocks(artifact: RepoArtifact) -> list[_Draft]:
    lines = artifact.content.splitlines(keepends=True)
    drafts: list[_Draft] = []
    scopes: list[str] = []
    open_pattern = re.compile(r"\{%\s*(?:block|macro)\s+([A-Za-z_][\w.-]*)")
    close_pattern = re.compile(r"\{%\s*end(?:block|macro)\b")
    for index, line in enumerate(lines, start=1):
        match = open_pattern.search(line)
        if match:
            scopes.append(match.group(1))
        if line.strip():
            _append_line_draft(
                artifact.path,
                ".".join(scopes) if scopes else "<module>",
                lines,
                index,
                index,
                drafts,
            )
        if close_pattern.search(line) and scopes:
            scopes.pop()
    return _deduplicate_drafts(drafts)


def _template_symbol_spans(artifact: RepoArtifact) -> list[SymbolSpan]:
    lines = artifact.content.splitlines(keepends=True)
    open_pattern = re.compile(r"\{%\s*(?:block|macro)\s+([A-Za-z_][\w.-]*)")
    close_pattern = re.compile(r"\{%\s*end(?:block|macro)\b")
    stack: list[tuple[str, int]] = []
    spans: list[SymbolSpan] = []
    for index, line in enumerate(lines, start=1):
        match = open_pattern.search(line)
        if match:
            parent = stack[-1][0] if stack else ""
            symbol = f"{parent}.{match.group(1)}" if parent else match.group(1)
            stack.append((symbol, index))
        if close_pattern.search(line) and stack:
            symbol, start = stack.pop()
            spans.append(SymbolSpan(artifact.path, symbol, start, index))
    last_line = max(1, len(lines))
    while stack:
        symbol, start = stack.pop()
        spans.append(SymbolSpan(artifact.path, symbol, start, last_line))
    return sorted(spans, key=lambda item: (item.line_start, item.line_end, item.symbol))


def _whole_artifact_evidence(artifact: RepoArtifact) -> list[_Draft]:
    """Represent one non-Python artifact as one module-scoped Evidence item."""

    lines = artifact.content.splitlines(keepends=True)
    if not lines:
        return []
    return [
        _Draft(
            path=artifact.path,
            symbol="<module>",
            line_start=1,
            line_end=len(lines),
            content=artifact.content,
        )
    ]


def _deduplicate_drafts(drafts: Iterable[_Draft]) -> list[_Draft]:
    by_range: dict[tuple[str, str, int | None, int | None], _Draft] = {}
    for draft in drafts:
        key = (draft.path, draft.symbol, draft.line_start, draft.line_end)
        existing = by_range.get(key)
        if existing is None:
            by_range[key] = draft
        elif existing.content != draft.content:
            raise ExtractionError(f"same Evidence range produced different text: {key}")
    return sorted(by_range.values(), key=_code_draft_key)


def _code_draft_key(draft: _Draft) -> tuple[Any, ...]:
    return (
        draft.path.casefold(),
        draft.path,
        draft.line_start or 0,
        draft.line_end or 0,
        draft.symbol.casefold(),
        draft.symbol,
    )


def _is_python(artifact: RepoArtifact) -> bool:
    suffix = PurePosixPath(artifact.path).suffix.casefold()
    first_line = artifact.content.splitlines()[0] if artifact.content.splitlines() else ""
    return suffix in {".py", ".pyw"} or (first_line.startswith("#!") and "python" in first_line.casefold())


def _node_start(node: ast.AST) -> int:
    decorators = getattr(node, "decorator_list", [])
    starts = [int(getattr(node, "lineno", 1))]
    starts.extend(int(decorator.lineno) for decorator in decorators)
    return min(starts)


def _header_end(node: ast.AST, body: list[ast.stmt]) -> int:
    start = int(getattr(node, "lineno", 1))
    if body:
        return max(start, int(body[0].lineno) - 1)
    return int(getattr(node, "end_lineno", start))


def _collect_python_spans(
    path: str,
    body: list[ast.stmt],
    scope: str | None,
    spans: dict[tuple[str, str], list[SymbolSpan]],
) -> None:
    for statement in body:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)) and _is_overload_definition(statement):
                continue
            symbol = f"{scope}.{statement.name}" if scope else statement.name
            _append_symbol_span(
                spans,
                SymbolSpan(
                    path=path,
                    symbol=symbol,
                    line_start=_node_start(statement),
                    line_end=int(getattr(statement, "end_lineno", statement.lineno)),
                ),
            )
            _collect_python_spans(path, statement.body, symbol, spans)
            continue
        for nested_body in _nested_statement_bodies(statement):
            _collect_python_spans(path, nested_body, scope, spans)


def _append_symbol_span(
    spans: dict[tuple[str, str], list[SymbolSpan]], span: SymbolSpan
) -> None:
    spans.setdefault((span.path, span.symbol), []).append(span)


def _nested_statement_bodies(statement: ast.stmt) -> list[list[ast.stmt]]:
    bodies: list[list[ast.stmt]] = []
    for _, value in ast.iter_fields(statement):
        if isinstance(value, list) and value and all(
            isinstance(item, ast.stmt) for item in value
        ):
            bodies.append(value)
        elif isinstance(value, ast.ExceptHandler):
            bodies.append(value.body)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, ast.ExceptHandler):
                    bodies.append(item.body)
                elif isinstance(item, ast.match_case):
                    bodies.append(item.body)
    return bodies


def _is_overload_definition(
    statement: ast.FunctionDef | ast.AsyncFunctionDef,
) -> bool:
    return any(
        _decorator_label(decorator).rsplit(".", 1)[-1] == "overload"
        for decorator in statement.decorator_list
    )


def _decorator_label(node: ast.AST) -> str:
    if isinstance(node, ast.Call):
        return _decorator_label(node.func)
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _decorator_label(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""

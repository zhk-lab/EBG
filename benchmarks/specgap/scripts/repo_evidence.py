"""Safe repository snapshots and deterministic full-repository evidence indexing.

This module intentionally has no command-line interface.  It provides the
repository-facing building blocks used by the SpecGAP generators:

* download an immutable GitHub commit from ``codeload.github.com``;
* extract it without trusting tar paths, links, or special files;
* compute a deterministic content-only tree hash;
* turn implementation, test, fixture, configuration, build-metadata, and
  documentation files into line-addressable snippets.

``EvidenceSnippet.excerpt`` is always unnumbered source text copied directly
from the decoded file.  Line numbers are added only by
``format_snippets_for_llm`` so consumers can hash or quote the raw excerpt.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shutil
import stat
import tarfile
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Mapping, Sequence
from urllib.parse import urlsplit

import requests


TREE_HASH_ALGORITHM = "sha256-length-prefixed-path-and-content-v1"
SNAPSHOT_MARKER_VERSION = 1

DEFAULT_MAX_DOWNLOAD_BYTES = 512 * 1024 * 1024
DEFAULT_MAX_UNPACKED_BYTES = 2 * 1024 * 1024 * 1024
DEFAULT_MAX_ARCHIVE_MEMBERS = 200_000
DEFAULT_MAX_EVIDENCE_FILE_BYTES = 4 * 1024 * 1024
DEFAULT_MAX_EVIDENCE_FILES = 50_000
DEFAULT_CHUNK_LINES = 80
DEFAULT_CHUNK_OVERLAP = 12

_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40}$")
_GITHUB_COMPONENT_RE = re.compile(r"^[A-Za-z0-9_.-]+$")

_IGNORED_DIRECTORY_NAMES = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".tox",
        ".nox",
        ".hypothesis",
        ".cache",
        ".coverage",
        ".eggs",
        ".idea",
        ".vscode",
        ".venv",
        "venv",
        "env",
        "node_modules",
        "site-packages",
        "dist",
        "build",
        "target",
        "coverage",
        "htmlcov",
    }
)

_SOURCE_EXTENSIONS = frozenset(
    {
        ".py",
        ".pyi",
        ".pyx",
        ".pxd",
        ".js",
        ".jsx",
        ".mjs",
        ".cjs",
        ".ts",
        ".tsx",
        ".java",
        ".go",
        ".rs",
        ".rb",
        ".php",
        ".c",
        ".h",
        ".cc",
        ".cpp",
        ".cxx",
        ".hpp",
        ".cs",
        ".swift",
        ".kt",
        ".kts",
        ".scala",
        ".sh",
        ".bash",
        ".zsh",
        ".fish",
        ".ps1",
        ".sql",
        ".r",
        ".lua",
        ".pl",
        ".pm",
        ".ex",
        ".exs",
        ".erl",
        ".hrl",
        ".fs",
        ".fsx",
        ".clj",
        ".cljs",
        ".groovy",
        ".vue",
        ".svelte",
    }
)

_CONFIG_EXTENSIONS = frozenset(
    {
        ".json",
        ".jsonc",
        ".toml",
        ".yaml",
        ".yml",
        ".ini",
        ".cfg",
        ".conf",
        ".properties",
        ".xml",
        ".gradle",
        ".cmake",
        ".mk",
        ".lock",
        ".txt",
    }
)

_DOCUMENTATION_EXTENSIONS = frozenset(
    {
        ".md",
        ".mdx",
        ".rst",
        ".adoc",
        ".asciidoc",
    }
)

_FIXTURE_DIRECTORY_NAMES = frozenset(
    {
        "fixture",
        "fixtures",
        "testdata",
        "test_data",
        "test-data",
        "snapshots",
        "__snapshots__",
    }
)

_BUILD_METADATA_FILENAMES = frozenset(
    {
        "pyproject.toml",
        "setup.py",
        "setup.cfg",
        "requirements.txt",
        "requirements-dev.txt",
        "pipfile",
        "pipfile.lock",
        "poetry.lock",
        "package.json",
        "cargo.toml",
        "cargo.lock",
        "go.mod",
        "go.sum",
        "gemfile",
        "gemfile.lock",
        "composer.json",
        "dockerfile",
        "makefile",
        "cmakelists.txt",
    }
)

_CONFIG_FILENAMES = frozenset(
    {
        "pyproject.toml",
        "setup.py",
        "setup.cfg",
        "tox.ini",
        "pytest.ini",
        "mypy.ini",
        "ruff.toml",
        ".ruff.toml",
        ".flake8",
        ".editorconfig",
        ".gitignore",
        ".gitattributes",
        ".dockerignore",
        "requirements.txt",
        "requirements-dev.txt",
        "pipfile",
        "pipfile.lock",
        "poetry.lock",
        "package.json",
        "pnpm-workspace.yaml",
        "yarn.lock",
        "cargo.toml",
        "cargo.lock",
        "go.mod",
        "go.sum",
        "gemfile",
        "gemfile.lock",
        "composer.json",
        "dockerfile",
        "makefile",
        "cmakelists.txt",
        "procfile",
    }
)

_EXCLUDED_EVIDENCE_FILENAMES = frozenset(
    {
        "package-lock.json",
        "npm-shrinkwrap.json",
        "pnpm-lock.yaml",
    }
)

_LANGUAGE_BY_EXTENSION = {
    ".py": "python",
    ".pyi": "python",
    ".pyx": "python",
    ".pxd": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".php": "php",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".swift": "swift",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".scala": "scala",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
    ".fish": "shell",
    ".ps1": "powershell",
    ".sql": "sql",
    ".r": "r",
    ".lua": "lua",
    ".pl": "perl",
    ".pm": "perl",
    ".json": "json",
    ".jsonc": "json",
    ".toml": "toml",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".ini": "ini",
    ".cfg": "ini",
    ".xml": "xml",
}

class RepositorySnapshotError(RuntimeError):
    """Base error for repository snapshot failures."""


class UnsafeArchiveError(RepositorySnapshotError):
    """Raised when an archive contains an unsafe or unsupported member."""


class SnapshotIntegrityError(RepositorySnapshotError):
    """Raised when an existing or downloaded snapshot fails integrity checks."""


class EvidenceExtractionError(RuntimeError):
    """Raised when a repository cannot be indexed safely and deterministically."""


@dataclass(frozen=True, slots=True)
class GitHubRepository:
    """A validated GitHub repository identity."""

    owner: str
    name: str

    @property
    def canonical_url(self) -> str:
        return f"https://github.com/{self.owner}/{self.name}"

    def codeload_url(self, commit_sha: str) -> str:
        return (
            f"https://codeload.github.com/{self.owner}/{self.name}"
            f"/tar.gz/{commit_sha}"
        )


@dataclass(frozen=True, slots=True)
class SnapshotInfo:
    """Metadata for a verified, gitless repository snapshot."""

    repo_url: str
    owner: str
    repo: str
    commit_sha: str
    codeload_url: str
    destination: str
    tree_sha256: str
    tree_hash_algorithm: str
    file_count: int
    total_bytes: int
    archive_sha256: str | None = None
    reused: bool = False
    marker_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EvidenceSnippet:
    """A line-addressable excerpt copied from a repository file."""

    file_path: str
    file_sha256: str
    evidence_type: str
    language: str
    symbol_kind: str
    qualified_name: str | None
    start_line: int
    end_line: int
    excerpt: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def numbered_excerpt(self) -> str:
        return add_line_numbers(self.excerpt, self.start_line)


def parse_github_repo_url(repo_url: str) -> GitHubRepository:
    """Validate a GitHub repository URL and return its owner/name.

    Accepted forms are ``https://github.com/OWNER/REPO`` (optionally ending in
    ``.git`` or ``/``) and the common read-only SSH form
    ``git@github.com:OWNER/REPO.git``.  Subpages, credentials, query strings,
    fragments, percent escapes, and non-GitHub hosts are rejected.
    """

    raw = str(repo_url).strip()
    if not raw:
        raise ValueError("GitHub repository URL is empty")

    if raw.startswith("git@github.com:"):
        path = raw[len("git@github.com:") :]
    else:
        parsed = urlsplit(raw)
        if parsed.scheme.lower() not in {"https", "http"}:
            raise ValueError("GitHub repository URL must use http(s) or git@github.com")
        if parsed.hostname is None or parsed.hostname.lower() not in {
            "github.com",
            "www.github.com",
        }:
            raise ValueError("repository URL host must be github.com")
        if parsed.username or parsed.password or parsed.port:
            raise ValueError("repository URL must not contain credentials or a port")
        if parsed.query or parsed.fragment:
            raise ValueError("repository URL must not contain a query or fragment")
        if "%" in parsed.path:
            raise ValueError("percent-encoded repository URL paths are not accepted")
        path = parsed.path

    path = path.strip("/")
    if path.endswith(".git"):
        path = path[:-4]
    parts = path.split("/")
    if len(parts) != 2 or not all(parts):
        raise ValueError("GitHub repository URL must identify exactly OWNER/REPO")
    owner, repo = parts
    if any(
        component in {".", ".."}
        or not _GITHUB_COMPONENT_RE.fullmatch(component)
        for component in (owner, repo)
    ):
        raise ValueError("invalid GitHub owner or repository name")
    return GitHubRepository(owner=owner, name=repo)


def validate_commit_sha(commit_sha: str) -> str:
    """Return a normalized commit hash, requiring all 40 hexadecimal digits."""

    normalized = str(commit_sha).strip().lower()
    if not _COMMIT_RE.fullmatch(normalized):
        raise ValueError("commit_sha must contain exactly 40 hexadecimal characters")
    return normalized


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA-256 for one regular, non-symlink file."""

    file_path = Path(path)
    if file_path.is_symlink():
        raise ValueError(f"refusing to hash symlink: {file_path}")
    try:
        before = file_path.stat()
    except OSError as exc:
        raise ValueError(f"cannot stat file: {file_path}") from exc
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"not a regular file: {file_path}")

    digest = hashlib.sha256()
    bytes_read = 0
    with file_path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
            bytes_read += len(chunk)
        after = os.fstat(handle.fileno())

    if (
        bytes_read != before.st_size
        or after.st_size != before.st_size
        or after.st_mtime_ns != before.st_mtime_ns
    ):
        raise SnapshotIntegrityError(f"file changed while hashing: {file_path}")
    return digest.hexdigest()


def _iter_tree_files(root: Path) -> list[tuple[bytes, str, Path]]:
    if root.is_symlink():
        raise SnapshotIntegrityError(f"snapshot root is a symlink: {root}")
    if not root.is_dir():
        raise ValueError(f"snapshot root is not a directory: {root}")

    entries: list[tuple[bytes, str, Path]] = []
    for current, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        retained_dirs: list[str] = []
        for dirname in sorted(dirnames):
            child = current_path / dirname
            if child.is_symlink():
                raise SnapshotIntegrityError(
                    f"snapshot contains a directory symlink: {child.relative_to(root)}"
                )
            if dirname == ".git":
                continue
            retained_dirs.append(dirname)
        dirnames[:] = retained_dirs

        for filename in sorted(filenames):
            child = current_path / filename
            relative = child.relative_to(root).as_posix()
            if child.is_symlink():
                raise SnapshotIntegrityError(
                    f"snapshot contains a file symlink: {relative}"
                )
            try:
                mode = child.stat().st_mode
            except OSError as exc:
                raise SnapshotIntegrityError(f"cannot stat snapshot entry: {relative}") from exc
            if not stat.S_ISREG(mode):
                raise SnapshotIntegrityError(
                    f"snapshot contains a non-regular file: {relative}"
                )
            path_bytes = relative.encode("utf-8")
            entries.append((path_bytes, relative, child))

    entries.sort(key=lambda item: item[0])
    return entries


def _tree_stats(root: str | os.PathLike[str]) -> tuple[str, int, int]:
    """Return ``(tree_sha256, file_count, total_bytes)`` for ``root``."""

    root_path = Path(root)
    digest = hashlib.sha256()
    total_bytes = 0
    entries = _iter_tree_files(root_path)

    for path_bytes, relative, file_path in entries:
        before = file_path.stat()
        digest.update(len(path_bytes).to_bytes(8, "big"))
        digest.update(path_bytes)
        digest.update(before.st_size.to_bytes(8, "big"))

        bytes_read = 0
        with file_path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                bytes_read += len(chunk)
            after = os.fstat(handle.fileno())

        if (
            bytes_read != before.st_size
            or after.st_size != before.st_size
            or after.st_mtime_ns != before.st_mtime_ns
        ):
            raise SnapshotIntegrityError(
                f"file changed while computing tree hash: {relative}"
            )
        total_bytes += bytes_read

    return digest.hexdigest(), len(entries), total_bytes


def tree_sha256(root: str | os.PathLike[str]) -> str:
    """Compute the deterministic SpecGAP tree SHA-256.

    Files are ordered by UTF-8 encoded POSIX relative path.  For each regular
    file the digest receives:

    ``uint64_be(len(path)) || path || uint64_be(len(content)) || content``.

    Directories, mtimes, and permission bits are deliberately excluded.  A
    ``.git`` directory is ignored; every symbolic link or special file causes
    an error.
    """

    return _tree_stats(root)[0]


def _default_marker_path(destination: Path) -> Path:
    return destination.parent / f".{destination.name}.snapshot.json"


def _read_marker(marker_path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _write_marker_atomic(marker_path: Path, payload: Mapping[str, Any]) -> None:
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{marker_path.name}.tmp-",
        dir=marker_path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(dict(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, marker_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _validated_existing_snapshot(
    *,
    repo: GitHubRepository,
    commit_sha: str,
    destination: Path,
    codeload_url: str,
    marker_path: Path,
    expected_tree_sha256: str | None,
) -> SnapshotInfo | None:
    if not destination.exists():
        return None
    if not destination.is_dir() or destination.is_symlink():
        raise SnapshotIntegrityError(
            f"existing snapshot target is not a regular directory: {destination}"
        )

    marker = _read_marker(marker_path)
    marker_matches = bool(
        marker
        and marker.get("version") == SNAPSHOT_MARKER_VERSION
        and marker.get("repo_url") == repo.canonical_url
        and marker.get("commit_sha") == commit_sha
        and marker.get("tree_hash_algorithm") == TREE_HASH_ALGORITHM
        and isinstance(marker.get("tree_sha256"), str)
    )
    trusted_hash = (
        expected_tree_sha256.lower()
        if expected_tree_sha256 is not None
        else str(marker["tree_sha256"]).lower()
        if marker_matches and marker is not None
        else None
    )
    if trusted_hash is None:
        return None
    if not re.fullmatch(r"[0-9a-f]{64}", trusted_hash):
        raise SnapshotIntegrityError("expected tree SHA-256 is malformed")

    actual_hash, file_count, total_bytes = _tree_stats(destination)
    if actual_hash != trusted_hash:
        raise SnapshotIntegrityError(
            f"existing snapshot tree hash mismatch: expected {trusted_hash}, "
            f"found {actual_hash}"
        )
    if expected_tree_sha256 is not None and marker_matches and marker is not None:
        marker_hash = str(marker["tree_sha256"]).lower()
        if marker_hash != trusted_hash:
            raise SnapshotIntegrityError(
                "snapshot marker and expected tree SHA-256 disagree"
            )

    return SnapshotInfo(
        repo_url=repo.canonical_url,
        owner=repo.owner,
        repo=repo.name,
        commit_sha=commit_sha,
        codeload_url=codeload_url,
        destination=str(destination),
        tree_sha256=actual_hash,
        tree_hash_algorithm=TREE_HASH_ALGORITHM,
        file_count=file_count,
        total_bytes=total_bytes,
        archive_sha256=(
            str(marker.get("archive_sha256"))
            if marker_matches and marker and marker.get("archive_sha256")
            else None
        ),
        reused=True,
        marker_path=str(marker_path),
    )


def _download_archive(
    *,
    url: str,
    output_path: Path,
    timeout: tuple[float, float] | float,
    max_download_bytes: int,
    chunk_size: int,
    session: requests.Session | None,
) -> str:
    client: Any = session if session is not None else requests
    response: requests.Response | None = None
    try:
        response = client.get(
            url,
            stream=True,
            timeout=timeout,
            headers={
                "Accept": "application/gzip, application/octet-stream",
                "Accept-Encoding": "identity",
                "User-Agent": "AgentVerifyBench-SpecGAP/1.0",
            },
            allow_redirects=True,
        )
        if response.status_code != 200:
            preview = response.text[:500] if hasattr(response, "text") else ""
            raise RepositorySnapshotError(
                f"codeload returned HTTP {response.status_code}: {preview}"
            )
        content_length = response.headers.get("Content-Length")
        if content_length:
            try:
                declared_size = int(content_length)
            except ValueError as exc:
                raise RepositorySnapshotError(
                    "codeload returned an invalid Content-Length"
                ) from exc
            if declared_size > max_download_bytes:
                raise RepositorySnapshotError(
                    f"archive exceeds compressed-size limit ({declared_size} > "
                    f"{max_download_bytes})"
                )

        digest = hashlib.sha256()
        downloaded = 0
        with output_path.open("xb") as handle:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                downloaded += len(chunk)
                if downloaded > max_download_bytes:
                    raise RepositorySnapshotError(
                        f"archive exceeds compressed-size limit ({max_download_bytes})"
                    )
                handle.write(chunk)
                digest.update(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        if downloaded == 0:
            raise RepositorySnapshotError("codeload returned an empty archive")
        return digest.hexdigest()
    except requests.RequestException as exc:
        raise RepositorySnapshotError(f"failed to download {url}: {exc}") from exc
    finally:
        if response is not None:
            response.close()


def _safe_tar_parts(member_name: str) -> tuple[str, ...]:
    if not member_name or "\x00" in member_name or "\\" in member_name:
        raise UnsafeArchiveError(f"unsafe tar member name: {member_name!r}")
    pure_path = PurePosixPath(member_name)
    if pure_path.is_absolute():
        raise UnsafeArchiveError(f"absolute tar member path: {member_name!r}")
    parts = tuple(part for part in pure_path.parts if part not in {"", "."})
    if not parts or any(part == ".." for part in parts):
        raise UnsafeArchiveError(f"traversing tar member path: {member_name!r}")
    return parts


def _extract_codeload_archive(
    archive_path: Path,
    output_root: Path,
    *,
    max_unpacked_bytes: int,
    max_members: int,
) -> None:
    try:
        archive = tarfile.open(archive_path, mode="r:gz")
    except (OSError, tarfile.TarError) as exc:
        raise UnsafeArchiveError("download is not a valid gzip-compressed tar archive") from exc

    with archive:
        try:
            members = archive.getmembers()
        except (OSError, tarfile.TarError) as exc:
            raise UnsafeArchiveError("cannot read tar member table") from exc
        if not members:
            raise UnsafeArchiveError("archive has no members")
        if len(members) > max_members:
            raise UnsafeArchiveError(
                f"archive member limit exceeded ({len(members)} > {max_members})"
            )

        validated: list[tuple[tarfile.TarInfo, tuple[str, ...]]] = []
        wrapper_names: set[str] = set()
        seen_paths: set[tuple[str, ...]] = set()
        file_paths: set[tuple[str, ...]] = set()
        total_size = 0

        for member in members:
            parts = _safe_tar_parts(member.name)
            wrapper_names.add(parts[0])
            if len(wrapper_names) > 1:
                raise UnsafeArchiveError(
                    "codeload archive does not have one top-level wrapper directory"
                )
            relative_parts = parts[1:]

            if member.issym() or member.islnk():
                raise UnsafeArchiveError(
                    f"archive links are not allowed: {member.name!r}"
                )
            if not (member.isdir() or member.isfile()):
                raise UnsafeArchiveError(
                    f"unsupported tar member type: {member.name!r}"
                )
            if not relative_parts:
                if not member.isdir():
                    raise UnsafeArchiveError(
                        "top-level codeload wrapper must be a directory"
                    )
                continue
            if ".git" in relative_parts:
                raise UnsafeArchiveError(
                    f"archive unexpectedly contains .git metadata: {member.name!r}"
                )
            if relative_parts in seen_paths:
                raise UnsafeArchiveError(f"duplicate tar member path: {member.name!r}")
            seen_paths.add(relative_parts)

            for index in range(1, len(relative_parts)):
                if relative_parts[:index] in file_paths:
                    raise UnsafeArchiveError(
                        f"tar file used as a parent directory: {member.name!r}"
                    )
            if member.isfile():
                if any(
                    existing[: len(relative_parts)] == relative_parts
                    for existing in seen_paths
                    if len(existing) > len(relative_parts)
                ):
                    raise UnsafeArchiveError(
                        f"tar file conflicts with a child path: {member.name!r}"
                    )
                if member.size < 0:
                    raise UnsafeArchiveError(
                        f"tar member has a negative size: {member.name!r}"
                    )
                total_size += member.size
                if total_size > max_unpacked_bytes:
                    raise UnsafeArchiveError(
                        f"archive exceeds unpacked-size limit ({max_unpacked_bytes})"
                    )
                file_paths.add(relative_parts)
            validated.append((member, relative_parts))

        if len(wrapper_names) != 1:
            raise UnsafeArchiveError("archive has no top-level wrapper")

        output_root.mkdir(parents=True, exist_ok=False)
        resolved_root = output_root.resolve()

        for member, relative_parts in validated:
            destination = output_root.joinpath(*relative_parts)
            resolved_destination = destination.resolve(strict=False)
            try:
                resolved_destination.relative_to(resolved_root)
            except ValueError as exc:
                raise UnsafeArchiveError(
                    f"tar member escapes output directory: {member.name!r}"
                ) from exc

            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
                destination.chmod(0o755)
                continue

            destination.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise UnsafeArchiveError(
                    f"cannot read regular tar member: {member.name!r}"
                )
            copied = 0
            try:
                with destination.open("xb") as target:
                    while copied < member.size:
                        chunk = source.read(min(1024 * 1024, member.size - copied))
                        if not chunk:
                            break
                        target.write(chunk)
                        copied += len(chunk)
                    extra = source.read(1)
                    target.flush()
                    os.fsync(target.fileno())
            finally:
                source.close()
            if copied != member.size or extra:
                raise UnsafeArchiveError(
                    f"tar member size mismatch: {member.name!r}"
                )
            destination.chmod(0o755 if member.mode & 0o111 else 0o644)


def download_github_snapshot(
    repo_url: str,
    commit_sha: str,
    destination: str | os.PathLike[str],
    *,
    expected_tree_sha256: str | None = None,
    reuse_existing: bool = True,
    marker_path: str | os.PathLike[str] | None = None,
    timeout: tuple[float, float] | float = (20.0, 180.0),
    chunk_size: int = 1024 * 1024,
    max_download_bytes: int = DEFAULT_MAX_DOWNLOAD_BYTES,
    max_unpacked_bytes: int = DEFAULT_MAX_UNPACKED_BYTES,
    max_members: int = DEFAULT_MAX_ARCHIVE_MEMBERS,
    session: requests.Session | None = None,
) -> SnapshotInfo:
    """Download and atomically install a safe, gitless GitHub snapshot.

    An existing target is reused only when either ``expected_tree_sha256`` is
    supplied or an adjacent, matching sidecar marker exists, and a fresh tree
    hash agrees with that trusted value.  No marker is ever written inside the
    repository snapshot.

    Extraction occurs in a temporary sibling directory.  The final rename is
    therefore atomic on the destination filesystem.  Existing unverified
    targets are never overwritten.
    """

    repo = parse_github_repo_url(repo_url)
    commit = validate_commit_sha(commit_sha)
    destination_path = Path(destination).expanduser().resolve(strict=False)
    if destination_path.name in {"", ".", ".."}:
        raise ValueError("destination must identify a concrete directory")
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    sidecar_path = (
        Path(marker_path).expanduser().resolve(strict=False)
        if marker_path is not None
        else _default_marker_path(destination_path)
    )
    try:
        sidecar_path.relative_to(destination_path)
    except ValueError:
        pass
    else:
        raise ValueError("snapshot marker must be outside the repository directory")

    expected_hash = (
        str(expected_tree_sha256).strip().lower()
        if expected_tree_sha256 is not None
        else None
    )
    if expected_hash is not None and not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
        raise ValueError("expected_tree_sha256 must contain 64 hexadecimal characters")

    codeload_url = repo.codeload_url(commit)
    if destination_path.exists():
        if not reuse_existing:
            raise FileExistsError(f"snapshot destination already exists: {destination_path}")
        existing = _validated_existing_snapshot(
            repo=repo,
            commit_sha=commit,
            destination=destination_path,
            codeload_url=codeload_url,
            marker_path=sidecar_path,
            expected_tree_sha256=expected_hash,
        )
        if existing is None:
            raise SnapshotIntegrityError(
                "existing snapshot has no trusted matching marker or expected tree hash: "
                f"{destination_path}"
            )
        return existing

    staging_path = Path(
        tempfile.mkdtemp(
            prefix=f".{destination_path.name}.tmp-",
            dir=destination_path.parent,
        )
    )
    archive_path = staging_path / "snapshot.tar.gz"
    payload_path = staging_path / "snapshot"
    installed = False
    try:
        archive_hash = _download_archive(
            url=codeload_url,
            output_path=archive_path,
            timeout=timeout,
            max_download_bytes=max_download_bytes,
            chunk_size=chunk_size,
            session=session,
        )
        _extract_codeload_archive(
            archive_path,
            payload_path,
            max_unpacked_bytes=max_unpacked_bytes,
            max_members=max_members,
        )
        actual_hash, file_count, total_bytes = _tree_stats(payload_path)
        if expected_hash is not None and actual_hash != expected_hash:
            raise SnapshotIntegrityError(
                f"downloaded snapshot tree hash mismatch: expected {expected_hash}, "
                f"found {actual_hash}"
            )
        if destination_path.exists():
            raise FileExistsError(
                f"snapshot destination appeared during download: {destination_path}"
            )

        os.replace(payload_path, destination_path)
        installed = True
        marker_payload = {
            "version": SNAPSHOT_MARKER_VERSION,
            "repo_url": repo.canonical_url,
            "owner": repo.owner,
            "repo": repo.name,
            "commit_sha": commit,
            "codeload_url": codeload_url,
            "tree_sha256": actual_hash,
            "tree_hash_algorithm": TREE_HASH_ALGORITHM,
            "file_count": file_count,
            "total_bytes": total_bytes,
            "archive_sha256": archive_hash,
        }
        _write_marker_atomic(sidecar_path, marker_payload)
        return SnapshotInfo(
            repo_url=repo.canonical_url,
            owner=repo.owner,
            repo=repo.name,
            commit_sha=commit,
            codeload_url=codeload_url,
            destination=str(destination_path),
            tree_sha256=actual_hash,
            tree_hash_algorithm=TREE_HASH_ALGORITHM,
            file_count=file_count,
            total_bytes=total_bytes,
            archive_sha256=archive_hash,
            reused=False,
            marker_path=str(sidecar_path),
        )
    except BaseException:
        # If the atomic rename succeeded, retain the verified repository.  A
        # marker-write failure should not silently delete a complete snapshot.
        if installed:
            raise
        raise
    finally:
        shutil.rmtree(staging_path, ignore_errors=True)


def extract_line_excerpt(text: str, start_line: int, end_line: int) -> str:
    """Return inclusive 1-based lines without changing their original endings."""

    if start_line < 1:
        raise ValueError("start_line must be at least 1")
    if end_line < start_line:
        raise ValueError("end_line must not precede start_line")
    lines = text.splitlines(keepends=True)
    if start_line > len(lines):
        raise ValueError(
            f"start_line {start_line} is beyond the text's {len(lines)} lines"
        )
    return "".join(lines[start_line - 1 : end_line])


def excerpt_from_file(
    path: str | os.PathLike[str],
    start_line: int,
    end_line: int,
    *,
    encoding: str = "utf-8",
) -> str:
    """Read and return an inclusive, unnumbered excerpt from a text file."""

    return extract_line_excerpt(
        Path(path).read_text(encoding=encoding),
        start_line,
        end_line,
    )


def add_line_numbers(excerpt: str, start_line: int = 1) -> str:
    """Add display-only line numbers to an unnumbered excerpt."""

    if start_line < 1:
        raise ValueError("start_line must be at least 1")
    lines = excerpt.splitlines()
    if not lines:
        return ""
    width = len(str(start_line + len(lines) - 1))
    return "\n".join(
        f"{line_number:>{width}} | {line}"
        for line_number, line in enumerate(lines, start=start_line)
    )


def _is_test_path(relative_path: PurePosixPath) -> bool:
    lower_parts = tuple(part.lower() for part in relative_path.parts)
    basename = lower_parts[-1]
    stem = PurePosixPath(basename).stem
    return bool(
        any(
            part in {"test", "tests", "testing", "spec", "specs", "__tests__"}
            for part in lower_parts[:-1]
        )
        or basename == "conftest.py"
        or stem.startswith("test_")
        or stem.endswith("_test")
        or stem.endswith("_tests")
        or basename.endswith(".spec.js")
        or basename.endswith(".spec.ts")
        or basename.endswith(".test.js")
        or basename.endswith(".test.ts")
    )


def _is_config_path(relative_path: PurePosixPath) -> bool:
    lower_parts = tuple(part.lower() for part in relative_path.parts)
    basename = lower_parts[-1]
    return bool(
        basename in _CONFIG_FILENAMES
        or relative_path.suffix.lower() in _CONFIG_EXTENSIONS
        or ".github" in lower_parts
        or "config" in lower_parts[:-1]
        or "configs" in lower_parts[:-1]
    )


def _is_fixture_path(relative_path: PurePosixPath) -> bool:
    lower_parts = tuple(part.lower() for part in relative_path.parts)
    basename = lower_parts[-1]
    return bool(
        basename == "conftest.py"
        or any(
            part in _FIXTURE_DIRECTORY_NAMES for part in lower_parts[:-1]
        )
    )


def _is_documentation_path(relative_path: PurePosixPath) -> bool:
    lower_parts = tuple(part.lower() for part in relative_path.parts)
    basename = lower_parts[-1]
    stem = PurePosixPath(basename).stem
    suffix = relative_path.suffix.lower()
    documentation_suffix = (
        suffix in _DOCUMENTATION_EXTENSIONS or suffix == ".txt"
    )
    return bool(
        suffix in _DOCUMENTATION_EXTENSIONS
        or (
            documentation_suffix
            and (
                "docs" in lower_parts[:-1]
                or "doc" in lower_parts[:-1]
            )
        )
        or (
            documentation_suffix
            and (
                stem.startswith("readme")
                or stem.startswith("changelog")
                or stem.startswith("contributing")
            )
        )
    )


def _is_build_metadata_path(relative_path: PurePosixPath) -> bool:
    return relative_path.name.lower() in _BUILD_METADATA_FILENAMES


def _classify_evidence(relative_path: PurePosixPath) -> str | None:
    basename = relative_path.name.lower()
    suffix = relative_path.suffix.lower()
    if basename in _EXCLUDED_EVIDENCE_FILENAMES:
        return None
    if _is_fixture_path(relative_path):
        if (
            suffix in _SOURCE_EXTENSIONS
            or suffix in _CONFIG_EXTENSIONS
            or suffix in _DOCUMENTATION_EXTENSIONS
        ):
            return "fixture"
        return None
    if _is_test_path(relative_path):
        if (
            suffix in _SOURCE_EXTENSIONS
            or suffix in _CONFIG_EXTENSIONS
            or suffix in _DOCUMENTATION_EXTENSIONS
        ):
            return "test"
        return None
    if _is_build_metadata_path(relative_path):
        return "build_metadata"
    if _is_documentation_path(relative_path):
        return "documentation"
    if _is_config_path(relative_path):
        return "configuration"
    if suffix in _SOURCE_EXTENSIONS:
        return "implementation"
    return None


def classify_evidence_path(
    relative_path: str | PurePosixPath,
) -> str | None:
    """Return the deterministic repository-index category for one path."""

    path = (
        relative_path
        if isinstance(relative_path, PurePosixPath)
        else PurePosixPath(str(relative_path).replace("\\", "/"))
    )
    return _classify_evidence(path)


def _language_for_path(relative_path: PurePosixPath) -> str:
    suffix = relative_path.suffix.lower()
    if suffix in _LANGUAGE_BY_EXTENSION:
        return _LANGUAGE_BY_EXTENSION[suffix]
    basename = relative_path.name.lower()
    if basename in {"dockerfile", "makefile"}:
        return basename
    return suffix.lstrip(".") or "text"


def _looks_binary(data: bytes) -> bool:
    if b"\x00" in data:
        return True
    if not data:
        return False
    sample = data[:8192]
    suspicious = sum(
        byte < 32 and byte not in {8, 9, 10, 12, 13}
        for byte in sample
    )
    return suspicious / len(sample) > 0.02


def _read_evidence_text(path: Path, max_file_bytes: int) -> tuple[str, bytes]:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise EvidenceExtractionError(
            f"cannot stat eligible evidence file: {path}"
        ) from exc
    if size > max_file_bytes:
        raise EvidenceExtractionError(
            "eligible evidence file exceeds the explicit indexing limit "
            f"({size} > {max_file_bytes} bytes): {path}"
        )
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise EvidenceExtractionError(
            f"cannot read eligible evidence file: {path}"
        ) from exc
    if len(data) != size or _looks_binary(data):
        raise EvidenceExtractionError(
            f"eligible evidence file is binary or changed while reading: {path}"
        )
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EvidenceExtractionError(
            f"eligible evidence file is not UTF-8: {path}"
        ) from exc
    if any(len(line) > 50_000 for line in text.splitlines()):
        raise EvidenceExtractionError(
            "eligible evidence file contains an unindexable line longer than "
            f"50000 characters: {path}"
        )
    return text, data


def _iter_evidence_files(
    root: Path,
    *,
    max_files: int,
) -> Iterator[tuple[Path, PurePosixPath, str]]:
    candidates: list[tuple[str, Path, PurePosixPath, str]] = []
    for current, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        retained_dirs = []
        for dirname in sorted(dirnames):
            child = current_path / dirname
            if child.is_symlink():
                continue
            if dirname in _IGNORED_DIRECTORY_NAMES:
                continue
            retained_dirs.append(dirname)
        dirnames[:] = retained_dirs

        for filename in sorted(filenames):
            path = current_path / filename
            if path.is_symlink() or not path.is_file():
                continue
            relative = PurePosixPath(path.relative_to(root).as_posix())
            evidence_type = _classify_evidence(relative)
            if evidence_type is None:
                continue
            candidates.append((relative.as_posix(), path, relative, evidence_type))
            if len(candidates) > max_files:
                raise EvidenceExtractionError(
                    f"repository evidence file limit exceeded ({max_files})"
                )

    candidates.sort(key=lambda item: item[0].encode("utf-8"))
    for _, path, relative, evidence_type in candidates:
        yield path, relative, evidence_type


def _chunk_ranges(
    start_line: int,
    end_line: int,
    *,
    chunk_lines: int,
    overlap_lines: int,
) -> Iterator[tuple[int, int]]:
    if chunk_lines < 1:
        raise ValueError("chunk_lines must be positive")
    if overlap_lines < 0 or overlap_lines >= chunk_lines:
        raise ValueError("overlap_lines must be in [0, chunk_lines)")
    cursor = start_line
    while cursor <= end_line:
        chunk_end = min(end_line, cursor + chunk_lines - 1)
        yield cursor, chunk_end
        if chunk_end == end_line:
            break
        cursor = chunk_end - overlap_lines + 1


class _PythonSymbolCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.stack: list[tuple[str, str]] = []
        self.symbols: list[tuple[int, int, str, str]] = []

    @staticmethod
    def _node_start(node: ast.AST) -> int:
        starts = [int(getattr(node, "lineno"))]
        for decorator in getattr(node, "decorator_list", []):
            decorator_line = getattr(decorator, "lineno", None)
            if decorator_line is not None:
                starts.append(int(decorator_line))
        return min(starts)

    def _record(self, node: ast.AST, kind: str, name: str) -> None:
        start = self._node_start(node)
        end = int(getattr(node, "end_lineno", start) or start)
        qualified = ".".join([entry[0] for entry in self.stack] + [name])
        self.symbols.append((start, end, kind, qualified))

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._record(node, "class", node.name)
        self.stack.append((node.name, "class"))
        self.generic_visit(node)
        self.stack.pop()

    def _visit_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> None:
        if self.stack and self.stack[-1][1] == "class":
            kind = "method"
        elif self.stack:
            kind = "nested_function"
        else:
            kind = "function"
        self._record(node, kind, node.name)
        self.stack.append((node.name, "function"))
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)


def _nonblank(text: str) -> bool:
    return bool(text.strip())


def _uncovered_module_ranges(
    total_lines: int,
    tree: ast.Module,
) -> Iterator[tuple[int, int]]:
    covered: list[tuple[int, int]] = []
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            start = _PythonSymbolCollector._node_start(node)
            end = int(getattr(node, "end_lineno", start) or start)
            covered.append((start, end))
    covered.sort()

    cursor = 1
    for start, end in covered:
        if cursor < start:
            yield cursor, start - 1
        cursor = max(cursor, end + 1)
    if cursor <= total_lines:
        yield cursor, total_lines


def _python_snippets(
    *,
    text: str,
    file_path: str,
    file_hash: str,
    evidence_type: str,
    language: str,
    chunk_lines: int,
    overlap_lines: int,
) -> list[EvidenceSnippet]:
    lines = text.splitlines(keepends=True)
    if not lines:
        return []
    parse_text = text[1:] if text.startswith("\ufeff") else text
    try:
        tree = ast.parse(parse_text, filename=file_path)
    except (SyntaxError, ValueError, TypeError):
        return _text_chunks(
            text=text,
            file_path=file_path,
            file_hash=file_hash,
            evidence_type=evidence_type,
            language=language,
            symbol_kind="text_chunk",
            chunk_lines=chunk_lines,
            overlap_lines=overlap_lines,
        )

    collector = _PythonSymbolCollector()
    collector.visit(tree)
    snippets: list[EvidenceSnippet] = []
    for start, end, kind, qualified_name in sorted(
        collector.symbols,
        key=lambda item: (item[0], item[1], item[2], item[3]),
    ):
        for chunk_start, chunk_end in _chunk_ranges(
            start,
            end,
            chunk_lines=chunk_lines,
            overlap_lines=overlap_lines,
        ):
            snippets.append(
                EvidenceSnippet(
                    file_path=file_path,
                    file_sha256=file_hash,
                    evidence_type=evidence_type,
                    language=language,
                    symbol_kind=kind,
                    qualified_name=qualified_name,
                    start_line=chunk_start,
                    end_line=chunk_end,
                    excerpt="".join(lines[chunk_start - 1 : chunk_end]),
                )
            )

    for uncovered_start, uncovered_end in _uncovered_module_ranges(len(lines), tree):
        for start, end in _chunk_ranges(
            uncovered_start,
            uncovered_end,
            chunk_lines=chunk_lines,
            overlap_lines=overlap_lines,
        ):
            excerpt = "".join(lines[start - 1 : end])
            if not _nonblank(excerpt):
                continue
            snippets.append(
                EvidenceSnippet(
                    file_path=file_path,
                    file_sha256=file_hash,
                    evidence_type=evidence_type,
                    language=language,
                    symbol_kind="module_chunk",
                    qualified_name="<module>",
                    start_line=start,
                    end_line=end,
                    excerpt=excerpt,
                )
            )
    return snippets


def _text_chunks(
    *,
    text: str,
    file_path: str,
    file_hash: str,
    evidence_type: str,
    language: str,
    symbol_kind: str,
    chunk_lines: int,
    overlap_lines: int,
) -> list[EvidenceSnippet]:
    lines = text.splitlines(keepends=True)
    if not lines:
        return []
    snippets: list[EvidenceSnippet] = []
    for start, end in _chunk_ranges(
        1,
        len(lines),
        chunk_lines=chunk_lines,
        overlap_lines=overlap_lines,
    ):
        excerpt = "".join(lines[start - 1 : end])
        if not _nonblank(excerpt):
            continue
        snippets.append(
            EvidenceSnippet(
                file_path=file_path,
                file_sha256=file_hash,
                evidence_type=evidence_type,
                language=language,
                symbol_kind=symbol_kind,
                qualified_name=None,
                start_line=start,
                end_line=end,
                excerpt=excerpt,
            )
        )
    return snippets


def extract_repository_snippets(
    repo_root: str | os.PathLike[str],
    *,
    max_file_bytes: int = DEFAULT_MAX_EVIDENCE_FILE_BYTES,
    max_files: int = DEFAULT_MAX_EVIDENCE_FILES,
    chunk_lines: int = DEFAULT_CHUNK_LINES,
    overlap_lines: int = DEFAULT_CHUNK_OVERLAP,
) -> list[EvidenceSnippet]:
    """Index all eligible UTF-8 evidence files in ``repo_root``.

    Python files are split by AST class/function/method ranges, with additional
    module-level chunks for imports, constants, and executable statements.
    Other implementation, test, fixture, configuration, build-metadata,
    documentation, and data files are split into overlapping line blocks.
    VCS, dependency, cache, and generated build directories are excluded.
    Once a path is classified as eligible evidence, oversized, binary,
    non-UTF-8, unreadable, or unchunkable content is a hard extraction error
    rather than being silently skipped.
    """

    root = Path(repo_root).expanduser().resolve()
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"repository root is not a regular directory: {root}")

    snippets: list[EvidenceSnippet] = []
    for path, relative, evidence_type in _iter_evidence_files(root, max_files=max_files):
        text, data = _read_evidence_text(path, max_file_bytes)
        file_hash = hashlib.sha256(data).hexdigest()
        language = _language_for_path(relative)
        file_path = relative.as_posix()

        if relative.suffix.lower() in {".py", ".pyi"}:
            snippets.extend(
                _python_snippets(
                    text=text,
                    file_path=file_path,
                    file_hash=file_hash,
                    evidence_type=evidence_type,
                    language=language,
                    chunk_lines=chunk_lines,
                    overlap_lines=overlap_lines,
                )
            )
        else:
            kind = (
                "test_chunk"
                if evidence_type == "test"
                else "config_chunk"
                if evidence_type == "configuration"
                else "text_chunk"
            )
            snippets.extend(
                _text_chunks(
                    text=text,
                    file_path=file_path,
                    file_hash=file_hash,
                    evidence_type=evidence_type,
                    language=language,
                    symbol_kind=kind,
                    chunk_lines=chunk_lines,
                    overlap_lines=overlap_lines,
                )
            )

    snippets.sort(
        key=lambda snippet: (
            snippet.file_path.encode("utf-8"),
            snippet.start_line,
            snippet.end_line,
            snippet.symbol_kind,
            snippet.qualified_name or "",
        )
    )
    return snippets


def _display_excerpt(
    snippet: EvidenceSnippet,
    *,
    max_lines: int,
) -> tuple[str, int, int, bool]:
    lines = snippet.excerpt.splitlines(keepends=True)
    if len(lines) <= max_lines:
        return snippet.excerpt, snippet.start_line, snippet.end_line, False

    local_start = 0
    local_end = local_start + max_lines
    display_start = snippet.start_line + local_start
    display_end = display_start + max_lines - 1
    return (
        "".join(lines[local_start:local_end]),
        display_start,
        display_end,
        True,
    )


def format_snippets_for_llm(
    snippets: Sequence[EvidenceSnippet],
    *,
    include_line_numbers: bool = True,
    max_lines_per_snippet: int = 160,
    max_chars: int = 48_000,
) -> str:
    """Format evidence with provenance and optional display-only line numbers."""

    if max_lines_per_snippet < 1:
        raise ValueError("max_lines_per_snippet must be positive")
    blocks: list[str] = []
    used = 0
    for index, snippet in enumerate(snippets, start=1):
        excerpt, shown_start, shown_end, clipped = _display_excerpt(
            snippet,
            max_lines=max_lines_per_snippet,
        )
        rendered_excerpt = (
            add_line_numbers(excerpt, shown_start)
            if include_line_numbers
            else excerpt.rstrip("\r\n")
        )
        header_lines = [
            f"[evidence {index}]",
            f"file_path: {snippet.file_path}",
            f"file_sha256: {snippet.file_sha256}",
            f"evidence_type: {snippet.evidence_type}",
            f"language: {snippet.language}",
            f"symbol_kind: {snippet.symbol_kind}",
            f"qualified_name: {snippet.qualified_name or '<none>'}",
            f"symbol_lines: {snippet.start_line}-{snippet.end_line}",
            f"displayed_lines: {shown_start}-{shown_end}",
        ]
        if clipped:
            header_lines.append("excerpt_clipped: true")
        block = "\n".join(header_lines) + "\n<excerpt>\n"
        block += rendered_excerpt + "\n</excerpt>"
        separator_cost = 2 if blocks else 0
        if blocks and used + separator_cost + len(block) > max_chars:
            break
        blocks.append(block)
        used += separator_cost + len(block)
    return "\n\n".join(blocks)


__all__ = [
    "DEFAULT_CHUNK_LINES",
    "DEFAULT_CHUNK_OVERLAP",
    "DEFAULT_MAX_ARCHIVE_MEMBERS",
    "DEFAULT_MAX_DOWNLOAD_BYTES",
    "DEFAULT_MAX_EVIDENCE_FILE_BYTES",
    "DEFAULT_MAX_EVIDENCE_FILES",
    "DEFAULT_MAX_UNPACKED_BYTES",
    "EvidenceExtractionError",
    "EvidenceSnippet",
    "GitHubRepository",
    "RepositorySnapshotError",
    "SnapshotInfo",
    "SnapshotIntegrityError",
    "TREE_HASH_ALGORITHM",
    "UnsafeArchiveError",
    "add_line_numbers",
    "download_github_snapshot",
    "excerpt_from_file",
    "extract_line_excerpt",
    "extract_repository_snippets",
    "format_snippets_for_llm",
    "parse_github_repo_url",
    "sha256_file",
    "tree_sha256",
    "validate_commit_sha",
]

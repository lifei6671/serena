"""Optional, project-scoped read-only Git tools."""

import os
import re
import subprocess
from pathlib import Path, PureWindowsPath

from serena.tools import Tool, ToolMarkerOptional


class _GitToolBase(Tool, ToolMarkerOptional):
    _COMMIT_FORMAT = "%H%x00%h%x00%P%x00%an%x00%aI%x00%s"
    _DIFF_OPTIONS = ["--no-ext-diff", "--no-textconv", "--no-renames", "--no-color"]

    def _run_git(self, root: Path, args: list[str]) -> str:
        # prevent inherited repository selectors from overriding the active project
        env = os.environ.copy()
        for key in (
            "GIT_DIR",
            "GIT_WORK_TREE",
            "GIT_COMMON_DIR",
            "GIT_INDEX_FILE",
            "GIT_OBJECT_DIRECTORY",
            "GIT_ALTERNATE_OBJECT_DIRECTORIES",
            "GIT_NAMESPACE",
            "GIT_PREFIX",
            "GIT_LITERAL_PATHSPECS",
            "GIT_GLOB_PATHSPECS",
            "GIT_NOGLOB_PATHSPECS",
            "GIT_ICASE_PATHSPECS",
        ):
            env.pop(key, None)
        env.update(GIT_TERMINAL_PROMPT="0", GIT_PAGER="cat", PAGER="cat", GIT_OPTIONAL_LOCKS="0", GIT_NO_LAZY_FETCH="1", LC_ALL="C")
        try:
            result = subprocess.run(
                ["git", "-C", str(root), *args],
                check=False,
                capture_output=True,
                stdin=subprocess.DEVNULL,
                env=env,
                timeout=10,
                shell=False,
            )
        except FileNotFoundError as exc:
            raise ValueError("Git executable was not found.") from exc
        except subprocess.TimeoutExpired as exc:
            raise ValueError(f"Git {args[0]} timed out after 10 seconds.") from exc
        if result.returncode:
            stderr = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", result.stderr.decode("utf-8", errors="replace")).strip()
            if "not a git repository" in stderr or "must be run in a work tree" in stderr:
                raise ValueError("The active Serena project is not inside a Git work tree.")
            raise ValueError(f"Git {args[0]} failed (exit code {result.returncode}): {stderr}")
        return result.stdout.decode("utf-8", errors="replace")

    def _resolve_repository(self) -> tuple[Path, Path]:
        project = Path(self.get_project_root()).resolve()
        root = Path(self._run_git(project, ["rev-parse", "--show-toplevel"]).removesuffix("\n")).resolve()
        return root, project

    def _resolve_project_pathspec(self, root: Path, project: Path, path: str | None = None) -> str:
        # validate both lexical traversal and filesystem links, retaining the Git path spelling
        relative = path if path is not None else "."
        if "\0" in relative or Path(relative).is_absolute() or PureWindowsPath(relative).drive or relative.startswith("\\"):
            raise ValueError("Path must remain inside the active Serena project.")
        target = Path(os.path.abspath(project / relative))
        if not target.is_relative_to(project) or not target.resolve().is_relative_to(project):
            raise ValueError("Path must remain inside the active Serena project.")
        repository_path = target.relative_to(root).as_posix()
        return ":(top,literal)" + ("" if repository_path == "." else repository_path)

    def _validate_revision(self, root: Path, revision: str) -> str:
        if not revision or revision.startswith("-") or "\0" in revision:
            raise ValueError(f"Unknown or invalid Git revision: {revision}")
        try:
            return self._run_git(root, ["rev-parse", "--verify", "--end-of-options", revision + "^{commit}"]).strip()
        except ValueError as exc:
            if str(exc).startswith("Git rev-parse failed"):
                raise ValueError(f"Unknown or invalid Git revision: {revision}") from exc
            raise

    def _branch_state(self, root: Path) -> dict:
        output = self._run_git(root, ["status", "--porcelain=v2", "--branch", "--untracked-files=no", "--", ":(top,literal)"])
        headers = dict(line[2:].split(" ", 1) for line in output.split("\0")[0].splitlines() if line.startswith("# "))
        name = headers.get("branch.head")
        ahead, behind = headers.get("branch.ab", "+0 -0").split()
        return {
            "name": None if name == "(detached)" else name,
            "head": None if headers.get("branch.oid") == "(initial)" else headers.get("branch.oid"),
            "detached": name == "(detached)",
            "upstream": headers.get("branch.upstream"),
            "ahead": int(ahead),
            "behind": abs(int(behind)),
        }

    def _shorten_patch(self, root: Path, args: list[str], pathspec: str) -> str:
        summary_args = [arg for arg in args if not arg.startswith("--unified=")]
        stat = self._run_git(root, [*summary_args, "--stat", "--", pathspec])
        names = self._run_git(root, [*summary_args, "--name-status", "--", pathspec])
        return stat + "\n" + names + "\nPatch omitted. Use path to narrow the query."


class GitStatusTool(_GitToolBase):
    """Reads branch state and changed files inside the active project."""

    def apply(self, max_answer_chars: int = -1) -> str:
        """Return JSON status; file paths are relative to the active project.

        :param max_answer_chars: maximum output length, or -1 for the configured default
        """
        root, project = self._resolve_repository()
        pathspec = self._resolve_project_pathspec(root, project)
        output = self._run_git(root, ["status", "--porcelain=v1", "-z", "--untracked-files=all", "--", pathspec])
        records = iter(output.split("\0"))
        changes = []
        for record in records:
            if not record:
                continue
            change = {"path": (root / record[3:]).relative_to(project).as_posix(), "index_status": record[0], "worktree_status": record[1]}
            if "R" in record[:2] or "C" in record[:2]:
                original = root / next(records)
                if original.is_relative_to(project):
                    change["original_path"] = original.relative_to(project).as_posix()
            changes.append(change)
        return self._limit_length(
            self._to_json(
                {"repository_root": str(root), "project_root": str(project), "branch": self._branch_state(root), "changes": changes}
            ),
            max_answer_chars,
        )


class GitDiffTool(_GitToolBase):
    """Reads the active project's staged, unstaged, or combined tracked-file diff."""

    def apply(self, scope: str = "unstaged", path: str | None = None, context_lines: int = 3, max_answer_chars: int = -1) -> str:
        """Return a complete unified diff, or a summary when it exceeds the output budget.

        :param scope: unstaged, staged, or all (tracked changes against HEAD)
        :param path: literal file or directory path relative to the active project
        :param context_lines: non-negative number of surrounding lines
        :param max_answer_chars: maximum output length, or -1 for the configured default
        """
        if scope not in ("unstaged", "staged", "all"):
            raise ValueError("scope must be unstaged, staged, or all.")
        if context_lines < 0:
            raise ValueError("context_lines must be non-negative.")
        root, project = self._resolve_repository()
        pathspec = self._resolve_project_pathspec(root, project, path)
        args = ["diff", *self._DIFF_OPTIONS, f"--unified={context_lines}"]
        if scope == "staged":
            args.append("--cached")
        elif scope == "all":
            args.append(self._validate_revision(root, "HEAD"))
        result = self._run_git(root, [*args, "--", pathspec])
        return self._limit_length(result, max_answer_chars, [lambda: self._shorten_patch(root, args, pathspec)])


class GitLogTool(_GitToolBase):
    """Reads commit summaries affecting the active project."""

    def apply(self, limit: int = 20, revision: str = "HEAD", path: str | None = None, max_answer_chars: int = -1) -> str:
        """Return JSON history without full commit message bodies.

        :param limit: number of commits, between 1 and 100
        :param revision: a commit SHA or revision such as HEAD, HEAD~1, or a branch
        :param path: literal file or directory path relative to the active project
        :param max_answer_chars: maximum output length, or -1 for the configured default
        """
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100.")
        root, project = self._resolve_repository()
        pathspec = self._resolve_project_pathspec(root, project, path)
        oid = self._validate_revision(root, revision)
        output = self._run_git(
            root,
            [
                "log",
                "--no-show-signature",
                "--no-notes",
                "-z",
                f"--max-count={limit}",
                f"--format={self._COMMIT_FORMAT}",
                oid,
                "--",
                pathspec,
            ],
        )
        fields = output.removesuffix("\0").split("\0") if output else []
        commits = []
        for i in range(0, len(fields), 6):
            sha, short, parents, author, authored_at, subject = fields[i : i + 6]
            commits.append(
                {
                    "hash": sha,
                    "short_hash": short,
                    "parents": parents.split(),
                    "author": author,
                    "authored_at": authored_at,
                    "subject": subject,
                }
            )
        return self._limit_length(self._to_json({"revision": revision, "commits": commits}), max_answer_chars)


class GitShowTool(_GitToolBase):
    """Reads commit metadata and changes restricted to the active project."""

    def apply(self, revision: str, path: str | None = None, include_patch: bool = True, max_answer_chars: int = -1) -> str:
        """Return commit metadata and stat, optionally with a complete unified patch.

        :param revision: a commit SHA or revision such as HEAD, HEAD~1, or a branch
        :param path: literal file or directory path relative to the active project
        :param include_patch: whether to include the unified patch
        :param max_answer_chars: maximum output length, or -1 for the configured default
        """
        root, project = self._resolve_repository()
        pathspec = self._resolve_project_pathspec(root, project, path)
        oid = self._validate_revision(root, revision)
        args = ["show", *self._DIFF_OPTIONS, "--no-show-signature", "--no-notes", "--format=medium", oid]
        result = self._run_git(root, [*args, "--stat", *(["--patch"] if include_patch else []), "--", pathspec])
        return self._limit_length(result, max_answer_chars, [lambda: self._shorten_patch(root, args, pathspec)])


class GitBranchTool(_GitToolBase):
    """Reads local and optionally remote-tracking branches without accessing the network."""

    def apply(self, include_remote: bool = False, max_answer_chars: int = -1) -> str:
        """Return JSON branch information for the active project's repository.

        :param include_remote: include locally stored remote-tracking refs
        :param max_answer_chars: maximum output length, or -1 for the configured default
        """
        root, _ = self._resolve_repository()
        state = self._branch_state(root)
        output = self._run_git(
            root,
            [
                "for-each-ref",
                "--format=%(refname:short)%00%(objectname)%00%(upstream:short)%00%(HEAD)",
                "refs/heads/",
                *(["refs/remotes/"] if include_remote else []),
            ],
        )
        branches = []
        for line in output.splitlines():
            name, oid, upstream, current = line.split("\0")
            branches.append({"name": name, "oid": oid, "upstream": upstream or None, "current": current == "*"})
        return self._limit_length(
            self._to_json({"current": state["name"], "detached": state["detached"], "branches": branches}), max_answer_chars
        )


class GitWorktreeListTool(_GitToolBase):
    """Reads the repository's registered worktrees."""

    def apply(self, max_answer_chars: int = -1) -> str:
        """Return JSON worktree metadata without changing worktree registration.

        :param max_answer_chars: maximum output length, or -1 for the configured default
        """
        root, _ = self._resolve_repository()
        output = self._run_git(root, ["worktree", "list", "--porcelain", "-z"])
        worktrees = []
        for record in output.split("\0\0"):
            if not record:
                continue
            fields = dict((field.split(" ", 1) + [""])[:2] for field in record.split("\0") if field)
            worktrees.append(
                {
                    "path": fields["worktree"],
                    "head": fields.get("HEAD"),
                    "branch": fields.get("branch"),
                    "bare": "bare" in fields,
                    "detached": "detached" in fields,
                }
            )
        return self._limit_length(self._to_json({"worktrees": worktrees}), max_answer_chars)

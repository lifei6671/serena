import json
import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from serena.agent import SerenaAgent
from serena.config.serena_config import SerenaConfig
from serena.mcp import SerenaMCPFactory
from serena.tools.git_tools import GitBranchTool, GitDiffTool, GitLogTool, GitShowTool, GitStatusTool, GitWorktreeListTool
from serena.tools.tools_base import ToolRegistry

GIT_TOOLS = (GitStatusTool, GitDiffTool, GitLogTool, GitShowTool, GitBranchTool, GitWorktreeListTool)


def git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True, encoding="utf-8").stdout.strip()


def write(root: Path, path: str, content: str) -> None:
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8", newline="\n")


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    # isolate fixture commits from the developer's Git settings and identity
    for key in list(os.environ):
        if key.startswith("GIT_"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_AUTHOR_NAME", "Test Author")
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", "test@example.invalid")
    monkeypatch.setenv("GIT_COMMITTER_NAME", "Test Author")
    monkeypatch.setenv("GIT_COMMITTER_EMAIL", "test@example.invalid")
    git(tmp_path, "init", "-b", "main")
    write(tmp_path, "backend/file space.txt", "original\n")
    write(tmp_path, "frontend/outside.txt", "outside original\n")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-m", "initial")
    return tmp_path


def agent(root: Path) -> MagicMock:
    result = MagicMock(spec=SerenaAgent)
    result.get_active_project_or_raise.return_value.project_root = str(root)
    result.serena_config = SerenaConfig()
    return result


def test_status_states(repo: Path):
    tool = GitStatusTool(agent(repo))
    assert json.loads(tool.apply())["changes"] == []
    write(repo, "new space.txt", "new\n")
    write(repo, "backend/file space.txt", "staged\n")
    git(repo, "add", "backend")
    write(repo, "backend/file space.txt", "worktree\n")
    (repo / "frontend/outside.txt").unlink()
    changes = {c["path"]: (c["index_status"], c["worktree_status"]) for c in json.loads(tool.apply())["changes"]}
    assert changes == {"new space.txt": ("?", "?"), "backend/file space.txt": ("M", "M"), "frontend/outside.txt": (" ", "D")}
    git(repo, "add", ".")
    changes = {c["path"]: (c["index_status"], c["worktree_status"]) for c in json.loads(tool.apply())["changes"]}
    assert changes["backend/file space.txt"] == ("M", " ")
    assert changes["frontend/outside.txt"] == ("D", " ")


def test_rename_and_detached(repo: Path):
    git(repo, "mv", "backend/file space.txt", "backend/renamed space.txt")
    status = json.loads(GitStatusTool(agent(repo / "backend")).apply())
    assert status["changes"] == [
        {"path": "renamed space.txt", "index_status": "R", "worktree_status": " ", "original_path": "file space.txt"}
    ]
    git(repo, "commit", "-m", "rename")
    git(repo, "checkout", "--detach")
    state = json.loads(GitStatusTool(agent(repo)).apply())["branch"]
    assert state["name"] is None and state["detached"]
    assert state["head"] == git(repo, "rev-parse", "HEAD")


def test_diff_scopes_and_subtree(repo: Path):
    write(repo, "backend/file space.txt", "staged\n")
    git(repo, "add", "backend")
    write(repo, "backend/file space.txt", "unstaged\n")
    write(repo, "frontend/outside.txt", "outside secret\n")
    write(repo, "backend/untracked.txt", "untracked secret\n")
    tool = GitDiffTool(agent(repo / "backend"))
    assert "-staged\n+unstaged" in tool.apply()
    assert "-original\n+staged" in tool.apply(scope="staged")
    combined = tool.apply(scope="all", path="file space.txt", context_lines=0)
    assert "-original\n+unstaged" in combined
    assert "secret" not in tool.apply(scope="all")
    assert "outside secret" in GitDiffTool(agent(repo)).apply()
    paths = [c["path"] for c in json.loads(GitStatusTool(agent(repo / "backend")).apply())["changes"]]
    assert paths == ["file space.txt", "untracked.txt"]


@pytest.mark.parametrize(
    "path", ["../frontend", "../../escape", "/absolute", "C:\\absolute", "C:relative", "\\\\server\\share", "bad\0path"]
)
def test_path_boundary(repo: Path, path: str):
    for tool, kwargs in ((GitDiffTool, {}), (GitLogTool, {}), (GitShowTool, {"revision": "HEAD"})):
        with pytest.raises(ValueError, match="Path must remain inside"):
            tool(agent(repo / "backend")).apply(path=path, **kwargs)


def test_literal_pathspec(repo: Path):
    write(repo, "backend/[abc].txt", "literal\n")
    write(repo, "backend/a.txt", "other\n")
    git(repo, "add", ".")
    diff = GitDiffTool(agent(repo / "backend")).apply(scope="staged", path="[abc].txt")
    assert "+literal" in diff and "+other" not in diff
    assert GitDiffTool(agent(repo / "backend")).apply(scope="staged", path=":(top)**") == ""


def test_large_patch_summaries(repo: Path):
    write(repo, "backend/file space.txt", "a long added line\n" * 1000)
    tool = GitDiffTool(agent(repo))
    summary = tool.apply(max_answer_chars=1000)
    assert "Patch omitted" in summary and "file space.txt" in summary and "@@" not in summary
    assert len(summary) <= 1000
    tool.agent.serena_config.default_max_tool_answer_chars = 1000
    assert tool.apply() == summary
    git(repo, "add", ".")
    git(repo, "commit", "-m", "large")
    summary = GitShowTool(agent(repo)).apply("HEAD", max_answer_chars=1500)
    assert "Patch omitted" in summary and "@@" not in summary
    assert len(summary) <= 1500


def test_log_and_show_revisions_and_filters(repo: Path):
    write(repo, "frontend/outside.txt", "outside secret\n")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "frontend only")
    write(repo, "backend/file space.txt", "inside change\n")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "backend change", "-m", "message body")
    git(repo, "branch", "feature/foo")
    a = agent(repo / "backend")
    log = json.loads(GitLogTool(a).apply())
    assert [c["subject"] for c in log["commits"]] == ["backend change", "initial"]
    assert log["commits"][0]["author"] == "Test Author"
    for revision in ("HEAD", "main", "feature/foo", git(repo, "rev-parse", "HEAD")):
        result = json.loads(GitLogTool(a).apply(limit=1, revision=revision))
        assert len(result["commits"]) == 1
        show = GitShowTool(a).apply(revision)
        assert "+inside change" in show and "outside secret" not in show
        assert "@@" not in GitShowTool(a).apply(revision, include_patch=False)
    assert json.loads(GitLogTool(a).apply(revision="HEAD~1"))["commits"][0]["subject"] == "initial"
    assert "outside secret" not in GitShowTool(a).apply("HEAD~1")
    assert json.loads(GitLogTool(a).apply(path="missing"))["commits"] == []


@pytest.mark.parametrize("revision", ["--output=oops", "does-not-exist", "HEAD:backend", "", "bad\0rev"])
def test_invalid_revision(repo: Path, revision: str):
    for tool in (GitLogTool, GitShowTool):
        with pytest.raises(ValueError, match="Unknown or invalid Git revision"):
            tool(agent(repo)).apply(revision=revision)


@pytest.mark.parametrize("limit", [0, 101])
def test_invalid_limit(repo: Path, limit: int):
    with pytest.raises(ValueError, match="limit must"):
        GitLogTool(agent(repo)).apply(limit=limit)


def test_branches_upstream_and_worktrees(repo: Path, tmp_path: Path):
    git(repo, "branch", "feature/foo")
    git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    git(repo, "config", "branch.main.remote", "origin")
    git(repo, "config", "branch.main.merge", "refs/heads/main")
    git(repo, "config", "remote.origin.fetch", "+refs/heads/*:refs/remotes/origin/*")
    git(repo, "commit", "--allow-empty", "-m", "ahead")
    tool = GitBranchTool(agent(repo))
    state = json.loads(GitStatusTool(agent(repo)).apply())["branch"]
    assert (state["upstream"], state["ahead"], state["behind"]) == ("origin/main", 1, 0)
    assert {b["name"] for b in json.loads(tool.apply())["branches"]} == {"main", "feature/foo"}
    assert {b["name"] for b in json.loads(tool.apply(include_remote=True))["branches"]} == {"main", "feature/foo", "origin/main"}
    worktree = tmp_path / "linked space"
    git(repo, "worktree", "add", str(worktree), "feature/foo")
    trees = json.loads(GitWorktreeListTool(agent(worktree)).apply())["worktrees"]
    assert len(trees) == 2
    assert any(Path(w["path"]) == worktree and w["branch"] == "refs/heads/feature/foo" for w in trees)
    git(repo, "checkout", "--detach")
    assert json.loads(tool.apply())["detached"]
    assert json.loads(tool.apply())["current"] is None


def test_empty_repository(tmp_path: Path):
    git(tmp_path, "init", "-b", "main")
    a = agent(tmp_path)
    state = json.loads(GitStatusTool(a).apply())["branch"]
    assert state["head"] is None and state["name"] == "main"
    assert json.loads(GitBranchTool(a).apply())["branches"] == []
    assert GitDiffTool(a).apply(scope="staged") == ""
    with pytest.raises(ValueError, match="Unknown or invalid"):
        GitDiffTool(a).apply(scope="all")


def test_non_repository_and_no_project(tmp_path: Path):
    with pytest.raises(ValueError, match="not inside a Git work tree"):
        GitStatusTool(agent(tmp_path)).apply()
    a = agent(tmp_path)
    a.get_active_project_or_raise.side_effect = ValueError("No active project")
    with pytest.raises(ValueError, match="No active project"):
        GitStatusTool(a).apply()


@pytest.mark.parametrize(
    "error, message", [(FileNotFoundError(), "Git executable was not found"), (subprocess.TimeoutExpired("git", 10), "timed out after 10")]
)
def test_process_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: Exception, message: str):
    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(ValueError, match=message):
        GitStatusTool(agent(tmp_path)).apply()


def test_read_only_and_external_diff_disabled(repo: Path):
    write(repo, "backend/file space.txt", "modified\n")
    git(repo, "config", "diff.external", "nonexistent-external-diff")
    git(repo, "config", "diff.test.textconv", "nonexistent-textconv")
    write(repo, ".gitattributes", "*.txt diff=test\n")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "attributes")
    write(repo, "backend/file space.txt", "modified again\n")
    before = {p.relative_to(repo).as_posix(): p.read_bytes() for p in repo.rglob("*") if p.is_file()}
    a = agent(repo)
    for cls in GIT_TOOLS:
        cls(a).apply(**({"revision": "HEAD"} if cls is GitShowTool else {}))
    after = {p.relative_to(repo).as_posix(): p.read_bytes() for p in repo.rglob("*") if p.is_file()}
    assert before == after


def test_registry_and_runtime_context(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SERENA_USAGE_REPORTING", "false")
    registry = ToolRegistry()
    names = {cls.get_name_from_cls() for cls in GIT_TOOLS}
    assert names <= set(registry.get_tool_names_optional())
    assert names.isdisjoint(registry.get_tool_names_default_enabled())
    cfg = SerenaConfig().with_headless_mode_overrides()
    factory = SerenaMCPFactory(transport="stdio", context="chatgpt-review")
    factory.agent = factory._create_serena_agent(cfg)
    tools = list(factory._iter_tools())
    exposed = {tool.get_name() for tool in tools}
    assert names <= exposed
    assert "execute_shell_command" not in exposed
    assert all(tool.is_readonly() for tool in tools)
    for tool in tools:
        factory.make_mcp_tool(tool, openai_tool_compatible=True)


def test_cross_subtree_rename(repo: Path):
    git(repo, "mv", "frontend/outside.txt", "backend/moved.txt")
    a = agent(repo / "backend")
    status = json.loads(GitStatusTool(a).apply())
    assert all(c["path"].startswith("moved") for c in status["changes"])
    assert "frontend" not in json.dumps(status["changes"])
    diff = GitDiffTool(a).apply(scope="staged")
    assert "frontend" not in diff and "new file mode" in diff
    git(repo, "commit", "-m", "move into project")
    assert "frontend" not in GitShowTool(a).apply("HEAD")


def test_empty_commit_subject(repo: Path):
    write(repo, "backend/file space.txt", "changed\n")
    git(repo, "add", ".")
    git(repo, "commit", "--allow-empty-message", "-m", "")
    commits = json.loads(GitLogTool(agent(repo)).apply())["commits"]
    assert commits[0]["subject"] == ""
    assert commits[1]["subject"] == "initial"


def test_environment_cannot_redirect_repository(repo: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GIT_DIR", str(repo / "missing.git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(repo / "frontend"))
    monkeypatch.setenv("GIT_INDEX_FILE", str(repo / "missing-index"))
    monkeypatch.setenv("GIT_LITERAL_PATHSPECS", "1")
    state = json.loads(GitStatusTool(agent(repo / "backend")).apply())
    assert Path(state["repository_root"]) == repo
    assert state["changes"] == []
    assert len(json.loads(GitLogTool(agent(repo / "backend")).apply())["commits"]) == 1


def test_symlink_escape(repo: Path):
    link = repo / "backend/link"
    try:
        link.symlink_to(repo / "frontend", target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"Symlink creation unavailable: {exc}")
    with pytest.raises(ValueError, match="Path must remain inside"):
        GitDiffTool(agent(repo / "backend")).apply(path="link/outside.txt")


def test_output_limits_and_argument_errors(repo: Path):
    for cls in (GitStatusTool, GitLogTool, GitBranchTool, GitWorktreeListTool):
        assert "answer is too long" in cls(agent(repo)).apply(max_answer_chars=1)
        with pytest.raises(ValueError, match="Must be positive"):
            cls(agent(repo)).apply(max_answer_chars=0)
    with pytest.raises(ValueError, match="scope must"):
        GitDiffTool(agent(repo)).apply(scope="write")
    with pytest.raises(ValueError, match="context_lines must"):
        GitDiffTool(agent(repo)).apply(context_lines=-1)


def test_git_failure_is_sanitized(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    run = MagicMock(return_value=subprocess.CompletedProcess(["git"], 128, b"", b"fatal: dubious ownership\x1b[31m\x07\n"))
    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(ValueError, match="Git rev-parse failed \\(exit code 128\\): fatal: dubious ownership") as exc:
        GitStatusTool(agent(tmp_path)).apply()
    assert "\x1b" not in str(exc.value) and "\x07" not in str(exc.value)
    assert run.call_count == 1


def test_worktree_porcelain_records(repo: Path, monkeypatch: pytest.MonkeyPatch):
    real_run = subprocess.run

    def run(args, **kwargs):
        if "worktree" in args:
            data = b"worktree /repo space\x00HEAD abc\x00branch refs/heads/main\x00locked reason\x00\x00"
            data += b"worktree /detached\npath\x00HEAD def\x00detached\x00prunable reason\x00\x00"
            data += b"worktree /bare\x00bare\x00\x00"
            return subprocess.CompletedProcess(args, 0, data, b"")
        return real_run(args, **kwargs)

    monkeypatch.setattr(subprocess, "run", run)
    trees = json.loads(GitWorktreeListTool(agent(repo)).apply())["worktrees"]
    assert trees == [
        {"path": "/repo space", "head": "abc", "branch": "refs/heads/main", "bare": False, "detached": False},
        {"path": "/detached\npath", "head": "def", "branch": None, "bare": False, "detached": True},
        {"path": "/bare", "head": None, "branch": None, "bare": True, "detached": False},
    ]

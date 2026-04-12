"""Tests for release.py — version bumping, commit parsing, and git operations."""

import subprocess
from pathlib import Path

import pytest

from release import (
    bump,
    gather_commits,
    git_tag,
    map_commit_type,
    parse_commit_message,
    prep_json,
    read_version,
    strip_trailers,
    write_version,
)


# ---------------------------------------------------------------------------
# TestReadVersion
# ---------------------------------------------------------------------------
class TestReadVersion:
    def test_reads_version(self, tmp_path: Path) -> None:
        f = tmp_path / "VERSION"
        f.write_text("1.2.3\n")
        assert read_version(f) == (1, 2, 3)

    def test_reads_version_without_newline(self, tmp_path: Path) -> None:
        f = tmp_path / "VERSION"
        f.write_text("4.5.6")
        assert read_version(f) == (4, 5, 6)


# ---------------------------------------------------------------------------
# TestBumpVersion
# ---------------------------------------------------------------------------
class TestBumpVersion:
    def test_bump_patch(self) -> None:
        assert bump(1, 2, 3, "patch") == (1, 2, 4)

    def test_bump_minor(self) -> None:
        assert bump(1, 2, 3, "minor") == (1, 3, 0)

    def test_bump_major(self) -> None:
        assert bump(1, 2, 3, "major") == (2, 0, 0)

    def test_bump_invalid(self) -> None:
        with pytest.raises(ValueError, match="Invalid bump part"):
            bump(1, 2, 3, "invalid")


# ---------------------------------------------------------------------------
# TestWriteVersion
# ---------------------------------------------------------------------------
class TestWriteVersion:
    def test_writes_version(self, tmp_path: Path) -> None:
        f = tmp_path / "VERSION"
        write_version(f, (2, 3, 4))
        assert f.read_text() == "2.3.4\n"


# ---------------------------------------------------------------------------
# TestParseCommitMessage
# ---------------------------------------------------------------------------
class TestParseCommitMessage:
    def test_feat_with_scope(self) -> None:
        assert parse_commit_message("feat(viewer): add map layer") == (
            "feat",
            "viewer",
            "add map layer",
        )

    def test_fix_without_scope(self) -> None:
        assert parse_commit_message("fix: correct offset") == (
            "fix",
            None,
            "correct offset",
        )

    def test_no_prefix(self) -> None:
        assert parse_commit_message("random commit message") == (
            None,
            None,
            "random commit message",
        )

    def test_chore(self) -> None:
        assert parse_commit_message("chore: update deps") == (
            "chore",
            None,
            "update deps",
        )

    def test_release(self) -> None:
        assert parse_commit_message("release: v1.0.0") == (
            "release",
            None,
            "v1.0.0",
        )


# ---------------------------------------------------------------------------
# TestMapCommitType
# ---------------------------------------------------------------------------
class TestMapCommitType:
    def test_feat(self) -> None:
        assert map_commit_type("feat") == "Added"

    def test_fix(self) -> None:
        assert map_commit_type("fix") == "Fixed"

    def test_refactor(self) -> None:
        assert map_commit_type("refactor") == "Changed"

    def test_chore(self) -> None:
        assert map_commit_type("chore") is None

    def test_release(self) -> None:
        assert map_commit_type("release") is None

    def test_none(self) -> None:
        assert map_commit_type(None) == "Other"


# ---------------------------------------------------------------------------
# TestStripTrailers
# ---------------------------------------------------------------------------
class TestStripTrailers:
    def test_strips_co_authored_by(self) -> None:
        body = "Some description\n\nCo-Authored-By: User <user@example.com>"
        assert strip_trailers(body) == "Some description"

    def test_no_trailers(self) -> None:
        body = "Just a plain body"
        assert strip_trailers(body) == "Just a plain body"

    def test_empty_body(self) -> None:
        assert strip_trailers("") == ""

    def test_only_trailer(self) -> None:
        body = "Co-Authored-By: User <user@example.com>"
        assert strip_trailers(body) == ""


# ---------------------------------------------------------------------------
# Helper: create a temporary git repo
# ---------------------------------------------------------------------------
def _init_repo(tmp_path: Path) -> Path:
    """Create a temporary git repo with an initial commit and tag."""
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    # Initial commit + tag
    (tmp_path / "README.md").write_text("# Test\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "chore: initial commit"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "tag", "-a", "v1.0.0", "-m", "v1.0.0"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    return tmp_path


def _add_commits(tmp_path: Path) -> None:
    """Add two commits after the tag."""
    (tmp_path / "analysis.py").write_text("# analysis\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "feat(viewer): add analysis module"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    (tmp_path / "layers.py").write_text("# layers\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "fix: correct layer rendering"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )


# ---------------------------------------------------------------------------
# TestGatherCommits
# ---------------------------------------------------------------------------
class TestGatherCommits:
    def test_gathers_since_tag(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        _add_commits(repo)
        commits = gather_commits(cwd=repo)
        assert len(commits) == 2
        # Most recent first (git log order)
        assert commits[0]["changelog_section"] == "Fixed"
        assert commits[1]["changelog_section"] == "Added"

    def test_gathers_changed_files(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        _add_commits(repo)
        commits = gather_commits(cwd=repo)
        assert "layers.py" in commits[0]["files_changed"]
        assert "analysis.py" in commits[1]["files_changed"]

    def test_includes_release_commits(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        _add_commits(repo)
        # Add a release commit
        (tmp_path / "VERSION").write_text("1.1.0\n")
        subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "release: v1.1.0"],
            cwd=repo,
            check=True,
            capture_output=True,
        )
        commits = gather_commits(cwd=repo)
        # release type is in OMITTED_TYPES so changelog_section is None
        assert len(commits) == 3
        assert commits[0]["type"] == "release"
        assert commits[0]["changelog_section"] is None


# ---------------------------------------------------------------------------
# TestGatherCommitsNoTags
# ---------------------------------------------------------------------------
class TestGatherCommitsNoTags:
    def test_gathers_all_when_no_tags(self, tmp_path: Path) -> None:
        subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )
        (tmp_path / "file.txt").write_text("hello\n")
        subprocess.run(
            ["git", "add", "."], cwd=tmp_path, check=True, capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", "feat: first commit"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )
        (tmp_path / "file2.txt").write_text("world\n")
        subprocess.run(
            ["git", "add", "."], cwd=tmp_path, check=True, capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", "fix: second commit"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )
        commits = gather_commits(cwd=tmp_path)
        assert len(commits) == 2


# ---------------------------------------------------------------------------
# TestGatherCommitsSinceRef
# ---------------------------------------------------------------------------
class TestGatherCommitsSinceRef:
    def test_gathers_since_specific_ref(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        # Get the hash of HEAD (tagged commit)
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        )
        ref = result.stdout.strip()
        _add_commits(repo)
        commits = gather_commits(cwd=repo, since=ref)
        assert len(commits) == 2


# ---------------------------------------------------------------------------
# TestPrepJson
# ---------------------------------------------------------------------------
class TestPrepJson:
    def _make_repo_with_version(self, tmp_path: Path) -> Path:
        repo = _init_repo(tmp_path)
        (repo / "VERSION").write_text("1.0.0\n")
        subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "chore: add version file"],
            cwd=repo,
            check=True,
            capture_output=True,
        )
        # Re-tag so version file commit is before the tag
        subprocess.run(
            ["git", "tag", "-a", "v1.0.1", "-m", "v1.0.1"],
            cwd=repo,
            check=True,
            capture_output=True,
        )
        _add_commits(repo)
        return repo

    def test_returns_valid_json(self, tmp_path: Path) -> None:
        repo = self._make_repo_with_version(tmp_path)
        result = prep_json("minor", version_file=repo / "VERSION", cwd=repo)
        assert result["old_version"] == "1.0.0"
        assert result["new_version"] == "1.1.0"
        assert isinstance(result["commits"], list)
        assert len(result["commits"]) == 2

    def test_includes_changed_modules(self, tmp_path: Path) -> None:
        repo = self._make_repo_with_version(tmp_path)
        result = prep_json("patch", version_file=repo / "VERSION", cwd=repo)
        assert "analysis.py" in result["changed_modules"]
        assert "layers.py" in result["changed_modules"]

    def test_includes_user_facing_modules(self, tmp_path: Path) -> None:
        repo = self._make_repo_with_version(tmp_path)
        result = prep_json("patch", version_file=repo / "VERSION", cwd=repo)
        assert "analysis.py" in result["user_facing_modules"]
        assert "layers.py" in result["user_facing_modules"]


# ---------------------------------------------------------------------------
# TestGitTag
# ---------------------------------------------------------------------------
class TestGitTag:
    def test_creates_commit_and_tag(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        (repo / "VERSION").write_text("2.0.0\n")
        subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "chore: add version"],
            cwd=repo,
            check=True,
            capture_output=True,
        )
        # Modify VERSION for the release
        (repo / "VERSION").write_text("2.1.0\n")
        git_tag("2.1.0", files=["VERSION"], cwd=repo)

        # Verify tag exists
        result = subprocess.run(
            ["git", "tag", "-l", "v2.1.0"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        )
        assert "v2.1.0" in result.stdout

        # Verify commit message
        result = subprocess.run(
            ["git", "log", "-1", "--format=%s"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        )
        assert result.stdout.strip() == "release: v2.1.0"

    def test_uses_default_files(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        # Create some default files
        (repo / "VERSION").write_text("3.0.0\n")
        (repo / "CHANGELOG.md").write_text("# Changelog\n")
        (repo / "README.md").write_text("# README\n")
        subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "chore: add files"],
            cwd=repo,
            check=True,
            capture_output=True,
        )
        # Modify for release
        (repo / "VERSION").write_text("3.1.0\n")
        (repo / "CHANGELOG.md").write_text("# Changelog\n\n## 3.1.0\n")
        # Note: docs/API.md doesn't exist — should be skipped
        git_tag("3.1.0", cwd=repo)

        result = subprocess.run(
            ["git", "tag", "-l", "v3.1.0"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        )
        assert "v3.1.0" in result.stdout

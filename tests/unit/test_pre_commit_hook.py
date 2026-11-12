"""
The pre-commit hook (.pre-commit-config.yaml): on `git commit`, ruff fixes the lint errors it
can in the staged Python files and stops the commit so the fixes can be reviewed and added;
errors it cannot fix block the commit.

Each test makes a throwaway git repository with this project's hook configuration and ruff
settings, installs the hook with `pre-commit install` and makes real commits. The hook runs the
project's ruff through uv (`uv run --frozen ruff ...`). A throwaway repository is not a uv
project, so there it runs this environment's ruff (`python -m ruff`), the same locked version.
"""
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
HOOK_ENTRY = 'uv run --frozen ruff check --fix'
# The install command in the README and CONTRIBUTING (after `uv run pre-commit`)
INSTALL = ('install', '--allow-missing-config')

FIXABLE = "import sys\nimport os\n\nx = 1   \nprint(os.sep, sys.version, x)\n"     # unsorted imports, trailing spaces
FIXED = "import os\nimport sys\n\nx = 1\nprint(os.sep, sys.version, x)\n"
CLEAN = "import os\n\nprint(os.sep)\n"
LONG = f"MESSAGE = '{'x' * 120}'\n"                                                       # 135 characters: E501


def _hook() -> dict:
    config = yaml.safe_load((ROOT / '.pre-commit-config.yaml').read_text())
    return next(h for repo in config['repos'] for h in repo['hooks'] if h['id'] == 'ruff')


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    """Runs git; its output (and the hook's, which git sends to stderr) is in .stdout."""
    return subprocess.run(['git', *args], cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A git repository with the project's hook (installed) and ruff settings, and one commit."""
    repo = tmp_path / 'repo'
    repo.mkdir()
    monkeypatch.setenv('PRE_COMMIT_HOME', str(tmp_path / 'pre-commit-cache'))
    (repo / 'pyproject.toml').write_text((ROOT / 'pyproject.toml').read_text())
    hook = dict(_hook(), entry=f"{shlex.quote(sys.executable)} -m ruff check --fix")
    (repo / '.pre-commit-config.yaml').write_text(yaml.safe_dump({'repos': [{'repo': 'local', 'hooks': [hook]}]}))
    _git(repo, 'init', '-q')
    _git(repo, 'config', 'user.email', 'test@example.com')
    _git(repo, 'config', 'user.name', 'Test')
    _git(repo, 'config', 'commit.gpgsign', 'false')
    subprocess.run([sys.executable, '-m', 'pre_commit', *INSTALL], cwd=repo, check=True, capture_output=True)
    assert _commit(repo, 'pyproject.toml', '.pre-commit-config.yaml').returncode == 0
    return repo


def _commit(repo: Path, *files: str) -> subprocess.CompletedProcess:
    _git(repo, 'add', *files)
    return _git(repo, 'commit', '-m', 'test')


def _commits(repo: Path) -> int:
    return int(_git(repo, 'rev-list', '--count', 'HEAD').stdout)


def test_the_hook_runs_the_projects_ruff_like_ci():
    hook = _hook()
    assert hook['entry'] == HOOK_ENTRY and hook['language'] == 'system' and 'python' in hook['types_or']
    project = (ROOT / 'pyproject.toml').read_text()   # (tomllib is Python 3.11+)
    dev = project.split('[dependency-groups]')[1].split(']')[0]
    # One ruff for the hook, CI and `uv run ruff check .`: the locked dev dependency, with pyproject's rules
    assert '"ruff>=' in dev and '"pre-commit>=' in dev
    assert 'name = "ruff"' in (ROOT / 'uv.lock').read_text()
    assert 'uv run ruff check' in (ROOT / '.github/workflows/tests.yml').read_text()
    # Staged files are passed by name, so excluded files must stay excluded when named
    assert '\nforce-exclude = true\n' in project.split('[tool.ruff]')[1].split('[tool.ruff.lint]')[0]


def test_fixable_errors_are_fixed_and_the_commit_stops_until_they_are_added(repo):
    (repo / 'bad.py').write_text(FIXABLE)
    first = _commit(repo, 'bad.py')
    assert first.returncode != 0 and 'files were modified by this hook' in first.stdout
    assert (repo / 'bad.py').read_text() == FIXED                      # fixed in place, for review
    assert _commits(repo) == 1                                        # nothing committed yet
    assert _commit(repo, 'bad.py').returncode == 0                    # git add the fixes, commit again
    assert _git(repo, 'show', 'HEAD:bad.py').stdout == FIXED


def test_an_error_ruff_cannot_fix_blocks_the_commit_with_its_file_line_and_rule(repo):
    (repo / 'long.py').write_text(LONG)
    result = _commit(repo, 'long.py')
    assert result.returncode != 0
    assert 'long.py:1:121' in result.stdout and 'E501' in result.stdout
    assert _commits(repo) == 1


def test_clean_python_and_other_files_commit_without_extra_steps(repo):
    (repo / 'clean.py').write_text(CLEAN)
    (repo / 'notes.md').write_text("Trailing spaces in Markdown are not Python   \n")
    assert _commit(repo, 'clean.py', 'notes.md').returncode == 0
    assert (repo / 'notes.md').read_text().endswith('   \n')             # only Python files are touched


def test_only_staged_files_are_checked_and_unstaged_edits_are_left_alone(repo):
    (repo / 'other.py').write_text(CLEAN)
    assert _commit(repo, 'other.py').returncode == 0
    (repo / 'other.py').write_text(FIXABLE)                            # edited, not staged
    (repo / 'clean.py').write_text(CLEAN)
    assert _commit(repo, 'clean.py').returncode == 0
    assert (repo / 'other.py').read_text() == FIXABLE                  # untouched
    assert _git(repo, 'show', 'HEAD:other.py').stdout == CLEAN         # and not committed


def test_branches_without_the_hook_config_still_commit(repo):
    # Worktrees share one hooks folder, and older branches have no .pre-commit-config.yaml
    _git(repo, 'checkout', '-q', '-b', 'old-branch')
    _git(repo, 'rm', '-q', '.pre-commit-config.yaml')
    (repo / 'bad.py').write_text(FIXABLE)
    result = _commit(repo, 'bad.py')
    assert result.returncode == 0, result.stdout
    assert (repo / 'bad.py').read_text() == FIXABLE                    # not checked there; CI still is


def test_the_docs_give_the_install_command():
    command = 'uv run pre-commit ' + ' '.join(INSTALL)
    assert command in (ROOT / 'README.md').read_text()
    assert command in (ROOT / 'DOCS' / 'CONTRIBUTING.md').read_text()
    assert command in (ROOT / '.pre-commit-config.yaml').read_text()


def test_files_excluded_from_linting_stay_excluded_when_staged(repo):
    (repo / 'tests' / 'fixtures').mkdir(parents=True)
    (repo / 'tests' / 'fixtures' / 'bad_code.py').write_text(FIXABLE)  # the security scanners' bad sample
    assert _commit(repo, 'tests/fixtures/bad_code.py').returncode == 0
    assert (repo / 'tests' / 'fixtures' / 'bad_code.py').read_text() == FIXABLE

"""Tests for project-root detection."""

import os

from trustee.utils import rootpath


class TestDetect:
    def test_finds_the_root_from_a_nested_directory(self, tmp_path):
        (tmp_path / ".git").mkdir()
        nested = tmp_path / "a" / "b" / "c"
        nested.mkdir(parents=True)
        (nested / "file.py").write_text("")
        assert rootpath.detect(str(nested)) == str(tmp_path)

    def test_finds_the_root_from_a_file_path(self, tmp_path):
        (tmp_path / ".git").mkdir()
        nested = tmp_path / "pkg"
        nested.mkdir()
        module = nested / "module.py"
        module.write_text("")
        assert rootpath.detect(str(module)) == str(tmp_path)

    def test_recognises_requirements_txt_as_a_root_marker(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("")
        nested = tmp_path / "src"
        nested.mkdir()
        (nested / "file.py").write_text("")
        assert rootpath.detect(str(nested)) == str(tmp_path)

    def test_accepts_a_custom_string_pattern(self, tmp_path):
        """The str check here replaced a six.string_types call."""
        (tmp_path / "setup.cfg").write_text("")
        nested = tmp_path / "src"
        nested.mkdir()
        (nested / "file.py").write_text("")
        assert rootpath.detect(str(nested), "setup.cfg") == str(tmp_path)

    def test_custom_pattern_ignores_the_default_markers(self, tmp_path):
        (tmp_path / ".git").mkdir()
        nested = tmp_path / "src"
        nested.mkdir()
        (nested / "file.py").write_text("")
        # .git is not the requested marker, so this must not resolve to tmp_path.
        assert rootpath.detect(str(nested), "pyproject.toml") != str(tmp_path)

    def test_defaults_to_the_current_directory(self, tmp_path, monkeypatch):
        (tmp_path / ".git").mkdir()
        monkeypatch.chdir(tmp_path)
        assert rootpath.detect() == str(tmp_path)

    def test_expands_user_relative_paths(self, tmp_path, monkeypatch):
        # expanduser reads HOME on POSIX but USERPROFILE on Windows, so set both
        # rather than making this test platform-specific.
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        (tmp_path / ".git").mkdir()
        assert rootpath.detect("~") == str(tmp_path)

    def test_returns_an_absolute_path(self, tmp_path, monkeypatch):
        (tmp_path / ".git").mkdir()
        monkeypatch.chdir(tmp_path)
        assert os.path.isabs(rootpath.detect("."))

    def test_finds_this_projects_own_root(self):
        detected = rootpath.detect(os.path.dirname(__file__))
        assert os.path.isfile(os.path.join(detected, "pyproject.toml"))

    def test_gives_up_inside_an_empty_directory(self, tmp_path):
        """Documents a real quirk: the walk aborts on the first empty directory.

        `find_root_path` returns None as soon as `listdir` comes back empty, so an
        empty subdirectory stops the search before it can reach the marker above it.
        """
        (tmp_path / ".git").mkdir()
        empty = tmp_path / "empty"
        empty.mkdir()
        assert rootpath.detect(str(empty)) is None

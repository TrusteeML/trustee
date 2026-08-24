"""Tests for the Logger helper."""

import logging

import pytest

from trustee.utils.log import Logger


@pytest.fixture
def log_path(tmp_path):
    return str(tmp_path / "output.log")


class TestLogger:
    def test_writes_the_message_to_its_file(self, log_path):
        Logger(path=log_path).log("hello world")
        with open(log_path) as handle:
            assert "hello world" in handle.read()

    def test_joins_multiple_arguments_with_spaces(self, log_path):
        Logger(path=log_path).log("alpha", "beta", "gamma")
        with open(log_path) as handle:
            assert "alpha beta gamma" in handle.read()

    def test_stringifies_non_string_arguments(self, log_path):
        Logger(path=log_path).log("count:", 42, ["a", "b"], {"k": 1})
        with open(log_path) as handle:
            body = handle.read()
        assert "count: 42" in body
        assert "['a', 'b']" in body

    def test_includes_the_level_name(self, log_path):
        Logger(path=log_path).log("an informational message")
        with open(log_path) as handle:
            assert "INFO" in handle.read()

    def test_honours_an_explicit_level(self, log_path):
        Logger(path=log_path).log("a warning message", level=logging.WARNING)
        with open(log_path) as handle:
            assert "WARNING" in handle.read()

    def test_filters_below_the_configured_level(self, log_path):
        Logger(path=log_path, level=logging.ERROR).log("should not appear", level=logging.INFO)
        with open(log_path) as handle:
            assert "should not appear" not in handle.read()

    def test_installs_a_stream_and_a_file_handler(self, log_path):
        logger = Logger(path=log_path)
        kinds = {type(h) for h in logger.handlers}
        assert logging.FileHandler in kinds
        assert any(issubclass(k, logging.StreamHandler) for k in kinds)

    def test_is_usable_as_a_standard_logger(self, log_path):
        logger = Logger(path=log_path)
        assert isinstance(logger, logging.Logger)
        logger.info("via the standard interface")
        with open(log_path) as handle:
            assert "via the standard interface" in handle.read()

    def test_appends_across_calls(self, log_path):
        logger = Logger(path=log_path)
        logger.log("first")
        logger.log("second")
        with open(log_path) as handle:
            body = handle.read()
        assert "first" in body and "second" in body

"""Tests for the CLI."""

from click.testing import CliRunner

from llamacpp_cli.cli import cli


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "llamacpp" in result.output


def test_cli_short_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["-h"])
    assert result.exit_code == 0
    assert "llamacpp" in result.output


def test_cli_version():
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_pull_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["pull", "--help"])
    assert result.exit_code == 0
    assert "MODEL" in result.output


def test_run_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["run", "--help"])
    assert result.exit_code == 0
    assert "MODEL" in result.output


def test_serve_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["serve", "--help"])
    assert result.exit_code == 0
    assert "--port" in result.output


def test_list_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["list", "--help"])
    assert result.exit_code == 0


def test_ps_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["ps", "--help"])
    assert result.exit_code == 0


def test_rm_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["rm", "--help"])
    assert result.exit_code == 0
    assert "MODEL" in result.output

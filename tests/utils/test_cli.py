from typing import List
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from kolena._utils.cli import base_command

TEST_API_TOKEN = "XXXX"


def test_b():
    runner = CliRunner()
    result = runner.invoke(base_command, ["--help"])
    assert result.exit_code == 0


def mock_initialize(tok: str):
    if tok != TEST_API_TOKEN:
        raise RuntimeError("bad token")


@pytest.mark.parametrize(
    "args,errmsg",
    [
        (["badsubcommand"], "No such command"),
        (["evaluator", "register"], "Missing option"),
        (["evaluator", "register", "--workflow", "my workflow", "--evaluator-name", "foo"], "Missing option"),
        (["evaluator", "register", "--workflow", "my workflow", "--image", "foo"], "Missing option"),
        (["evaluator", "register", "--evaluator-name", "foo", "--image", "bar"], "Missing option"),
        (["evaluator", "list"], "Missing option"),
        (["evaluator", "get", "--workflow", "my workflow"], "Missing option"),
        (["evaluator", "get", "--evaluator-name", "foo"], "Missing option"),
        (
            ["evaluator", "get", "--workflow", "my workflow", "--evaluator-name", "foo", "--include-secret", "bar"],
            "Got unexpected extra argument",
        ),
        (["repository", "create"], "Missing option"),
    ],
)
def test__cli_bad_input(args: List[str], errmsg: str) -> None:
    runner = CliRunner()
    result = runner.invoke(base_command, args)
    assert result.exception
    assert errmsg in result.output


def test__cli_evaluator_common_options() -> None:
    workflow = "my workflow"
    args = [
        "evaluator",
        "register",
        "--workflow",
        workflow,
        "--evaluator-name",
        "my evaluator",
        "--image",
        "ubuntu",
        "--secret",
        '"foobar"',
        "--api-token",
        TEST_API_TOKEN,
    ]
    runner = CliRunner()

    with patch("kolena.initialize", side_effect=mock_initialize):
        with patch("kolena.workflow.workflow.register_evaluator"):
            result = runner.invoke(base_command, args)
            assert not result.exception

    args = ["evaluator", "list", "--workflow", workflow, "--api-token", TEST_API_TOKEN]
    with patch("kolena.initialize", side_effect=mock_initialize):
        with patch("kolena.workflow.workflow.list_evaluators"):
            result = runner.invoke(base_command, args)
            assert not result.exception

    args = [
        "evaluator",
        "get",
        "--workflow",
        workflow,
        "--evaluator-name",
        "test-evaluator",
        "--api-token",
        TEST_API_TOKEN,
        "--no-include-secret",
    ]
    with patch("kolena.initialize", side_effect=mock_initialize):
        with patch("kolena.workflow.workflow.get_evaluator"):
            result = runner.invoke(base_command, args)
            assert not result.exception


def test__cli_repository() -> None:
    api_token = TEST_API_TOKEN
    args = [
        "repository",
        "create",
        "--name",
        "myrepo",
        "--api-token",
        api_token,
    ]
    runner = CliRunner()

    with patch("kolena.initialize", side_effect=mock_initialize):
        with patch("kolena._utils.repository.create"):
            result = runner.invoke(base_command, args)
            assert not result.exception

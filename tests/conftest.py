import pytest
from pytest import Item


def pytest_addoption(parser):
    parser.addoption(
        "--notebooks", action="store_true", default=False, help="run notebooks"
    )


def pytest_runtest_setup(item: Item) -> None:
    if "notebook" in item.keywords and not item.config.getoption("--notebooks"):
        pytest.skip("need --notebooks option to run")

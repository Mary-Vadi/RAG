import matplotlib

matplotlib.use("Agg")

import pytest

from tiny_models import build_tiny_bert, build_tiny_modernbert


@pytest.fixture(scope="session")
def tiny_bert(tmp_path_factory):
    return build_tiny_bert(tmp_path_factory.mktemp("tiny-bert"))


@pytest.fixture(scope="session")
def tiny_modernbert(tmp_path_factory):
    return build_tiny_modernbert(tmp_path_factory.mktemp("tiny-modernbert"))

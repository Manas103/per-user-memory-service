import pytest

from src import db
from src.embedding import train_embedder


@pytest.fixture(scope="session")
def embedder():
    model, _losses = train_embedder(epochs=30)
    return model


@pytest.fixture
def conn():
    c = db.get_conn()
    db.reset_schema(c)
    yield c
    c.close()

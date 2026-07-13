import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from objectcount.capture.server import create_app


def test_phone_page_served(tmp_path):
    client = TestClient(create_app(out_dir=tmp_path))
    response = client.get("/phone")
    assert response.status_code == 200
    assert "getUserMedia" in response.text
    assert "/ws/ingest" in response.text


def test_viewer_page_served(tmp_path):
    client = TestClient(create_app(out_dir=tmp_path))
    response = client.get("/")
    assert response.status_code == 200
    assert "/ws/view" in response.text

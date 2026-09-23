import pytest
from fastapi.testclient import TestClient

from job_agent.api import create_app
from tests.test_graph import make_graph


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(make_graph(tmp_path))) as c:
        yield c


def test_full_run_over_http(client):
    run = client.post("/runs", json={"search_term": "python", "location": "Europe"}).json()
    assert run["status"] == "waiting" and run["pending"]["type"] == "pick"
    assert [j["company"] for j in run["pending"]["shortlist"]] == ["Acme", "Data Co"]
    run_id = run["run_id"]

    run = client.post(f"/runs/{run_id}/resume", json={"chosen": [0]}).json()
    assert run["pending"]["type"] == "review" and "PhD" not in run["pending"]["letter"]

    run = client.post(f"/runs/{run_id}/resume", json={"action": "reject", "feedback": "warmer"}).json()
    assert run["pending"]["type"] == "review"

    run = client.post(f"/runs/{run_id}/resume", json={"action": "approve"}).json()
    assert run["status"] == "done" and run["pending"] is None and len(run["saved"]) == 1
    assert run["usage"]["llm_calls"] > 0 and run["usage"]["models"] == ["fake"]

    assert client.get(f"/runs/{run_id}").json()["status"] == "done"
    events = client.get(f"/runs/{run_id}/transcript").json()
    assert "user_review" in [e["type"] for e in events]
    markdown = client.get(f"/runs/{run_id}/transcript", params={"format": "markdown"})
    assert markdown.headers["content-type"].startswith("text/markdown") and "warmer" in markdown.text


def test_errors(client):
    assert client.get("/runs/nope").status_code == 404
    assert client.post("/runs/nope/resume", json={"chosen": [0]}).status_code == 404

    run_id = client.post("/runs", json={}).json()["run_id"]
    assert client.post(f"/runs/{run_id}/resume", json={"action": "approve"}).status_code == 422  # waiting for a pick
    client.post(f"/runs/{run_id}/resume", json={"chosen": []})  # pick nothing: run ends
    assert client.post(f"/runs/{run_id}/resume", json={"chosen": [0]}).status_code == 409


def test_two_runs_are_independent(client):
    a = client.post("/runs", json={}).json()["run_id"]
    b = client.post("/runs", json={}).json()["run_id"]
    client.post(f"/runs/{a}/resume", json={"chosen": [0]})
    assert client.get(f"/runs/{a}").json()["pending"]["type"] == "review"
    assert client.get(f"/runs/{b}").json()["pending"]["type"] == "pick"

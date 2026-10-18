"""The web_search tool: Google Search through Gemini, for what the book can't answer.

The network call is faked; what is tested is the contract the agent depends on --
prose it can speak, labelled as coming from the web, and an honest failure.
"""
from fastapi.testclient import TestClient

from hark.server import main


def _found(query, context=""):
    return {"text": "Dark matter is still unexplained.",
            "sources": [("nasa.gov", "https://nasa.gov/x"), ("cern.ch", "https://cern.ch/y")]}


def test_results_are_spoken_prose_labelled_as_the_web(monkeypatch):
    monkeypatch.setattr(main.local_llm, "google_search", _found)
    out = main.tool_web_search("sW", "is dark matter still unexplained")
    assert out["ok"]
    assert "Dark matter is still unexplained." in out["message"]
    assert "nasa.gov" in out["message"] and "not the book" in out["message"]
    assert out["sources"][0] == {"title": "nasa.gov", "url": "https://nasa.gov/x"}


def test_no_query_and_no_search_fail_honestly(monkeypatch):
    assert main.tool_web_search("sW", "  ")["ok"] is False
    monkeypatch.setattr(main.local_llm, "google_search", lambda q, context="": None)
    out = main.tool_web_search("sW", "anything")
    assert out["ok"] is False and "could not check online" in out["message"]


def test_the_query_is_bounded(monkeypatch):
    seen = {}

    def spy(query, context=""):
        seen["q"] = query
        return _found(query)

    monkeypatch.setattr(main.local_llm, "google_search", spy)
    main.tool_web_search("sW", "x" * 5000)
    assert len(seen["q"]) == 300


def test_local_dispatch_and_http_endpoint(monkeypatch):
    monkeypatch.setattr(main.local_llm, "google_search", _found)
    assert main.run_local_tool("sW", "web_search", {"query": "dark matter"})["ok"]
    r = TestClient(main.app).post("/tools/web_search?session_id=sW", json={"query": "dark matter"})
    assert r.status_code == 200 and r.json()["ok"]


def test_the_cloud_agent_and_the_router_both_offer_it():
    import json
    from pathlib import Path
    from hark.server.local_voice import NeedleRouter
    spec = json.loads((Path(main.__file__).parent / "agent.json").read_text(encoding="utf-8"))
    assert any(t["name"] == "web_search" for t in spec["tools"])
    assert any(t["name"] == "web_search" for t in NeedleRouter.TOOLS)

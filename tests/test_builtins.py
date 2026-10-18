"""Several built-in titles: the shelf lists each with its catalog metadata, and
a question goes to whichever one the listener's heartbeat names.

Uses the committed public-domain indexes in data/, and pins main's built-ins so
a local .env (HARK_BOOKS) cannot change what is being tested.
"""
import pytest
from fastapi.testclient import TestClient

from hark.core.library import Library
from hark.server import main

ROOT = main.ROOT


@pytest.fixture
def two_titles(monkeypatch):
    builtins = {"calculus": Library(ROOT / "data" / "calculus.db"),
                "einstein-relativity": Library(ROOT / "data" / "einstein-relativity.db")}
    catalog = {"einstein-relativity": {"id": "einstein-relativity", "title": "Relativity",
                                       "author": "Albert Einstein", "kind": "book",
                                       "topic": "Physics", "blurb": "Space and time.",
                                       "audio_url": "https://example.org/rel.mp3"}}
    monkeypatch.setattr(main, "BUILTINS", builtins)
    monkeypatch.setattr(main, "CATALOG", catalog)
    monkeypatch.setattr(main, "BOOK", "calculus")
    monkeypatch.setattr(main, "library", builtins["calculus"])
    return builtins


def test_the_shelf_lists_every_built_in_with_its_metadata(two_titles):
    books = TestClient(main.app).get("/api/books").json()["books"]
    built = [b for b in books if b.get("builtin")]
    assert [b["id"] for b in built] == ["calculus", "einstein-relativity"]
    rel = built[1]
    assert (rel["title"], rel["author"], rel["kind"], rel["topic"], rel["blurb"]) == (
        "Relativity", "Albert Einstein", "book", "Physics", "Space and time.")
    assert rel["audio_url"] == "https://example.org/rel.mp3"
    assert rel["duration"] > 0 and rel["chunks"] == len(two_titles["einstein-relativity"])
    # A title the catalog does not describe still gets a card, served locally.
    assert built[0]["audio_url"] == "/audio/calculus.mp3"


def _passage(client, sid, book, seconds):
    client.post("/api/playhead", json={"session_id": sid, "seconds": seconds, "book": book})
    return client.post(f"/tools/passage_at_playhead?session_id={sid}",
                       json={"session_id": sid}).json()["message"]


def test_the_heartbeat_picks_which_built_in_answers(two_titles):
    client = TestClient(main.app)
    rel = _passage(client, "sRel", "einstein-relativity", 300)
    calc = _passage(client, "sCalc", "calculus", 300)
    rel_text = " ".join(c.text for c in two_titles["einstein-relativity"].window(300))
    calc_text = " ".join(c.text for c in two_titles["calculus"].window(300))
    assert rel_text[:60] in rel and calc_text[:60] not in rel
    assert calc_text[:60] in calc


def test_an_unknown_book_falls_back_to_the_default(two_titles):
    main.playheads.set_book("sGone", "no-such-book")
    lib, title = main.library_for("sGone")
    assert lib is two_titles["calculus"]


def test_the_outline_names_the_built_in_by_its_title(two_titles):
    main.playheads.set_book("sTitle", "einstein-relativity")
    _lib, title = main.library_for("sTitle")
    assert title == "Relativity"


def test_a_missing_index_is_skipped_not_fatal():
    opened = main._open_builtins(["calculus", "not-built-yet"])
    assert list(opened) == ["calculus"]

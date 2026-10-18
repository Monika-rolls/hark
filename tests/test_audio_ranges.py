"""Audio is served in capped ranges: Cloud Run refuses an HTTP/1 response over
32 MiB, and the <audio> element's first request is the open range "bytes=0-"."""
from fastapi.testclient import TestClient

from hark.server import main

URL = "/audio/calculus.mp3"


def _size():
    return (main.AUDIO / "calculus.mp3").stat().st_size


def test_an_open_range_is_capped(monkeypatch):
    monkeypatch.setattr(main, "AUDIO_CHUNK", 1000)
    r = TestClient(main.app).get(URL, headers={"Range": "bytes=0-"})
    assert r.status_code == 206 and len(r.content) == 1000
    assert r.headers["content-range"] == f"bytes 0-999/{_size()}"
    assert r.headers["content-type"] == "audio/mpeg"


def test_the_browser_can_ask_for_the_next_piece_and_the_tail(monkeypatch):
    monkeypatch.setattr(main, "AUDIO_CHUNK", 1000)
    client = TestClient(main.app)
    r = client.get(URL, headers={"Range": "bytes=1000-"})
    assert r.headers["content-range"].startswith("bytes 1000-1999/")
    tail = client.get(URL, headers={"Range": "bytes=-10"})
    assert tail.status_code == 206 and len(tail.content) == 10
    assert client.get(URL, headers={"Range": f"bytes={_size()}-"}).status_code == 416


def test_a_plain_request_streams_the_whole_file():
    r = TestClient(main.app).get(URL)
    assert r.status_code == 200 and len(r.content) == _size()


def test_nothing_but_audio_in_that_folder():
    client = TestClient(main.app)
    for bad in ("/audio/..%2F..%2Fhark%2Fserver%2Fmain.py", "/audio/missing.mp3",
                "/media/podcast-32k.db", "/media/catalog.local.json"):
        assert client.get(bad).status_code == 404, bad

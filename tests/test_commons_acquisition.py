import hashlib
import io
import json
from datetime import datetime, timezone
from email.message import Message
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
import pytest

from tools.training.acquire_commons import (
    CommonsPolicy,
    acquire_commons,
    iter_category_files,
    main,
)


class Response:
    def __init__(self, body: bytes, content_length: int | None = None):
        self._stream = io.BytesIO(body)
        self.closed = False
        self.headers = Message()
        if content_length is not None:
            self.headers["Content-Length"] = str(content_length)

    def read(self, size: int = -1) -> bytes:
        return self._stream.read(size)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
        return False

    def close(self):
        self.closed = True


class InterruptedResponse(Response):
    def __init__(self, body: bytes):
        super().__init__(body)
        self._reads = 0

    def read(self, size: int = -1) -> bytes:
        self._reads += 1
        if self._reads > 1:
            raise OSError("connection interrupted")
        return self._stream.read(min(size, 8))


class FakeOpener:
    def __init__(self, responses):
        self.responses = list(responses)
        self.urls: list[str] = []

    def __call__(self, request):
        self.urls.append(getattr(request, "full_url", str(request)))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def api_response(payload: dict) -> Response:
    return Response(json.dumps(payload).encode("utf-8"))


def image_bytes(extension: str = ".jpg", value: int = 80) -> bytes:
    image = np.full((24, 32, 3), value, dtype=np.uint8)
    ok, encoded = cv2.imencode(extension, image)
    assert ok
    return encoded.tobytes()


def valid_imageinfo(url: str, **overrides) -> dict:
    info = {
        "url": url,
        "width": 640,
        "height": 480,
        "size": 2048,
        "mime": "image/jpeg",
        "mediatype": "BITMAP",
        "extmetadata": {
            "Artist": {"value": "Example Author"},
            "LicenseShortName": {"value": "CC BY-SA 4.0"},
            "LicenseUrl": {
                "value": "https://creativecommons.org/licenses/by-sa/4.0/"
            },
        },
    }
    info.update(overrides)
    return info


def write_policy(path: Path, **overrides) -> Path:
    policy = {
        "api_url": "https://commons.wikimedia.org/w/api.php",
        "category": "Category:Toilet paper rolls",
        "max_files": 100,
        "max_bytes": 20 * 1024 * 1024,
        "min_width": 480,
        "min_height": 360,
        "accepted_mime_types": ["image/jpeg", "image/png", "image/webp"],
        "accepted_licenses": [
            "CC0",
            "Public domain",
            "CC BY 2.0",
            "CC BY 3.0",
            "CC BY 4.0",
            "CC BY-SA 2.0",
            "CC BY-SA 3.0",
            "CC BY-SA 4.0",
        ],
        "user_agent": "ObjectCount/0.1 (training data acquisition)",
        "request_delay_seconds": 0.0,
        "max_retries": 1,
        "retry_backoff_seconds": [0.0],
    }
    policy.update(overrides)
    path.write_text(json.dumps(policy), encoding="utf-8")
    return path


def page(page_id: int, title: str, info: dict) -> dict:
    return {"pageid": page_id, "title": title, "imageinfo": [info]}


def api_pages(*pages: dict, continuation: str | None = None) -> dict:
    payload = {"query": {"pages": list(pages)}}
    if continuation:
        payload["continue"] = {"gcmcontinue": continuation, "continue": "gcmcontinue||"}
    return payload


def acquisition_opener(*file_responses, pages: list[dict] | None = None) -> FakeOpener:
    pages = pages or [page(index + 1, f"File:{index}.jpg", valid_imageinfo(
        f"https://upload/{index}.jpg")) for index in range(len(file_responses))]
    return FakeOpener([api_response(api_pages(*pages)), *file_responses])


def test_commons_policy_loads_exact_category_and_limits(tmp_path):
    policy = CommonsPolicy.from_json(write_policy(tmp_path / "commons.json"))

    assert policy.category == "Category:Toilet paper rolls"
    assert policy.max_files == 100
    assert policy.max_bytes == 20 * 1024 * 1024
    assert (policy.min_width, policy.min_height) == (480, 360)
    assert policy.accepted_mime_types == ("image/jpeg", "image/png", "image/webp")


def test_iter_category_files_follows_continue_and_yields_only_raster_originals(tmp_path):
    opener = FakeOpener([
        api_response(api_pages(
            page(1, "File:One.jpg", valid_imageinfo("https://upload/one.jpg")),
            page(9, "File:Vector.svg", valid_imageinfo(
                "https://upload/vector.svg", mime="image/svg+xml", mediatype="DRAWING")),
            continuation="next",
        )),
        api_response(api_pages(
            page(2, "File:Two.png", valid_imageinfo(
                "https://upload/two.png", mime="image/png")),
        )),
    ])

    files = list(iter_category_files(
        CommonsPolicy.from_json(write_policy(tmp_path / "commons.json")), opener))

    assert [item["page_id"] for item in files] == [1, 2]
    second_query = parse_qs(urlparse(opener.urls[1]).query)
    assert second_query["gcmcontinue"] == ["next"]
    first_query = parse_qs(urlparse(opener.urls[0]).query)
    assert first_query["generator"] == ["categorymembers"]
    assert first_query["gcmnamespace"] == ["6"]
    assert first_query["prop"] == ["imageinfo"]
    assert first_query["iiprop"] == ["url|size|mime|extmetadata"]
    assert files[0]["url"] == "https://upload/one.jpg"


def test_iter_category_files_closes_retryable_api_http_error(tmp_path):
    failed_response = Response(b"unavailable")
    unavailable = HTTPError(
        "https://commons.wikimedia.org/w/api.php", 503, "unavailable", {},
        failed_response,
    )
    opener = FakeOpener([unavailable, api_response(api_pages())])

    assert list(iter_category_files(
        CommonsPolicy.from_json(write_policy(tmp_path / "commons.json")),
        opener,
    )) == []
    assert failed_response.closed


def test_iter_category_files_closes_api_http_errors_when_retry_is_exhausted(
        tmp_path):
    responses = [Response(b"unavailable"), Response(b"still unavailable")]
    opener = FakeOpener([
        HTTPError("https://commons.wikimedia.org/w/api.php", 503,
                  "unavailable", {}, response)
        for response in responses
    ])

    with pytest.raises(HTTPError):
        list(iter_category_files(
            CommonsPolicy.from_json(write_policy(tmp_path / "commons.json")),
            opener,
        ))
    assert all(response.closed for response in responses)


def manifest_items(output: Path) -> list[dict]:
    return json.loads((output / "manifest.json").read_text(encoding="utf-8"))["items"]


def test_acquire_rejects_missing_author_and_license_url(tmp_path):
    missing_author = valid_imageinfo("https://upload/author.jpg")
    del missing_author["extmetadata"]["Artist"]
    missing_license_url = valid_imageinfo("https://upload/license.jpg")
    del missing_license_url["extmetadata"]["LicenseUrl"]
    opener = acquisition_opener(
        pages=[page(1, "File:Author.jpg", missing_author),
               page(2, "File:License.jpg", missing_license_url)],
    )

    acquire_commons(write_policy(tmp_path / "policy.json"), tmp_path / "out", opener)

    assert [(item["status"], item["reason"]) for item in manifest_items(tmp_path / "out")] == [
        ("rejected", "missing-author"),
        ("rejected", "missing-license-url"),
    ]
    assert len(opener.urls) == 1


def test_acquire_rejects_mime_and_undersized_images_before_download(tmp_path):
    opener = acquisition_opener(
        pages=[
            page(1, "File:Raw.tiff", valid_imageinfo(
                "https://upload/raw.tiff", mime="image/tiff")),
            page(2, "File:Small.jpg", valid_imageinfo(
                "https://upload/small.jpg", width=479)),
        ],
    )

    acquire_commons(write_policy(tmp_path / "policy.json"), tmp_path / "out", opener)

    assert [item["reason"] for item in manifest_items(tmp_path / "out")] == [
        "rejected-mime", "undersized-image"]
    assert len(opener.urls) == 1


def test_acquire_rejects_oversized_content_length_without_writing_image(tmp_path):
    jpeg = image_bytes()
    response = Response(jpeg, content_length=len(jpeg))
    opener = acquisition_opener(response)

    acquire_commons(
        write_policy(tmp_path / "policy.json", max_bytes=len(jpeg) - 1),
        tmp_path / "out", opener)

    assert manifest_items(tmp_path / "out")[0]["reason"] == "content-length-exceeds-limit"
    assert not list((tmp_path / "out/images").iterdir())
    assert response.closed


def test_acquire_aborts_stream_that_exceeds_size_limit(tmp_path):
    jpeg = image_bytes()
    opener = acquisition_opener(Response(jpeg))

    acquire_commons(
        write_policy(tmp_path / "policy.json", max_bytes=len(jpeg) - 1),
        tmp_path / "out", opener)

    assert manifest_items(tmp_path / "out")[0]["reason"] == "download-exceeds-limit"
    assert not list((tmp_path / "out/images").iterdir())


def test_acquire_accepts_once_and_rejects_duplicate_content(tmp_path):
    jpeg = image_bytes()
    opener = acquisition_opener(Response(jpeg), Response(jpeg))

    manifest = acquire_commons(
        write_policy(tmp_path / "policy.json"), tmp_path / "out", opener)

    digest = hashlib.sha256(jpeg).hexdigest()
    assert manifest["items"][0]["status"] == "accepted"
    assert manifest["items"][0]["sha256"] == digest
    assert manifest["items"][0]["path"] == f"images/{digest[:16]}.jpg"
    assert manifest["items"][1]["status"] == "rejected"
    assert manifest["items"][1]["reason"] == "duplicate-content"
    assert len(list((tmp_path / "out/images").iterdir())) == 1


def test_acquire_records_stable_page_url_and_deterministic_utc_retrieval_time(
        tmp_path):
    retrieved_at = datetime(2026, 7, 13, 1, 2, 3, tzinfo=timezone.utc)

    manifest = acquire_commons(
        write_policy(tmp_path / "policy.json"),
        tmp_path / "out",
        acquisition_opener(Response(image_bytes())),
        clock=lambda: retrieved_at,
    )

    item = manifest["items"][0]
    assert item["page_url"] == "https://commons.wikimedia.org/?curid=1"
    assert item["retrieved_at"] == "2026-07-13T01:02:03Z"


def test_acquire_records_interrupted_download_and_removes_temporary_file(tmp_path):
    opener = acquisition_opener(InterruptedResponse(image_bytes()))

    acquire_commons(write_policy(tmp_path / "policy.json"), tmp_path / "out", opener)

    item = manifest_items(tmp_path / "out")[0]
    assert item["status"] == "failed"
    assert item["reason"] == "download-interrupted"
    assert not list((tmp_path / "out/images").iterdir())


def test_acquire_retries_one_http_503_then_accepts(tmp_path):
    jpeg = image_bytes()
    failed_response = Response(b"unavailable")
    unavailable = HTTPError(
        "https://upload/0.jpg", 503, "unavailable", {}, failed_response)
    opener = acquisition_opener(
        unavailable,
        Response(jpeg),
        pages=[page(1, "File:0.jpg", valid_imageinfo("https://upload/0.jpg"))],
    )

    manifest = acquire_commons(
        write_policy(tmp_path / "policy.json"), tmp_path / "out", opener)

    assert manifest["items"][0]["status"] == "accepted"
    assert opener.urls.count("https://upload/0.jpg") == 2
    assert failed_response.closed


def test_acquire_honors_capped_retry_after_for_http_429(tmp_path):
    jpeg = image_bytes()
    failed_response = Response(b"rate limited")
    error = HTTPError(
        "https://upload/0.jpg", 429, "rate limited",
        {"Retry-After": "999"}, failed_response)
    sleeps = []

    manifest = acquire_commons(
        write_policy(tmp_path / "policy.json"), tmp_path / "out",
        acquisition_opener(
            error, Response(jpeg),
            pages=[page(1, "File:0.jpg", valid_imageinfo("https://upload/0.jpg"))],
        ),
        sleeper=sleeps.append,
    )

    assert manifest["items"][0]["status"] == "accepted"
    assert 120.0 in sleeps
    assert failed_response.closed


def test_acquire_replaces_failed_row_when_resuming(tmp_path):
    policy_path = write_policy(tmp_path / "policy.json", max_retries=0)
    output = tmp_path / "out"
    first = acquisition_opener(
        HTTPError("https://upload/0.jpg", 429, "rate limited", {}, None),
        pages=[page(1, "File:0.jpg", valid_imageinfo("https://upload/0.jpg"))],
    )
    acquire_commons(policy_path, output, first, sleeper=lambda _: None)

    second = acquisition_opener(
        Response(image_bytes()),
        pages=[page(1, "File:0.jpg", valid_imageinfo("https://upload/0.jpg"))],
    )
    manifest = acquire_commons(
        policy_path, output, second, sleeper=lambda _: None)

    assert len(manifest["items"]) == 1
    assert manifest["items"][0]["page_id"] == 1
    assert manifest["items"][0]["status"] == "accepted"


def test_acquire_stops_after_exhausted_http_429(tmp_path):
    policy_path = write_policy(tmp_path / "policy.json", max_retries=0)
    opener = acquisition_opener(
        HTTPError("https://upload/0.jpg", 429, "rate limited", {}, None),
        Response(image_bytes()),
    )

    manifest = acquire_commons(
        policy_path, tmp_path / "out", opener, sleeper=lambda _: None)

    assert len(manifest["items"]) == 1
    assert manifest["items"][0]["reason"] == "http-error-429"
    assert "https://upload/1.jpg" not in opener.urls


def test_acquire_closes_both_image_http_errors_when_retry_is_exhausted(tmp_path):
    responses = [Response(b"unavailable"), Response(b"still unavailable")]
    errors = [
        HTTPError("https://upload/0.jpg", 503, "unavailable", {}, response)
        for response in responses
    ]
    opener = acquisition_opener(
        *errors,
        pages=[page(1, "File:0.jpg", valid_imageinfo("https://upload/0.jpg"))],
    )

    manifest = acquire_commons(
        write_policy(tmp_path / "policy.json"), tmp_path / "out", opener)

    assert manifest["items"][0]["reason"] == "http-error-503"
    assert all(response.closed for response in responses)


def test_acquire_closes_response_with_malformed_content_length(tmp_path):
    response = Response(image_bytes())
    response.headers["Content-Length"] = "not-an-integer"

    manifest = acquire_commons(
        write_policy(tmp_path / "policy.json"), tmp_path / "out",
        acquisition_opener(response),
    )

    assert manifest["items"][0]["status"] == "failed"
    assert response.closed


def test_acquire_writes_manifest_after_each_item_and_resumes_without_duplicates(
        tmp_path):
    jpeg = image_bytes()
    policy_path = write_policy(tmp_path / "policy.json")
    output = tmp_path / "out"
    acquire_commons(policy_path, output, acquisition_opener(Response(jpeg)))
    second = acquisition_opener(Response(jpeg), pages=[
        page(1, "File:0.jpg", valid_imageinfo("https://upload/0.jpg")),
        page(2, "File:New.jpg", valid_imageinfo("https://upload/new.jpg")),
    ])

    manifest = acquire_commons(policy_path, output, second)

    assert [item["page_id"] for item in manifest["items"]] == [1, 2]
    assert manifest["items"][1]["reason"] == "duplicate-content"
    assert second.urls == [
        second.urls[0],
        "https://upload/new.jpg",
    ]


def test_acquire_writes_empty_manifest_when_category_has_no_files(tmp_path):
    output = tmp_path / "out"

    manifest = acquire_commons(
        write_policy(tmp_path / "policy.json"),
        output,
        FakeOpener([api_response(api_pages())]),
    )

    assert manifest["items"] == []
    assert manifest_items(output) == []


def test_acquire_fails_clearly_on_malformed_existing_manifest(tmp_path):
    output = tmp_path / "out"
    output.mkdir()
    (output / "manifest.json").write_text("{broken", encoding="utf-8")

    with pytest.raises(ValueError, match="malformed Commons manifest"):
        acquire_commons(
            write_policy(tmp_path / "policy.json"), output,
            FakeOpener([]),
        )


@pytest.mark.parametrize(("field", "value"), [
    ("schema_version", 99),
    ("source", "Another source"),
    ("category", "Category:Paper towels"),
])
def test_acquire_rejects_incompatible_existing_manifest_identity(
        tmp_path, field, value):
    output = tmp_path / "out"
    output.mkdir()
    manifest = {
        "schema_version": 1,
        "source": "Wikimedia Commons",
        "category": "Category:Toilet paper rolls",
        "items": [],
    }
    manifest[field] = value
    (output / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=field):
        acquire_commons(
            write_policy(tmp_path / "policy.json"), output,
            FakeOpener([]),
        )


def test_acquire_rejects_malformed_existing_manifest_item(tmp_path):
    output = tmp_path / "out"
    output.mkdir()
    manifest = {
        "schema_version": 1,
        "source": "Wikimedia Commons",
        "category": "Category:Toilet paper rolls",
        "items": [{"page_id": 1}],
    }
    (output / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="item 0 missing required fields"):
        acquire_commons(
            write_policy(tmp_path / "policy.json"), output,
            FakeOpener([]),
        )


def test_acquire_rejects_accepted_item_whose_image_is_missing(tmp_path):
    policy_path = write_policy(tmp_path / "policy.json")
    output = tmp_path / "out"
    manifest = acquire_commons(
        policy_path, output, acquisition_opener(Response(image_bytes())))
    (output / manifest["items"][0]["path"]).unlink()

    with pytest.raises(ValueError, match="accepted path does not exist"):
        acquire_commons(policy_path, output, FakeOpener([]))


def test_acquire_rejects_accepted_item_with_inconsistent_path(tmp_path):
    policy_path = write_policy(tmp_path / "policy.json")
    output = tmp_path / "out"
    acquire_commons(
        policy_path, output, acquisition_opener(Response(image_bytes())))
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["items"][0]["path"] = "../outside.jpg"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="path is inconsistent"):
        acquire_commons(policy_path, output, FakeOpener([]))


def test_acquire_removes_manifest_temporary_file_when_replace_fails(
        tmp_path, monkeypatch):
    output = tmp_path / "out"

    def fail_replace(source, destination):
        raise OSError("replace failed")

    monkeypatch.setattr("tools.training.acquire_commons.os.replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        acquire_commons(
            write_policy(tmp_path / "policy.json"), output,
            FakeOpener([api_response(api_pages())]),
        )
    assert not (output / ".manifest.json.tmp").exists()


def test_cli_acquisition_creates_artifacts_and_prints_totals(tmp_path, capsys):
    output = tmp_path / "out"

    exit_code = main([
        "--config", str(write_policy(tmp_path / "policy.json")),
        "--out", str(output),
    ], opener=acquisition_opener(Response(image_bytes())))

    assert exit_code == 0
    assert (output / "manifest.json").is_file()
    assert len(list((output / "images").iterdir())) == 1
    assert "accepted=1 rejected=0 failed=0" in capsys.readouterr().out


def test_cli_verify_checks_manifest_and_prints_totals(tmp_path, capsys):
    output = tmp_path / "out"
    acquire_commons(
        write_policy(tmp_path / "policy.json"), output,
        acquisition_opener(Response(image_bytes())),
        clock=lambda: datetime(2026, 7, 13, tzinfo=timezone.utc),
    )
    capsys.readouterr()

    exit_code = main(["--verify", str(output)])

    assert exit_code == 0
    assert "accepted=1 rejected=0 failed=0" in capsys.readouterr().out


def test_cli_returns_nonzero_when_acquisition_or_verification_fails(
        tmp_path, capsys):
    missing_policy = main([
        "--config", str(tmp_path / "missing.json"),
        "--out", str(tmp_path / "out"),
    ])
    broken_output = tmp_path / "broken"
    broken_output.mkdir()
    (broken_output / "manifest.json").write_text("{broken", encoding="utf-8")

    broken_manifest = main(["--verify", str(broken_output)])

    assert missing_policy == 1
    assert broken_manifest == 1
    assert capsys.readouterr().err.count("error:") == 2

"""Wikimedia Commons에서 라이선스가 확인된 학습 이미지를 수집한다."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import cv2
import numpy as np


class UrlResponse(Protocol):
    headers: Any

    def read(self, size: int = -1) -> bytes: ...


UrlOpener = Callable[[Request], UrlResponse]


@dataclass(frozen=True)
class CommonsPolicy:
    api_url: str
    category: str
    max_files: int
    max_bytes: int
    min_width: int
    min_height: int
    accepted_mime_types: tuple[str, ...]
    accepted_licenses: tuple[str, ...]
    user_agent: str

    @classmethod
    def from_json(cls, path: Path) -> "CommonsPolicy":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        payload["accepted_mime_types"] = tuple(payload["accepted_mime_types"])
        payload["accepted_licenses"] = tuple(payload["accepted_licenses"])
        return cls(**payload)


def _metadata_value(metadata: dict, name: str) -> str | None:
    field = metadata.get(name)
    if not isinstance(field, dict):
        return None
    value = field.get("value")
    return value if isinstance(value, str) and value else None


def _open_with_retry(request: Request, opener: UrlOpener) -> UrlResponse:
    try:
        return opener(request)
    except HTTPError as error:
        if error.code != 503:
            raise
    return opener(request)


def _open_json(request: Request, opener: UrlOpener) -> dict:
    response = _open_with_retry(request, opener)
    if hasattr(response, "__enter__"):
        with response:
            return json.loads(response.read())
    return json.loads(response.read())


def iter_category_files(
        policy: CommonsPolicy, opener: UrlOpener) -> Iterator[dict]:
    continuation: str | None = None
    while True:
        parameters = {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "generator": "categorymembers",
            "gcmtitle": policy.category,
            "gcmnamespace": "6",
            "gcmtype": "file",
            "gcmlimit": "max",
            "prop": "imageinfo",
            "iiprop": "url|size|mime|extmetadata",
        }
        if continuation is not None:
            parameters["gcmcontinue"] = continuation
        request = Request(
            f"{policy.api_url}?{urlencode(parameters)}",
            headers={"User-Agent": policy.user_agent},
        )
        payload = _open_json(request, opener)
        pages = payload.get("query", {}).get("pages", [])
        if isinstance(pages, dict):
            pages = pages.values()
        for page in pages:
            imageinfo = page.get("imageinfo") or []
            if not imageinfo:
                continue
            info = imageinfo[0]
            mime_type = info.get("mime")
            if (not isinstance(mime_type, str)
                    or not mime_type.startswith("image/")
                    or mime_type == "image/svg+xml"
                    or not info.get("url")):
                continue
            metadata = info.get("extmetadata") or {}
            yield {
                "page_id": page.get("pageid"),
                "title": page.get("title"),
                "url": info["url"],
                "width": info.get("width"),
                "height": info.get("height"),
                "size": info.get("size"),
                "mime_type": mime_type,
                "author": _metadata_value(metadata, "Artist"),
                "license": _metadata_value(metadata, "LicenseShortName"),
                "license_url": _metadata_value(metadata, "LicenseUrl"),
                "metadata": {
                    name: field.get("value")
                    for name, field in metadata.items()
                    if isinstance(field, dict) and isinstance(field.get("value"), str)
                },
            }
        continuation = payload.get("continue", {}).get("gcmcontinue")
        if continuation is None:
            return


def acquire_commons(
        policy_path: Path, output: Path, opener: UrlOpener = urlopen) -> dict:
    policy = CommonsPolicy.from_json(policy_path)
    output = Path(output)
    images = output / "images"
    images.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {
            "schema_version": 1,
            "source": "Wikimedia Commons",
            "category": policy.category,
            "items": [],
        }

    completed_page_ids = {item.get("page_id") for item in manifest["items"]}
    known_hashes = {
        item["sha256"] for item in manifest["items"] if item.get("sha256")
    }
    accepted_count = sum(
        item.get("status") == "accepted" for item in manifest["items"])
    _write_manifest(manifest_path, manifest)

    for source in iter_category_files(policy, opener):
        if accepted_count >= policy.max_files:
            break
        if source["page_id"] in completed_page_ids:
            continue
        item = dict(source)
        reason = _metadata_rejection(source, policy)
        if reason is not None:
            item.update(status="rejected", reason=reason)
        else:
            _acquire_file(item, policy, images, known_hashes, opener)
            if item["status"] == "accepted":
                accepted_count += 1
        manifest["items"].append(item)
        completed_page_ids.add(item["page_id"])
        _write_manifest(manifest_path, manifest)
    return manifest


def _metadata_rejection(source: dict, policy: CommonsPolicy) -> str | None:
    if not source.get("author"):
        return "missing-author"
    if not source.get("license"):
        return "missing-license"
    if not source.get("license_url"):
        return "missing-license-url"
    if source["license"] not in policy.accepted_licenses:
        return "rejected-license"
    if source.get("mime_type") not in policy.accepted_mime_types:
        return "rejected-mime"
    width = source.get("width")
    height = source.get("height")
    if (not isinstance(width, int) or not isinstance(height, int)
            or width < policy.min_width or height < policy.min_height):
        return "undersized-image"
    return None


class _DownloadRejected(Exception):
    pass


def _acquire_file(item: dict, policy: CommonsPolicy, images: Path,
                  known_hashes: set[str], opener: UrlOpener) -> None:
    temporary: Path | None = None
    try:
        request = Request(item["url"], headers={"User-Agent": policy.user_agent})
        response = _open_with_retry(request, opener)
        content_length = response.headers.get("Content-Length")
        if content_length is not None and int(content_length) > policy.max_bytes:
            close = getattr(response, "close", None)
            if close is not None:
                close()
            item.update(status="rejected", reason="content-length-exceeds-limit")
            return
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".download-", suffix=".tmp", dir=images)
        os.close(descriptor)
        temporary = Path(temporary_name)
        digest = hashlib.sha256()
        total = 0
        try:
            with temporary.open("wb") as destination:
                while True:
                    chunk = response.read(64 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > policy.max_bytes:
                        raise _DownloadRejected("download-exceeds-limit")
                    destination.write(chunk)
                    digest.update(chunk)
        finally:
            close = getattr(response, "close", None)
            if close is not None:
                close()

        encoded = np.frombuffer(temporary.read_bytes(), dtype=np.uint8)
        if cv2.imdecode(encoded, cv2.IMREAD_COLOR) is None:
            raise _DownloadRejected("invalid-image")
        sha256 = digest.hexdigest()
        item["sha256"] = sha256
        if sha256 in known_hashes:
            item.update(status="rejected", reason="duplicate-content")
            return
        extension = {
            "image/jpeg": ".jpg",
            "image/png": ".png",
            "image/webp": ".webp",
        }[item["mime_type"]]
        relative_path = Path("images") / f"{sha256[:16]}{extension}"
        os.replace(temporary, images.parent / relative_path)
        temporary = None
        known_hashes.add(sha256)
        item.update(
            status="accepted",
            reason="accepted",
            path=relative_path.as_posix(),
            bytes=total,
        )
    except _DownloadRejected as error:
        item.update(status="rejected", reason=str(error))
    except HTTPError as error:
        item.update(status="failed", reason=f"http-error-{error.code}")
    except (OSError, ValueError):
        item.update(status="failed", reason="download-interrupted")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _write_manifest(path: Path, manifest: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)

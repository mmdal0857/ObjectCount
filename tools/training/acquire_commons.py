"""Wikimedia Commons에서 라이선스가 확인된 학습 이미지를 수집한다."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
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
Clock = Callable[[], datetime]


class ManifestValidationError(ValueError):
    """Existing acquisition state is unsafe to resume or verify."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _format_utc(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("retrieval clock must return a timezone-aware datetime")
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z")


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
    for attempt in range(2):
        try:
            return opener(request)
        except HTTPError as error:
            _close_response(error)
            if error.code != 503 or attempt == 1:
                raise
    raise AssertionError("HTTP retry loop did not return or raise")


def _close_response(response: object) -> None:
    close = getattr(response, "close", None)
    if close is not None:
        close()


def _open_json(request: Request, opener: UrlOpener) -> dict:
    response = _open_with_retry(request, opener)
    try:
        return json.loads(response.read())
    finally:
        _close_response(response)


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
                "page_url": (
                    "https://commons.wikimedia.org/?curid="
                    f"{page.get('pageid')}"
                ),
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
        policy_path: Path, output: Path, opener: UrlOpener = urlopen,
        *, clock: Clock = _utc_now) -> dict:
    policy = CommonsPolicy.from_json(policy_path)
    output = Path(output)
    images = output / "images"
    images.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.is_file():
        manifest = _load_manifest(manifest_path)
        _validate_manifest_identity(manifest, expected_category=policy.category)
        _validate_manifest_items(manifest, output=output)
    else:
        manifest = {
            "schema_version": 1,
            "source": "Wikimedia Commons",
            "category": policy.category,
            "items": [],
        }

    completed_page_ids = {item.get("page_id") for item in manifest["items"]}
    known_hashes = {
        item["sha256"] for item in manifest["items"]
        if item.get("status") == "accepted"
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
        item["retrieved_at"] = _format_utc(clock())
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
    response: UrlResponse | None = None
    try:
        request = Request(item["url"], headers={"User-Agent": policy.user_agent})
        response = _open_with_retry(request, opener)
        content_length = response.headers.get("Content-Length")
        if content_length is not None and int(content_length) > policy.max_bytes:
            item.update(status="rejected", reason="content-length-exceeds-limit")
            return
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".download-", suffix=".tmp", dir=images)
        os.close(descriptor)
        temporary = Path(temporary_name)
        digest = hashlib.sha256()
        total = 0
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
        if response is not None:
            _close_response(response)
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _write_manifest(path: Path, manifest: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    operation_failed = False
    try:
        temporary.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except BaseException:
        operation_failed = True
        raise
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            if not operation_failed:
                raise


def _print_summary(manifest: dict) -> None:
    totals = {
        status: sum(item.get("status") == status for item in manifest["items"])
        for status in ("accepted", "rejected", "failed")
    }
    print(" ".join(f"{status}={totals[status]}" for status in totals))


def verify_commons(output: Path) -> dict:
    output = Path(output)
    manifest_path = output / "manifest.json"
    manifest = _load_manifest(manifest_path)
    _validate_manifest_identity(manifest)
    _validate_manifest_items(manifest, output=output)
    return manifest


def _load_manifest(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ManifestValidationError(
            f"malformed Commons manifest {path}: {error.msg}") from error


def _validate_manifest_identity(
        manifest: dict, *, expected_category: str | None = None) -> None:
    if not isinstance(manifest, dict):
        raise ManifestValidationError("manifest must be a JSON object")
    expected = {
        "schema_version": 1,
        "source": "Wikimedia Commons",
    }
    if expected_category is not None:
        expected["category"] = expected_category
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise ManifestValidationError(
                f"manifest {field} must be {value!r}, got {manifest.get(field)!r}")
    if not isinstance(manifest.get("category"), str) or not manifest["category"]:
        raise ManifestValidationError("manifest category must be a non-empty string")
    if not isinstance(manifest.get("items"), list):
        raise ManifestValidationError("manifest items must be a list")


def _validate_manifest_items(manifest: dict, *, output: Path) -> None:
    required = {
        "page_id", "page_url", "title", "url", "width", "height", "size",
        "mime_type", "author", "license", "license_url", "metadata",
        "retrieved_at", "status", "reason",
    }
    seen_page_ids: set[int] = set()
    for index, item in enumerate(manifest["items"]):
        if not isinstance(item, dict):
            raise ManifestValidationError(f"manifest item {index} must be an object")
        missing = required.difference(item)
        if missing:
            raise ManifestValidationError(
                f"manifest item {index} missing required fields: "
                f"{', '.join(sorted(missing))}")
        page_id = item["page_id"]
        if isinstance(page_id, bool) or not isinstance(page_id, int) or page_id <= 0:
            raise ManifestValidationError(
                f"manifest item {index} page_id must be a positive integer")
        if page_id in seen_page_ids:
            raise ManifestValidationError(
                f"manifest item {index} duplicates page_id {page_id}")
        seen_page_ids.add(page_id)
        if item["page_url"] != f"https://commons.wikimedia.org/?curid={page_id}":
            raise ManifestValidationError(
                f"manifest item {index} page_url is inconsistent with page_id")
        for field in ("title", "url", "mime_type", "retrieved_at", "reason"):
            if not isinstance(item[field], str) or not item[field]:
                raise ManifestValidationError(
                    f"manifest item {index} {field} must be a non-empty string")
        try:
            retrieved = datetime.fromisoformat(
                item["retrieved_at"].replace("Z", "+00:00"))
        except ValueError as error:
            raise ManifestValidationError(
                f"manifest item {index} retrieved_at is not ISO 8601") from error
        if retrieved.tzinfo is None or retrieved.utcoffset() != timezone.utc.utcoffset(None):
            raise ManifestValidationError(
                f"manifest item {index} retrieved_at must be UTC")
        if item["status"] not in {"accepted", "rejected", "failed"}:
            raise ManifestValidationError(
                f"manifest item {index} has invalid status {item['status']!r}")
        if not isinstance(item["metadata"], dict):
            raise ManifestValidationError(
                f"manifest item {index} metadata must be an object")
        if item["status"] == "accepted":
            _validate_accepted_item(item, index=index, output=output)


def _validate_accepted_item(item: dict, *, index: int, output: Path) -> None:
    for field in ("author", "license", "license_url", "sha256", "path"):
        if not isinstance(item.get(field), str) or not item[field]:
            raise ManifestValidationError(
                f"manifest accepted item {index} {field} must be a non-empty string")
    for field in ("width", "height", "bytes"):
        if (isinstance(item.get(field), bool)
                or not isinstance(item.get(field), int) or item[field] <= 0):
            raise ManifestValidationError(
                f"manifest accepted item {index} {field} must be a positive integer")
    sha256 = item["sha256"]
    if len(sha256) != 64 or any(character not in "0123456789abcdef" for character in sha256):
        raise ManifestValidationError(
            f"manifest accepted item {index} sha256 must be lowercase hexadecimal")
    extension = {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
    }.get(item["mime_type"])
    expected_path = f"images/{sha256[:16]}{extension or ''}"
    if extension is None or item["path"] != expected_path:
        raise ManifestValidationError(
            f"manifest accepted item {index} path is inconsistent with sha256 and MIME")
    image_path = output / Path(item["path"])
    if not image_path.is_file():
        raise ManifestValidationError(
            f"manifest accepted item {index} accepted path does not exist: {item['path']}")
    payload = image_path.read_bytes()
    if len(payload) != item["bytes"]:
        raise ManifestValidationError(
            f"manifest accepted item {index} byte count does not match its path")
    if hashlib.sha256(payload).hexdigest() != sha256:
        raise ManifestValidationError(
            f"manifest accepted item {index} sha256 does not match its path")


def main(argv: list[str] | None = None, *, opener: UrlOpener = urlopen,
         clock: Clock = _utc_now) -> int:
    parser = argparse.ArgumentParser(
        description="Acquire and verify licensed Wikimedia Commons images.")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--verify", type=Path, metavar="OUTPUT")
    arguments = parser.parse_args(argv)
    if arguments.verify is not None:
        if arguments.config is not None or arguments.out is not None:
            parser.error("--verify cannot be combined with --config or --out")
        try:
            manifest = verify_commons(arguments.verify)
        except Exception as error:
            print(f"error: {error}", file=sys.stderr)
            return 1
        _print_summary(manifest)
        return 0
    if arguments.config is None or arguments.out is None:
        parser.error("acquisition requires --config and --out")

    try:
        manifest = acquire_commons(
            arguments.config, arguments.out, opener, clock=clock)
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    _print_summary(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

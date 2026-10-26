"""Sample phone capture frames without crossing capture-session boundaries."""
from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path


VALID_SPLITS = {"train", "val", "test"}


@dataclass(frozen=True)
class Sample:
    source_session: str
    sequence: int
    client_ts: float | None
    server_ts: float | None
    split: str
    source_path: str
    output_path: str


@dataclass(frozen=True)
class SampleReport:
    source_session: str
    split: str
    interval: int
    samples: list[Sample]
    missing_frames: list[str]


def _read_manifest(session_dir: Path) -> list[dict[str, object]]:
    manifest = session_dir / "manifest.jsonl"
    rows: list[dict[str, object]] = []
    sequences: set[int] = set()
    for line_number, line in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        try:
            sequence = int(row["seq"])
            relative = Path(str(row["file"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid manifest row {line_number}") from exc
        if sequence in sequences:
            raise ValueError(f"duplicate source sequence: {sequence}")
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"frame path escapes session: {relative}")
        row["seq"] = sequence
        row["file"] = relative.as_posix()
        sequences.add(sequence)
        rows.append(row)
    return sorted(rows, key=lambda row: int(row["seq"]))


def _write_report(output: Path, report: SampleReport) -> None:
    payload = asdict(report)
    report_path = output / "sampling-report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def sample_session(
    session_dir: Path,
    output: Path,
    split: str,
    interval: int,
    frozen: set[str],
) -> SampleReport:
    session_dir = Path(session_dir)
    output = Path(output)
    if split not in VALID_SPLITS:
        raise ValueError(f"unknown split: {split}")
    if interval <= 0:
        raise ValueError("interval must be positive")
    if session_dir.name in frozen and split != "test":
        raise ValueError(f"frozen evaluation session must use test: {session_dir.name}")

    rows = _read_manifest(session_dir)
    first_sequence = int(rows[0]["seq"]) if rows else 0
    missing = [
        str(row["file"])
        for row in rows
        if not (session_dir / str(row["file"])).is_file()
    ]
    destination = output / split / "images"
    destination.mkdir(parents=True, exist_ok=True)
    samples = []
    for row in rows:
        sequence = int(row["seq"])
        if (sequence - first_sequence) % interval:
            continue
        source = session_dir / str(row["file"])
        if not source.is_file():
            continue
        relative_output = Path(split) / "images" / (
            f"{session_dir.name}_{sequence:06d}.jpg"
        )
        shutil.copy2(source, output / relative_output)
        samples.append(
            Sample(
                source_session=session_dir.name,
                sequence=sequence,
                client_ts=row.get("client_ts"),
                server_ts=row.get("server_ts"),
                split=split,
                source_path=str(row["file"]),
                output_path=relative_output.as_posix(),
            )
        )

    report = SampleReport(
        source_session=session_dir.name,
        split=split,
        interval=interval,
        samples=samples,
        missing_frames=missing,
    )
    _write_report(output, report)
    return report

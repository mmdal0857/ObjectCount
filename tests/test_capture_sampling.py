import json
from pathlib import Path

import pytest

from tools.training.sample_capture import sample_session


def _write_session(root: Path, rows: list[dict[str, object]]) -> Path:
    session = root / "phone-session"
    frames = session / "frames"
    frames.mkdir(parents=True)
    for row in rows:
        frame = session / str(row["file"])
        if not row.get("missing"):
            frame.write_bytes(f"frame-{row['seq']}".encode())
        row.pop("missing", None)
    (session / "manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return session


def test_samples_by_source_sequence_and_preserves_session_identity(tmp_path):
    session = _write_session(
        tmp_path,
        [
            {"seq": 5, "client_ts": 5.1, "server_ts": 5.2, "file": "frames/000005.jpg"},
            {"seq": 1, "client_ts": 1.1, "server_ts": 1.2, "file": "frames/000001.jpg"},
            {"seq": 3, "client_ts": 3.1, "server_ts": 3.2, "file": "frames/000003.jpg"},
            {"seq": 2, "client_ts": 2.1, "server_ts": 2.2, "file": "frames/000002.jpg"},
        ],
    )

    report = sample_session(session, tmp_path / "sampled", "test", 2, set())

    assert [sample.sequence for sample in report.samples] == [1, 3, 5]
    assert [sample.output_path for sample in report.samples] == [
        "test/images/phone-session_000001.jpg",
        "test/images/phone-session_000003.jpg",
        "test/images/phone-session_000005.jpg",
    ]
    assert (tmp_path / "sampled/test/images/phone-session_000001.jpg").read_bytes() == b"frame-1"
    saved = json.loads((tmp_path / "sampled/sampling-report.json").read_text())
    assert saved["samples"][1]["client_ts"] == 3.1
    assert saved["samples"][1]["server_ts"] == 3.2


def test_reports_a_selected_manifest_frame_that_is_missing(tmp_path):
    session = _write_session(
        tmp_path,
        [
            {"seq": 1, "client_ts": 1.0, "server_ts": 1.5, "file": "frames/000001.jpg"},
            {
                "seq": 3,
                "client_ts": 3.0,
                "server_ts": 3.5,
                "file": "frames/000003.jpg",
                "missing": True,
            },
        ],
    )

    report = sample_session(session, tmp_path / "sampled", "test", 2, set())

    assert report.missing_frames == ["frames/000003.jpg"]
    assert [sample.sequence for sample in report.samples] == [1]


@pytest.mark.parametrize("split", ["train", "val"])
def test_rejects_frozen_session_outside_test(tmp_path, split):
    session = _write_session(
        tmp_path,
        [{"seq": 1, "client_ts": 1.0, "server_ts": 1.5, "file": "frames/000001.jpg"}],
    )

    with pytest.raises(ValueError, match="frozen evaluation session"):
        sample_session(session, tmp_path / "sampled", split, 1, {session.name})


def test_rejects_unknown_split_and_nonpositive_interval(tmp_path):
    session = _write_session(
        tmp_path,
        [{"seq": 1, "client_ts": 1.0, "server_ts": 1.5, "file": "frames/000001.jpg"}],
    )

    with pytest.raises(ValueError, match="unknown split"):
        sample_session(session, tmp_path / "sampled", "training", 1, set())
    with pytest.raises(ValueError, match="positive"):
        sample_session(session, tmp_path / "sampled", "test", 0, set())


def test_split_config_declares_each_existing_evidence_session_frozen():
    config = json.loads(
        Path("training/products/toilet-paper-roll/v2-splits.json").read_text()
    )
    expected = {
        "20260713-103448-Mozilla-5-0-Linux-Androi",
        "20260713-163302-Mozilla-5-0-Linux-Androi",
    }

    assert set(config["sessions"]) == expected
    assert set(config["frozen_test_sessions"]) == expected
    assert {entry["split"] for entry in config["sessions"].values()} == {"test"}

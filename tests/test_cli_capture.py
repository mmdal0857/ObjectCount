"""캡처 CLI — adb 안내·디스크 경고·인자 검증. 서버 기동은 mock."""
import pytest

pytest.importorskip("fastapi")

from objectcount.cli import capture as cli


def test_adb_missing_gives_manual_command(monkeypatch):
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)
    message = cli.try_adb_reverse(8377)
    assert "adb reverse tcp:8377 tcp:8377" in message


def test_adb_failure_gives_manual_command(monkeypatch):
    monkeypatch.setattr(cli.shutil, "which", lambda name: "adb")

    class Result:
        returncode = 1
        stderr = "no devices found"
        stdout = ""
    monkeypatch.setattr(cli.subprocess, "run", lambda *a, **k: Result())
    message = cli.try_adb_reverse(8377)
    assert "no devices found" in message
    assert "adb reverse tcp:8377" in message


def test_check_disk_warns_below_threshold(tmp_path, monkeypatch):
    class Usage:
        free = 1 * 1024 ** 3
    monkeypatch.setattr(cli.shutil, "disk_usage", lambda p: Usage())
    assert "경고" in cli.check_disk(tmp_path, threshold_gb=2.0)


def test_check_disk_ok(tmp_path, monkeypatch):
    class Usage:
        free = 100 * 1024 ** 3
    monkeypatch.setattr(cli.shutil, "disk_usage", lambda p: Usage())
    assert cli.check_disk(tmp_path, threshold_gb=2.0) is None


def test_package_required_without_no_detect(tmp_path, capsys):
    assert cli.main(["--out", str(tmp_path)]) == 1
    assert "--package" in capsys.readouterr().err


def test_no_detect_runs_server(tmp_path, monkeypatch):
    calls = {}
    import uvicorn
    monkeypatch.setattr(uvicorn, "run",
                        lambda app, **kw: calls.update(kw))
    code = cli.main(["--out", str(tmp_path), "--no-detect", "--no-adb",
                     "--port", "9000"])
    assert code == 0
    assert calls["port"] == 9000


def test_broken_package_degrades_to_capture_only(tmp_path, monkeypatch, capsys):
    import uvicorn
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: None)
    code = cli.main(["--out", str(tmp_path), "--no-adb",
                     "--package", str(tmp_path / "없는패키지")])
    assert code == 0  # 캡처가 우선 — 감지 실패는 경고 강등
    assert "감지 없이 캡처만" in capsys.readouterr().err

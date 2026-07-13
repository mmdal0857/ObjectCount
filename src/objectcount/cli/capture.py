"""캡처 서버 CLI — adb reverse 자동 설정·디스크 체크·서버 기동 (스펙 §10)."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import threading
from pathlib import Path

_REVERSE_INTERVAL_S = 5.0


def try_adb_reverse(port: int) -> str:
    manual = f"수동 설정: adb reverse tcp:{port} tcp:{port}"
    adb = shutil.which("adb")
    if adb is None:
        return f"adb를 찾을 수 없습니다. {manual}"
    result = subprocess.run([adb, "reverse", f"tcp:{port}", f"tcp:{port}"],
                            capture_output=True, text=True)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        return f"adb reverse 실패: {detail}. {manual}"
    return f"adb reverse tcp:{port} 설정 완료"


def adb_reverse_keeper(port: int, stop: threading.Event,
                       interval_s: float = _REVERSE_INTERVAL_S) -> None:
    """USB 재연결 시 adb reverse 규칙이 사라지므로 주기적으로 재설정한다 (실기기 검증 발견)."""
    while not stop.wait(interval_s):
        adb = shutil.which("adb")
        if adb is None:
            continue
        subprocess.run([adb, "reverse", f"tcp:{port}", f"tcp:{port}"],
                       capture_output=True, text=True)


def _start_reverse_keeper(port: int) -> threading.Event:
    stop = threading.Event()
    threading.Thread(target=adb_reverse_keeper, args=(port, stop),
                     daemon=True).start()
    return stop


def check_disk(out_dir: Path, threshold_gb: float = 2.0) -> str | None:
    free_gb = shutil.disk_usage(out_dir).free / 1024 ** 3
    if free_gb < threshold_gb:
        return (f"경고: 디스크 여유 {free_gb:.1f}GB — "
                f"{threshold_gb}GB 미만입니다")
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="objectcount-capture")
    parser.add_argument("--package",
                        help="모델 패키지 디렉토리 (--no-detect면 생략)")
    parser.add_argument("--out", required=True, help="캡처 세션 저장 디렉토리")
    parser.add_argument("--port", type=int, default=8377)
    parser.add_argument("--no-adb", action="store_true",
                        help="adb reverse 자동 설정 생략")
    parser.add_argument("--no-detect", action="store_true",
                        help="감지 없이 캡처·스풀만")
    parser.add_argument("--fps", type=int, default=5,
                        help="폰 페이지 기본 캡처 fps")
    args = parser.parse_args(argv)

    try:
        import uvicorn

        from ..capture.server import create_app
    except ImportError:
        print('오류: capture extra가 필요합니다 — pip install -e ".[capture]"',
              file=sys.stderr)
        return 1

    detector = None
    warning = None
    if args.no_detect:
        warning = "감지 없이 캡처만 (--no-detect)"
    elif not args.package:
        print("오류: --package가 필요합니다 (감지를 끄려면 --no-detect)",
              file=sys.stderr)
        return 1
    else:
        try:
            from ..detection.detector import OnnxDetector
            from ..models.package import load_package
            detector = OnnxDetector(load_package(args.package))
        except Exception as error:  # 캡처가 우선 — 감지 실패는 경고 강등 (스펙 §9)
            warning = f"모델 패키지 로드 실패: {error} — 감지 없이 캡처만 진행"
            print(warning, file=sys.stderr)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    disk_warning = check_disk(out_dir)
    if disk_warning:
        print(disk_warning, file=sys.stderr)

    stop_keeper = None
    if not args.no_adb:
        print(try_adb_reverse(args.port))
        stop_keeper = _start_reverse_keeper(args.port)

    print(f"폰에서 열기: http://localhost:{args.port}/phone")
    print(f"PC 뷰어:     http://localhost:{args.port}/")
    print(f"캡처 세션 저장: {out_dir}")

    app = create_app(out_dir=out_dir, detector=detector,
                     default_fps=args.fps, startup_warning=warning)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
    if stop_keeper is not None:
        stop_keeper.set()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

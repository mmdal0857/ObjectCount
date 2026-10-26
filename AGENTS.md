# ObjectCount

다중 핸드폰 카메라로 컨베이어 위 오브젝트를 카운팅하는 시스템. **Phase 1a(감지 코어) 완료.**
**다음 작업 = Phase 1b-1 추적·카운팅 코어** (트랙 ID 넘버링 + 카운트라인 카운팅, 영상 파일로 검증) → 1b-2 캡처 앱·실시간 → Phase 3 다중 카메라 융합.

## 핵심 문서

- 스펙: `docs/superpowers/specs/2026-07-12-phone-camera-object-counting-design.md`
- 용어집: `CONTEXT.md` — 라인/프로덕트/트랙/카운트 세션/의심 이벤트/보정 등. 코드·대화에서 이 용어를 쓴다
- ADR: `docs/adr/` — 0001 무손실 store-and-forward, 0002 유선 우선 연결

## 명령어

- 초기 세팅: `setup.bat` — `.venv` 생성, `pip install -e ".[dev,capture]"`, 테스트 실행. YOLO export 도구까지 설치하려면 `setup.bat --tools`
- 테스트: `.venv\Scripts\python -m pytest -q` (97개)
- 감지: `.venv\Scripts\objectcount-detect --package models/packages/coco-demo --input <이미지|폴더|영상> --out out/<이름>`
- 캡처: `.venv\Scripts\objectcount-capture --package models/packages/coco-demo --out out/captures` — 폰 Chrome에서 `http://localhost:8377/phone`, PC 뷰어 `http://localhost:8377/`. 캡처 세션 폴더는 `--input`으로 그대로 리플레이 가능
- 평가: `.venv\Scripts\objectcount-eval --package <pkg> --dataset <dir>` — **항상 exit 0** (하네스는 보고 도구; 실행 실패는 stderr `실행불가:` 접두어로 보고 결과와 구분)
- COCO 데모 패키지 재생성: `.venv\Scripts\python tools\export_pretrained.py` (선행: `pip install -e ".[tools]"`)

## 규칙

- **Detection 좌표는 [0,1] 정규화 xyxy** — 해상도 독립. 픽셀 변환은 `Detection.to_pixels()`로만
- **모델 가중치는 ONNX 고정**, 추론은 onnxruntime EP 자동 선택(CUDA→DirectML→CPU) — GPU 벤더 종속 코드 금지
- **카운팅 3원칙**(스펙 §1): 보수적(애매하면 적게 세고 의심 플래그) / 무손실(프레임 임의 폐기 금지) / 원본 불변(보정은 append-only)
- 프레임 입력은 `iter_frames()`의 `Iterator[FrameRecord]` 시그니처로만 — 폰 스트림도 같은 자리에 꽂힌다
- 개발 PC는 AMD GPU — 학습은 백엔드 인터페이스로 추상화(스펙 §4-③), GPU 추론은 `onnxruntime-directml` 교체로
- **OpenCV 파일 IO는 Windows 한글 경로에서 무음 실패** — 쓰기는 `imencode`+`write_bytes`(detect.py), 읽기는 `frames/image_io.py`의 `read_image()`(np.fromfile+imdecode, sources·spool 적용됨). 새 코드에서 `cv2.imread`/`cv2.imwrite` 직접 호출 금지

## 작업 구조

- 메인 레포 `F:\Project\ObjectCount`(master) / `C:\Users\mmdal\orca\workspaces\ObjectCount\*`는 브랜치별 worktree — **워크트리마다 `.venv` 별도 생성 필요** (`py -m venv .venv` 후 `pip install -e ".[dev,capture]"`)
- Codex 코드 리뷰는 `coelacanth` 워크트리의 `docs/reviews/*-for-AGENTS.md`로 전달됨

## 에이전트 작업 규칙

- 코드를 변경하기 전에 `CLAUDE.md`를 읽는다. 프로젝트 명령, 구현 규칙, 현재 단계의 후속 작업에 대한 기준 문서다.
- 도메인을 추론하기 전에 `CONTEXT.md`를 읽고 코드, 테스트, 문서, 상태 보고에 정의된 ObjectCount 용어를 사용한다.
- `CLAUDE.md`와 `CONTEXT.md`가 충돌하면 임의로 선택하지 말고 작업을 중단한 뒤 충돌을 보고한다.
- 사용자 변경사항을 보존하고 요청받은 범위 안에서만 작업한다.
- 명시적으로 허가받지 않은 커밋, 푸시, 파괴적인 Git 작업은 수행하지 않는다.
- `CLAUDE.md`에서 가장 좁고 적절한 검증 명령을 실행하고 명령과 결과를 보고한다.
- 완료 보고는 요약, 변경 파일, 검증, 위험 또는 미해결 항목 순서로 작성한다.

## Phase 1b 이월 항목 (최종 리뷰 follow-up)

README, 테스트 MANIFEST 중복 정리
(완료: FrameRecord timestamp / zero-area 박스 필터 / cv2.imwrite 한글 경로 / ROI 범위 검증 / eval 실행불가 구분 — 2026-07-13 캡처 도구 세션에서 처리)

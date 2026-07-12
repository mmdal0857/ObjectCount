# ObjectCount

다중 핸드폰 카메라로 컨베이어 위 오브젝트를 카운팅하는 시스템. **Phase 1a(감지 코어) 완료.**
**다음 작업 = Phase 1b-1 추적·카운팅 코어** (트랙 ID 넘버링 + 가상선 카운팅, 영상 파일로 검증) → 1b-2 캡처 앱·실시간 → Phase 3 다중 카메라 융합.

## 핵심 문서
- 스펙: `docs/superpowers/specs/2026-07-12-phone-camera-object-counting-design.md`
- 용어집: `CONTEXT.md` — 라인/품종/카운트 세션/의심 이벤트/보정 등. 코드·대화에서 이 용어를 쓴다
- ADR: `docs/adr/` — 0001 무손실 store-and-forward, 0002 유선 우선 연결

## 명령어
- 테스트: `.venv\Scripts\python -m pytest -q` (49개)
- 감지: `.venv\Scripts\objectcount-detect --package models/packages/coco-demo --input <이미지|폴더|영상> --out out/<이름>`
- 평가: `.venv\Scripts\objectcount-eval --package <pkg> --dataset <dir>` — **항상 exit 0** (하네스는 보고 도구, 오류도 stderr 보고 후 0)
- COCO 데모 패키지 재생성: `.venv\Scripts\python tools\export_pretrained.py` (선행: `pip install -e ".[tools]"`)

## 규칙
- **Detection 좌표는 [0,1] 정규화 xyxy** — 해상도 독립. 픽셀 변환은 `Detection.to_pixels()`로만
- **모델 가중치는 ONNX 고정**, 추론은 onnxruntime EP 자동 선택(CUDA→DirectML→CPU) — GPU 벤더 종속 코드 금지
- **카운팅 3원칙**(스펙 §1): 보수적(애매하면 적게 세고 의심 플래그) / 무손실(프레임 임의 폐기 금지) / 원본 불변(보정은 append-only)
- 프레임 입력은 `iter_frames()`의 `Iterator[FrameRecord]` 시그니처로만 — 폰 스트림도 같은 자리에 꽂힌다
- 개발 PC는 AMD GPU — 학습은 백엔드 인터페이스로 추상화(스펙 §4-③), GPU 추론은 `onnxruntime-directml` 교체로

## Phase 1b 이월 항목 (최종 리뷰 follow-up)
FrameRecord `timestamp` 필드(추적용), zero-area 박스 필터, `cv2.imwrite` 반환 검사(한글 경로 무음 실패), ROI [0,1] 범위 검증, README, eval "보고됨 vs 실행불가" 구분, 테스트 MANIFEST 중복 정리

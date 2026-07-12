# 웹 캡처 도구 (유선 폰 카메라 → PC 라이브 확인·스풀 녹화) — 설계

- 날짜: 2026-07-13
- 상태: 승인
- 위치: Phase 1b-1 선행 도구. 정식 안드로이드 캡처 앱(Phase 1b-2)의 개발용 선행 슬라이스
- 관련: 본설계 `2026-07-12-phone-camera-object-counting-design.md` §2-①·§3-8·§8,
  ADR-0001(무손실 store-and-forward), ADR-0002(유선 우선)

## 1. 목표

adb로 연결된 안드로이드 폰의 카메라를 **웹 페이지(브라우저)로 제어**(줌·포커스·노출)하고,
프레임을 USB 유선으로 PC에 전송해 **감지 오버레이를 라이브로 확인**하며, 수신 프레임 전부를
**스풀에 녹화해 리플레이 하네스 자산으로 축적**하는 개발용 도구.

- 폰에 설치하는 것 없음 — 브라우저(Chrome)만 사용
- 유선: `adb reverse`로 폰 브라우저 → PC 서버를 USB 터널링 (ADR-0002).
  `localhost`는 보안 컨텍스트이므로 HTTPS 인증서 없이 카메라 권한이 열린다
- 녹화된 캡처 세션 폴더는 `iter_frames()`로 그대로 리플레이 가능 — 추적·카운팅 코어(1b-1)의
  개발·검증 영상이 이 도구로 공급된다

### 비목표

- 정식 캡처 앱 대체 아님 — 폰 쪽 로컬 스풀(store-and-forward), 포그라운드 서비스, 발열·배터리
  관리, mDNS 프로비저닝은 1b-2 몫. 이 도구는 USB 직결 전제의 개발 도구
- 추적·카운팅 아님 — 라이브 뷰는 감지 박스까지. 트랙 ID·카운트 오버레이는 1b-1 코어 완성 후
  같은 자리에 얹는다
- 다중 폰 동시 접속 아님 — 폰 1대 전제 (MVP)

## 2. 용어

**캡처 세션 (Capture Session)**: 이 도구가 폰 접속 1회 동안 수신한 프레임과 메타데이터를 담은
스풀 폴더 1개. 도메인의 **카운트 세션과 무관** — 개발용 녹화 단위일 뿐이다.

## 3. 아키텍처

```
폰 Chrome (http://localhost:8377/phone)      PC
  getUserMedia + 줌/포커스/노출 제어      objectcount-capture CLI (FastAPI)
  5fps JPEG + seq + client_ts  ──WS /ws/ingest──▶ ① 스풀 기록 (수신 즉시, 전 프레임)
       ▲ USB (adb reverse tcp:8377)               ② 라이브 감지 워커 (최신 프레임만)
       └── ack(seq) ◀──────────────────           ③ 뷰어 브로드캐스트 ──WS /ws/view──▶
                                                     PC 브라우저 (오버레이+진단 바)
```

모듈 `src/objectcount/capture/`:

| 파일 | 역할 |
|---|---|
| `server.py` | FastAPI 앱 — `/phone` 캡처 페이지, `/` 뷰어 페이지, `/ws/ingest`, `/ws/view` |
| `spool.py` | `SpoolWriter`(JPEG+manifest 기록), `SpoolSource`(캡처 세션 → `FrameRecord` 이터레이터) |
| `live.py` | 감지 워커 — 크기 1 mailbox(최신 교체), `OnnxDetector` → `viz` 오버레이 → 뷰어 푸시 |
| `static/phone.html` | 캡처 페이지 (바닐라 JS, 빌드 스텝 없음) |
| `static/viewer.html` | 뷰어 페이지 (바닐라 JS) |
| `cli/capture.py` | `objectcount-capture` 진입점, adb reverse 자동 실행 |

기존 `detection`·`frames`·`viz`는 소비만 한다. 수정은 `FrameRecord` timestamp 추가와
`iter_frames()` 스풀 디스패치 두 곳뿐.

**우선순위 원칙**: 캡처·스풀이 최우선. 감지 워커가 죽어도(패키지 로드 실패 포함) 수신·기록은
계속되고 뷰어에는 원본 프레임 + 경고 배너를 보낸다. 라이브 뷰는 최신 프레임만 감지해 표시가
드롭될 수 있으나 **저장은 전 프레임** — 무손실 원칙의 도구판.

## 4. 스풀 포맷 (캡처 세션 폴더)

```
out/captures/20260713-153012-<기기명>/
  session.json     # 기기 정보, capabilities, 적용된 카메라 설정, 패키지명, 시작/종료 시각
  manifest.jsonl   # 프레임당 1줄: {"seq":1,"client_ts":…,"server_ts":…,"file":"frames/000001.jpg","w":1280,"h":720}
  frames/000001.jpg …
```

- 파일명은 seq 기반 6자리 0채움. manifest는 수신 순서대로 append-only
- `server_ts`가 단일 시간 기준(본설계 §3-8) — epoch 초. `client_ts`는 참고용
- 시퀀스 공백은 그대로 남긴다(결손의 증거) — `SpoolSource`는 공백을 건너뛰며 순서 보장
- 도중 중단된 세션도 유효 — `session.json`의 종료 시각만 비어 있다

## 5. FrameRecord timestamp (이월 항목 선반영)

`FrameRecord`에 `timestamp: float | None = None` 추가 — 초 단위, 소스 내 단조 증가.

| 소스 | 값 |
|---|---|
| 캡처 세션(스풀) | `server_ts` (epoch 초) |
| 영상 파일 | `frame_index / fps` (파일 상대). fps 조회 불가(0 반환) 시 `None` |
| 이미지·폴더 | `None` |

`iter_frames()`는 `manifest.jsonl`이 있는 폴더를 `SpoolSource`로 자동 디스패치한다.
기존 소비자(detect CLI 등)는 변경 없이 캡처 세션을 입력으로 받는다.
추적기(1b-1)는 이 timestamp를 프레임 간격 계산에 쓴다.

## 6. 폰 캡처 페이지

- `getUserMedia` 후면 카메라, 기본 1280×720, 캡처 fps 기본 5 (설정 가능)
- `track.getCapabilities()`로 기기 지원 항목 조회 후 컨트롤 동적 노출:
  - **줌** 슬라이더 (min/max/step은 capabilities 값)
  - **포커스** continuous ↔ manual 토글 + 거리 슬라이더 (지원 시)
  - **노출 잠금** (지원 시), 해상도 선택
  - 미지원 항목은 "이 기기 미지원"으로 비활성 — 기기별 편차를 화면에서 바로 확인
- 적용된 설정 스냅샷을 hello 시·변경 시 서버로 보내 `session.json`에 기록
- **Screen Wake Lock**으로 화면 꺼짐 방지. 페이지가 백그라운드로 가면 캡처 일시정지 + 경고 배너
- 상태 표시: 전송 seq / ack seq / 미확인 수

## 7. 전송 프로토콜 (WS `/ws/ingest`)

1. **hello** (JSON): 기기 정보, capabilities, 카메라 설정 → 서버가 캡처 세션 폴더 생성
2. **frame**: 메시지당 JSON 헤더(`seq`, `client_ts`, `w`, `h`) + JPEG 바이너리.
   `video → canvas → toBlob('image/jpeg', 0.85)` 고정 간격 캡처
3. **ack** (서버→폰, JSON): 스풀 기록 완료한 seq. 기록 후에만 ack — 정식 앱의 내구 기록 ack
   프로토콜(본설계 §2-①)을 미리 검증하는 셈
4. 연결 끊김 → 지수 백오프 재접속, 같은 캡처 세션에 이어쓰기. 폰 쪽 프레임 버퍼링은 하지
   않는다(비목표) — 공백은 seq로 탐지된다

## 8. 뷰어 페이지·진단

- 감지 오버레이 스트림(서버가 `viz`로 주석 JPEG 생성 후 푸시)
- 진단 바: 수신 fps · 감지 fps · 시퀀스 공백 수 · 세션 누적 프레임 · 스풀 경로 · 경고 배너
- `--no-detect` 시 원본 스트림만

## 9. 에러 처리

| 상황 | 동작 |
|---|---|
| adb 없음/기기 미연결 | 수동 `adb reverse tcp:8377 tcp:8377` 명령 안내 후 서버는 정상 기동 |
| 포트 점유 | 명확한 에러 메시지, `--port`로 변경 안내 |
| 모델 패키지 로드 실패 | 캡처·스풀 계속, 뷰어에 경고 배너 (`--no-detect`와 동일 동작) |
| 디스크 여유 부족 | 시작 시 임계(기본 2GB) 미만이면 경고 출력, 기동은 허용 |
| 시퀀스 공백 | 뷰어 진단 바 카운트 + manifest에 공백 그대로 보존 |
| 페이지 백그라운드 전환 | 폰 쪽 캡처 일시정지 + 배너 (프레임 유실을 침묵시키지 않음) |

## 10. CLI

```
objectcount-capture --package models/packages/coco-demo --out out/captures
                    [--port 8377] [--no-adb] [--no-detect] [--fps 5]
```

기동 로그에 폰에서 열 주소(`http://localhost:8377/phone`)와 PC 뷰어 주소를 출력한다.
`--fps`는 폰 캡처 페이지에 내려보내는 기본 캡처 fps다(페이지에서 변경 가능).
뷰어 페이지는 다중 접속을 허용한다(브로드캐스트).

## 11. 테스트 전략

- **유닛**: `SpoolWriter`↔`SpoolSource` 왕복(타임스탬프·seq 순서·공백 건너뛰기),
  `iter_frames()` 스풀 디스패치, manifest 파싱, timestamp 의미론(영상 fps 상대값 포함)
- **통합**: 가짜 폰 클라이언트(테스트 코드가 WS로 hello+JPEG 전송) → 스풀 기록·ack 순서·뷰어
  브로드캐스트 검증. FastAPI TestClient로 실제 폰·브라우저 없이 CI 실행. 감지 워커는 가짜
  검출기 주입
- **실기기 수동 체크리스트** (자동화 불가 항목):
  1. `adb reverse` 자동 실행 확인 → 폰 Chrome에서 `/phone` 접속
  2. 줌·포커스 조작이 폰 프리뷰와 PC 뷰어 양쪽에 반영되는지
  3. 물체를 카메라 앞에 통과시켜 감지 박스 라이브 확인
  4. 세션 종료 후 `objectcount-detect --input out/captures/<세션>` 리플레이 정상
  5. USB 분리→재연결 시 재접속·이어쓰기 확인

## 12. 의존성

`pyproject.toml`에 optional extra `[capture]`: `fastapi`, `uvicorn`, (WS 지원용) `websockets`.
코어 의존성은 불변. 페이지는 바닐라 JS — node 툴체인 없음.

## 13. 정식 캡처 앱(1b-2)과의 관계

| 항목 | 웹 캡처 도구 (본 설계) | 정식 캡처 앱 (1b-2) |
|---|---|---|
| 플랫폼 | 폰 브라우저 (설치 없음) | Kotlin/CameraX 네이티브 |
| 카메라 제어 | 기기 지원 범위 내 (편차 있음) | 완전 제어 |
| 폰 로컬 스풀 | 없음 (USB 직결 전제) | store-and-forward (ADR-0001) |
| 연결 | adb reverse 유선 | 유선 우선 (ADR-0002), mDNS 프로비저닝 |
| 재사용 | 프로토콜(hello·seq·ack)·스풀 포맷·뷰어를 그대로 승계 | — |

이 도구는 이후 등록 파이프라인의 **수집 모드** 대용(학습 샘플 확보)으로도 쓸 수 있다.

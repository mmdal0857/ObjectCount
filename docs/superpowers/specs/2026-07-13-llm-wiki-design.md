# LLM 위키 설계 — 프로젝트 지식 베이스와 멀티 LLM 참조 구조

날짜: 2026-07-13
상태: 승인됨 (브레인스토밍 세션에서 사용자 확정)

## 목적

작업 완료 후 housekeeping 시점에 프로젝트 지식을 위키로 정리하고, 협업하는 모든 LLM(Claude·Codex·OpenCode Gemma)이 같은 위키를 참조하게 한다.

## 결정 요약

| 질문 | 결정 |
|------|------|
| 적용 범위 | ObjectCount 구현 + 전역 컨벤션 동시 진행 |
| 참조 방식 | 진입점 포인터 + 디스패치 프롬프트에 위키 경로 명시 규칙 |
| 구조 | 경량 플랫 위키 (플랫 파일 + 수동 인덱스) |

## 문서 경계

위키는 중복 저장소가 아니라 **"지금 시스템이 어떻게 생겼나"의 살아있는 스냅샷**이다. 스펙·ADR은 시점 고정 문서, 위키는 코드를 따라간다.

| 문서 | 역할 | 위키와의 관계 |
|------|------|--------------|
| `CLAUDE.md` / `AGENTS.md` | 명령어·규칙·현재 페이즈 | 위키 인덱스를 가리킨다 |
| `CONTEXT.md` | 도메인 용어집 | 위키가 용어를 사용한다 (재정의 금지) |
| `docs/adr/` | 결정과 근거 | 위키가 링크한다 (복붙 금지) |
| `docs/superpowers/specs·plans/` | 설계·계획 (시점 고정) | 위키가 요약 후 링크한다 |
| `docs/wiki/` | 현재 상태 스냅샷 | housekeeping 때 갱신 |

## ObjectCount 구조

```
docs/wiki/
  WIKI_INDEX.md      # 에이전트 진입점 — 파일당 한 줄: 제목·요약·갱신일
  detection.md       # 감지 코어: Detection 모델, ONNX 패키지, EP 선택
  frames.md          # 프레임 파이프라인: iter_frames, sources, spool
  capture.md         # 캡처 도구: 폰 스트리밍, 세션 폴더, 리플레이
  training.md        # 학습 파이프라인: 품종 등록, WSL ROCm, 후보 게이트
  orchestration.md   # 멀티 LLM 작업 구조: 역할 분담, 워크플로, 채널
```

### 페이지 포맷

- frontmatter 3필드: `title`, `summary`, `last_updated`
- 본문 150줄 상한. 초과하면 분할보다 압축이 기본 — 오래된 세부는 ADR·스펙 링크로 대체
- 근거: Gemma 컨텍스트 32k에서 인덱스 + 페이지 2-3개를 소화하는 크기

## 참조 연결

1. **AGENTS.md** (포인터 버전): "서브시스템 이해가 필요하면 `docs/wiki/WIKI_INDEX.md`에서 해당 페이지만 골라 읽어라" 한 줄 추가
2. **CLAUDE.md** 핵심 문서 섹션에 위키 인덱스 추가
3. **디스패치 규칙** (전역): Orca 태스크·Codex 위임·Gemma 리뷰 프롬프트 작성자는 관련 위키 페이지 경로를 프롬프트에 명시한다. 약한 모델은 인덱스 탐색 없이 지정 페이지만 읽는다

## 갱신 플로우

housekeeping Step 2.5(기존 정의)가 그대로 작동한다: 세션 커밋 확인 → 영향 도메인 페이지만 갱신 → `last_updated` 갱신. 단 이 프로젝트는 인덱스 생성 스크립트를 쓰지 않으므로 "generate-wiki-index.js 실행" 항목을 "WIKI_INDEX.md 수동 갱신"으로 대체한다.

## 전역 컨벤션

`~/.claude/rules/llm-wiki-convention.md` 신규 작성:

- 위키 구조 표준: 플랫 파일 + WIKI_INDEX.md + frontmatter 3필드 + 150줄 상한
- 문서 경계 규칙 (위 표)
- 디스패치 프롬프트에 위키 경로 포함 규칙
- 신규 프로젝트 시드 절차: 기존 문서(스펙·CLAUDE.md·코드)에서 압축 작성 → 로컬 모델 리뷰 → 커밋

`~/.claude/CLAUDE.md`에는 rules 파일을 가리키는 한 줄만 추가한다.

## 초기 시드

Claude가 기존 문서에서 5개 페이지 초안을 압축 작성하고, Gemma 리뷰(무료)로 사실 검증 후 커밋한다. `orchestration.md`는 2026-07-13 파악한 멀티 LLM 구조(역할 분담, codex-collab 워크플로, Orca 태스크 시스템, OpenCode 워커)를 담는다.

## 비범위

- 위키 자동 주입 장치(OpenCode 설정·훅) — 토큰 고정비용 때문에 제외. 필요해지면 별도 설계
- 계층 디렉토리·자동 인덱스 — 페이지 수십 개 초과 시 재검토
- Drive 동기화 — 이 위키는 git이 단일 원천

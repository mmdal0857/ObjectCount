# Codex 및 OpenCode 기본 설정 설계

## 목표

Orca에서 새 작업을 시작할 때 Claude 대신 Codex를 기본 에이전트로 사용한다. 자동 브랜치·워크스페이스 이름 생성도 Codex로 통일한다. OpenCode는 로컬 LLM을 사용하는 보조 작업자로 구성하고, Codex와 저장소 지식을 공유하면서 검증 가능한 결과를 반환하게 한다.

## 현재 상태

- Orca 전역 `defaultTuiAgent`는 `claude`다.
- Source Control AI의 `branchName` 작업에는 에이전트가 명시되지 않아 전역 기본값인 Claude를 사용한다.
- OpenCode 1.17.18은 `lmstudio/google/gemma-4-12b`를 `http://127.0.0.1:1234/v1`에서 사용한다.
- OpenCode에는 별도의 기본 작업자 역할이나 전역 지침이 없다.
- 저장소에는 `CLAUDE.md`와 `CONTEXT.md`가 있지만 `AGENTS.md`는 없다.

## 설계

### Orca

Orca의 전역 기본 TUI 에이전트를 `codex`로 변경한다. Source Control AI의 `branchName` 레시피에도 `codex`를 명시한다. 레시피별 값을 명시함으로써 향후 전역 기본 에이전트가 바뀌더라도 자동 이름 생성은 계속 Codex를 사용한다.

실행 중인 Orca의 상태 파일을 직접 수정하면 앱의 후속 저장에 의해 변경이 덮어써질 수 있다. 따라서 Orca가 제공하는 설정 UI를 통해 값을 변경하고, 저장된 설정을 읽어 결과를 검증한다.

### OpenCode

현재 LM Studio 모델은 유지한다. `~/.config/opencode/agents/codex-worker.md`에 `codex-worker` 기본 에이전트를 만들고 `opencode.json`의 `default_agent`로 지정한다.

작업자 지침은 다음 계약을 따른다.

- Codex나 호출자가 전달한 작업 범위 안에서만 작업한다.
- 저장소의 `AGENTS.md`, `CONTEXT.md`, `CLAUDE.md` 등 존재하는 지침과 문맥 파일을 먼저 확인한다.
- 추측을 사실처럼 보고하지 않고, 불확실성은 명시한다.
- 요청받지 않은 커밋과 외부 상태 변경을 하지 않는다.
- `git push`, `git reset --hard`, `git clean`, 강제 브랜치 삭제 등 파괴적 작업을 수행하지 않는다.
- 완료 시 요약, 변경 파일, 실행 명령, 테스트 결과, 위험과 미해결 항목을 반환한다.
- 사용자와 직접 의사결정하지 않고, 필요한 결정은 blocker로 호출자에게 반환한다.

`~/.config/opencode/AGENTS.md`에는 모든 OpenCode 세션에 적용할 최소한의 공용 행동 규칙만 둔다. 긴 프로젝트 지식이나 세션별 대화 기록은 전역 메모리에 복제하지 않는다.

### 저장소 공용 지식

저장소 루트에 간결한 `AGENTS.md`를 추가한다. 이 파일은 Codex와 OpenCode가 공통으로 읽는 진입점이며, 기존 `CLAUDE.md`와 `CONTEXT.md`를 프로젝트의 상세 규칙과 현재 문맥 원본으로 지정한다. 같은 내용을 여러 파일에 복사하지 않는다.

공용 지식과 개인 실행 설정의 경계는 다음과 같다.

- 저장소: 프로젝트 구조, 명령, 규칙, 결정, 현재 단계
- OpenCode 전역 설정: 작업자 역할, 결과 형식, 안전 경계
- Orca 설정: 어떤 에이전트를 기본 실행 및 자동 이름 생성에 사용할지

## 권한과 오류 처리

OpenCode는 저장소 내부 읽기·검색·편집·테스트를 허용한다. 외부 디렉터리 접근은 승인을 요구하고, push와 파괴적 Git 명령은 거부한다. 커밋은 호출자가 명시적으로 요청했는지 판단할 수 있도록 기본적으로 승인을 요구한다.

LM Studio가 실행 중이 아니거나 모델이 로드되지 않은 경우 OpenCode는 실패 원인을 반환하고 다른 클라우드 모델로 자동 전환하지 않는다. Codex 또는 사용자가 재시도와 대체 모델을 결정한다.

## 검증

- Orca 저장 설정에서 `defaultTuiAgent`가 `codex`인지 확인한다.
- Source Control AI의 `branchName.agentId`가 `codex`인지 확인한다.
- `opencode debug config`로 병합된 설정이 유효한지 확인한다.
- `opencode agent list`에서 `codex-worker`가 기본 실행 가능한 primary agent인지 확인한다.
- 짧은 읽기 전용 작업을 OpenCode에 전달해 정해진 결과 형식과 저장소 문맥 로딩을 확인한다.
- 프로젝트 `AGENTS.md`가 기존 문서와 모순되거나 상세 내용을 중복하지 않는지 확인한다.

## 범위 밖 후속 작업

LLM 위키 설정은 별도의 설계와 구현 작업으로 진행한다. 위키가 참조할 지식 원본, 생성·갱신 방식, 저장 위치, 수동 또는 자동 파이프라인 여부를 먼저 정한다. 이번 설정에서는 위키 파일이나 파이프라인을 만들지 않는다.

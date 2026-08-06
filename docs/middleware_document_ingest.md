# 미들웨어 연동 가이드 — RAG 문서 적재/관리

MARS(A.X RAG 서버)의 문서 적재·관리 기능을 미들웨어가 연동하기 위한 문서다.
관리자 페이지(문서 업로드/목록/삭제)의 백엔드 흐름을 정의한다.

> 대상 독자: 미들웨어 서버 개발자
> MARS 포트: `:9000` (내부망, 서버 간 호출)
> 상세 스펙 원본: `docs/interfaces.md` §"문서 관리 API"

---

## 1. 핵심 원칙 — 역할 분담

```
[관리자 웹]  ──►  [미들웨어]                       ──►  [MARS :9000]
  파일 업로드      · 관리자 인증/권한 확인 (MARS엔 없음)     · 청킹·임베딩·색인
  목록/삭제 UI     · 원본을 자기 DB에 영속 저장 (소유자)     · UPLOAD_DIR = 임시
                   · MARS API 프록시 (아래 4종)            · 원본 영속 보관 안 함
                   · 재구축 시 원본 재전송으로 재적재
```

- **원본의 영속 보관 주체는 미들웨어다.** 관리자가 올린 문서 원본을
  미들웨어가 자기 DB(BLOB 등)에 사용자·업로드 이력과 매핑해 저장한다.
- **MARS는 인덱싱 엔진이다.** 받은 원본을 청킹·임베딩·색인(Milvus·BM25)하고,
  원본은 로컬 `UPLOAD_DIR`에 임시 스테이징만 한다 (영속 보관 아님).
- **인증은 미들웨어 책임이다.** MARS는 자체 인증/권한이 없다(내부망 신뢰).
  아래 API는 미들웨어가 관리자 권한을 확인한 뒤에만 호출/프록시한다.
- **전송 방향은 항상 미들웨어 → MARS 인바운드다.** MARS는 미들웨어로
  아웃바운드하지 않는다 (에어갭 규칙). 생성 파일(HWPX)의 fetch-and-store와
  동일한 방향성이다.

---

## 2. 왜 이렇게 바꾸나 (MARS 쪽 변경 배경)

기존에는 MARS가 `UPLOAD_DIR`에 원본을 로컬 보관했는데, 세 가지 문제가 있었다:

1. 삭제해도 원본이 안 지워져 무한 누적(GC 없음)
2. 서버 재구축(Milvus 초기화) 시 로컬 원본이 사라지면 재적재 불가
3. `GET /documents`가 권한 필터 없이 문서명을 노출

→ 원본 소유권을 미들웨어로 옮겨 이 문제들을 해소한다. **MARS API 자체는
그대로이고**(아래 4종 시그니처 변경 없음), 미들웨어가 "원본 소유자"
역할을 새로 맡는 것이 이번 변경의 골자다.

---

## 3. 미들웨어가 연동할 MARS API 4종

모두 이미 구현되어 있다. 시그니처 변경 없음.

### 3-1. `POST /documents` — 문서 적재/갱신

```
POST http://mars:9000/documents?name=<파일명>&domain=<도메인>&department=<부서>&visibility=<공개범위>
Content-Type: application/octet-stream
Body: <파일 바이트 그대로>
```

- **본문**: 파일 바이트 그대로 (multipart 아님). 미들웨어는 관리자가 올린
  파일을 자기 DB에 저장한 뒤, 그 바이트를 그대로 relay한다.
- **쿼리 파라미터**:
  | 파라미터 | 필수 | 값 |
  |---|---|---|
  | `name` | O | 파일명 (`source_doc`). 확장자로 형식 판별. 경로 성분 제거됨 |
  | `domain` | O | `HR` `TECH` `FINANCE_LEGAL` `GENERAL` `MANUAL` `DIRECTIVE` 중 하나 (엄격 검증) |
  | `department` | 조건부 | `visibility=DEPT_ONLY`면 필수 (소유 부서 코드) |
  | `visibility` | X | `ALL`(기본) \| `DEPT_ONLY` |
- **지원 형식**: `.md` `.txt` `.pdf` (텍스트 인코딩 UTF-8·UTF-8 BOM·CP949 자동
  인식, 스캔본 PDF는 실패). 최대 50MB
- **갱신**: 같은 `name`이 이미 있으면 기존 청크 삭제 후 재적재
- **응답 202** + 작업(job) 객체. 적재는 오래 걸려(CPU 문서당 수 분) 백그라운드
  실행. 아래 `job_id`로 진행 상태 폴링:

```json
{
  "job_id": "b1f...", "status": "queued", "source_doc": "훈령.pdf",
  "domain": "DIRECTIVE", "owning_department": "HQ", "visibility": "ALL",
  "submitted_at": "2026-07-07T10:00:00", "started_at": null, "finished_at": null,
  "chunks_indexed": null, "deleted_chunks": null, "error": null
}
```

- **에러**: 400(파라미터/형식/빈 본문), 413(50MB 초과)
- 적재/삭제는 MARS 내부에서 한 번에 하나만 직렬 실행된다 (BM25 재빌드 때문).
  여러 건을 연속 접수해도 MARS가 순서대로 처리한다.

### 3-2. `GET /documents/jobs/{job_id}` — 적재 진행 상태

```
GET http://mars:9000/documents/jobs/<job_id>
```

- 응답: 위 job 객체. 상태 전이 `queued → running → done | error`
- `done`이면 `chunks_indexed`(적재된 청크 수), `error`면 `error`(사유) 채워짐
- **주의**: MARS의 job 이력은 **인메모리**라 MARS 재시작 시 404가 된다
  (적재된 청크 자체는 Milvus에 유지). 관리자 페이지가 진행 이력을 오래
  보여줘야 하면, **미들웨어가 job_id·상태를 자기 DB에 미러링**할 것을 권장.
- `GET /documents/jobs?limit=20` — 최근 작업 목록(최신순)

### 3-3. `GET /documents` — 적재 문서 목록

```
GET http://mars:9000/documents?offset=0&limit=20&domain=<선택>
```

- 응답: 무한 스크롤 페이지네이션
```json
{
  "documents": [
    {"name": "휴가규정.md", "type": "MD", "domain": "HR",
     "visibility": "ALL", "owning_department": "HR_TEAM",
     "applied_at": "2026-07-05T19:09:47"}
  ],
  "total": 15, "offset": 0, "limit": 20, "has_more": false
}
```
- `has_more`가 `true`면 `offset += limit`으로 다음 페이지
- **보안 주의**: 이 API는 **ACL을 적용하지 않는다.** `DEPT_ONLY` 문서의
  존재(문서명·소유 부서)까지 전부 노출된다. **미들웨어가 관리자 권한을 확인한
  뒤에만** 노출할 것. 일반 사용자에게 그대로 프록시하면 부서 전용 문서명이
  새어나간다.

### 3-4. `DELETE /documents/{name}` — 문서 삭제

```
DELETE http://mars:9000/documents/<URL인코딩된 파일명>
```

- MARS에서 해당 문서의 청크(Milvus)를 삭제하고 BM25 재빌드. **동기 처리**
  (수 초~수십 초 — 완료까지 응답 대기)
- 응답: `{"name": str, "deleted_chunks": int, "deleted_parents": int}`
- **에러**: 404(미적재 문서), 409(다른 적재/삭제 작업 진행 중, 10초 대기 후)
- 미들웨어는 MARS 삭제와 **함께 자기 DB의 원본도 삭제**해 수명 주기를 맞춘다.

---

## 4. 미들웨어가 구현할 흐름

### 4-1. 문서 업로드 (관리자 페이지)

1. 관리자 권한 확인
2. 업로드된 파일을 **미들웨어 DB에 영속 저장** (원본 소유자)
3. MARS `POST /documents`로 바이트 relay → `job_id` 수신
4. `GET /documents/jobs/{job_id}` 폴링으로 진행 상태를 관리자 화면에 표시
   (또는 미들웨어 DB에 상태 미러링)
5. `done`이면 완료 처리, `error`면 사유 표시

### 4-2. 문서 목록/삭제 (관리자 페이지)

- 목록: MARS `GET /documents` 프록시 (관리자 전용 — §3-3 보안 주의)
- 삭제: MARS `DELETE /documents/{name}` + 미들웨어 DB 원본 삭제

### 4-3. 서버 재구축 시 재적재 (운영 절차)

MARS 재구축으로 Milvus·BM25가 비면, **미들웨어가 자기 DB의 원본들을 순회하며
`POST /documents`로 재전송**해 재색인한다. MARS 로컬 원본에 의존하지 않는다.

---

## 5. 관련: 생성 파일(HWPX) fetch-and-store — 이미 동일 패턴

문서 적재의 "미들웨어가 원본 소유"는 새로운 게 아니라, 이미 도입된 생성
파일 처리와 **같은 방향**이다. 참고로 나란히 둔다:

| | 적재 원본 (이 문서) | 생성 파일 (HWPX 등) |
|---|---|---|
| 방향 | 미들웨어 → MARS (relay) | MARS → 미들웨어 (미들웨어가 pull) |
| 트리거 | 관리자 업로드 | SSE `{"type":"file", ...}` 이벤트 |
| MARS 보관 | `UPLOAD_DIR` (임시) | `EXPORT_DIR` (임시, TTL 24h) |
| 영속 보관 | 미들웨어 DB | 미들웨어 DB |

생성 파일 fetch-and-store 상세는 `docs/interfaces.md` §5 `GET /files/{name}` 참조.

---

## 6. 미들웨어 자체 API 설계 (프론트 ↔ 미들웨어)

§3의 MARS API는 미들웨어가 **호출하는** 계약이고, 아래는 미들웨어가 관리자
페이지(프론트)에 **노출할** 자체 API다. 경로·필드는 예시이며 미들웨어 팀
컨벤션에 맞춰 조정한다. `{id}`는 미들웨어 DB의 문서 식별자(문서명이 아닌
내부 PK 권장 — 재업로드·이름 충돌 대비).

### 프론트 노출 API (관리자 페이지)

| # | API | 역할 | 내부 동작 |
|---|---|---|---|
| 1 | `POST /admin/documents` | 문서 업로드 | 원본을 미들웨어 DB 저장 → MARS `POST /documents` relay → job_id 보관 |
| 2 | `GET /admin/documents` | 문서 목록 | 미들웨어 DB 기준(소유자) + 문서별 적재 상태 병기. 검색·도메인 필터·페이지네이션 |
| 3 | `GET /admin/documents/{id}` | 문서 상세 | 메타데이터 + 적재 상태 + 원본 다운로드 링크 |
| 4 | `GET /admin/documents/{id}/status` | 적재 진행 상태 | MARS job 폴링 결과(또는 미러링 값)를 프론트에 전달 |
| 5 | `DELETE /admin/documents/{id}` | 문서 삭제 | MARS `DELETE /documents` + 미들웨어 DB 원본 삭제 (§7 순서 주의) |
| 6 | `GET /admin/documents/{id}/file` | 원본 다운로드/미리보기 | 미들웨어 DB에서 원본 반환 (미들웨어가 원본 소유자) |
| 7 | `POST /admin/documents/{id}/reindex` | 단건 재색인 | 미들웨어 DB 원본 → MARS `POST /documents` 재relay. 실패 복구·재구축 복구용 |
| 8 | `POST /admin/documents/reindex-all` | 전체 재색인 | 미들웨어 DB 전체 원본을 순회 relay (직렬). 재구축 후 일괄 복구 |
| 9 | `PATCH /admin/documents/{id}` | 메타데이터 수정 | domain/visibility/부서 변경 → 같은 원본으로 **재relay**(재적재) |

**항목별 주의**:

- **1 (업로드)**: 브라우저에서 오는 건 통상 `multipart/form-data`다. 미들웨어가
  이를 받아 파일 바이트를 꺼내 MARS로는 `application/octet-stream`으로 변환해
  relay한다(§3-1). **relay 전에 검증**(형식 `.md/.txt/.pdf`, 50MB, domain enum)해
  빠르게 실패시킨다 — MARS 왕복 낭비 방지.
- **2 (목록)**: MARS `GET /documents`가 아니라 **미들웨어 DB를 소스**로 한다
  (미들웨어가 원본 소유자이므로). 각 문서에 적재 상태(적재됨/진행중/실패)를
  병기하면 관리자가 한눈에 본다. 이름 검색은 MARS엔 없으므로 미들웨어 DB에서
  제공. DEPT_ONLY 노출 문제(§3-3)는 관리자 페이지 전제라 무방하나, 권한
  체크는 필수.
- **7·8 (재색인)**: "재구축 시 미들웨어가 재전송"(§4-3) 결정을 실제로 실행하는
  손잡이다 — 이게 없으면 재구축 복구가 수동 스크립트에만 의존한다. 7은 단건
  (job 실패 후 재시도 겸용), 8은 전체(재구축 직후 일괄). MARS는 적재를 직렬
  처리하므로 8은 순차 relay로 충분하다. 재색인은 **같은 `name`이면 MARS가
  갱신(기존 청크 삭제 후 재적재)**하므로 중복 걱정 없다(§3-1).
- **9 (메타데이터 수정)**: ⚠️ `domain`/`visibility`/`owning_department`는 Milvus
  청크에 박혀 있어 **단순 DB 업데이트로 반영되지 않는다.** 미들웨어가 원본
  바이트를 그대로 다시 `POST /documents`(같은 name, 바뀐 쿼리 파라미터)로
  relay해야 MARS가 재적재하며 새 메타로 색인한다. 즉 **7(재색인)의 특수형**
  이다 — 파일은 그대로, 메타만 바꿔 재relay.

### 내부 로직 (프론트 노출 아님)

| API | 왜 필요한가 |
|---|---|
| **정합성 재조정** | 미들웨어 DB ↔ MARS Milvus 불일치 감지·복구 (§7). 관리자 "동기화 점검" 버튼 또는 주기 배치로 노출 가능 |
| **감사 로그** | 누가 언제 무엇을 업로드/삭제/재색인했는지. 미들웨어는 사용자 신원을 아니까 자연스러운 책임 (MARS 감사 로그는 질의만 남김) |

### 선택 (규모 커지면)

- **일괄 업로드/삭제**: 관리자 편의. N개 단건의 반복으로 모델링 가능.
- **상태 푸시(SSE/WebSocket)**: 4(폴링) 대신 진행 상태를 실시간 푸시. 문서
  수가 많거나 적재가 길면 UX 개선.

---

## 7. ★ 두 저장소 정합성 — 이 설계의 핵심 리스크

원본(미들웨어 DB)과 색인(MARS Milvus)이 **분리된 두 저장소**라 어긋날 수 있다.
설계 시 반드시 순서와 복구 전략을 정한다.

**업로드 순서 (권장: 원본 먼저)**
1. 미들웨어 DB에 원본 저장 (`pending` 상태)
2. MARS `POST /documents` relay → job_id 보관, 상태 `indexing`
3. job `done` → `indexed`, `error` → `failed`(원본은 유지, 재색인 가능)
- 2에서 실패해도 원본은 DB에 있으니 재색인(7)으로 복구 가능. **원본을 먼저
  확보**하는 게 핵심 — 반대로 하면 relay 성공 후 원본 저장 실패 시 색인은
  됐는데 원본이 없어 재구축 불가.

**삭제 순서 (권장: MARS 먼저)**
1. MARS `DELETE /documents/{name}` (409면 재시도 — 다른 작업 진행 중)
2. 성공 후 미들웨어 DB 원본 삭제
- MARS를 먼저 지워야 "검색엔 나오는데 원본이 없는" 유령 문서를 피한다.
  MARS 삭제 실패 시 DB 원본을 남겨 재시도 가능하게.

**불일치 감지 (정합성 재조정, #10)**
- MARS `GET /documents` 문서명 집합 ↔ 미들웨어 DB 문서 집합 비교:
  - **미들웨어엔 있는데 MARS엔 없음** → 재구축 등으로 색인 유실 → 재색인(7)
  - **MARS엔 있는데 미들웨어엔 없음** → 고아 색인 → MARS에서 삭제 또는 경고
- 재구축 직후 자동으로 "MARS 비어 있음 → 전체 재색인(8)"이 대표 시나리오.

---

## 8. 전체 수명 주기 요약

```
[업로드]  프론트 → (검증) → 미들웨어 DB 저장 → MARS relay → job 폴링 → indexed
[조회]    프론트 ← 미들웨어 DB(목록/상세/원본) + MARS 적재 상태 병기
[삭제]    프론트 → MARS 삭제 → 미들웨어 DB 삭제
[재색인]  (재구축/실패) → 미들웨어 DB 원본 → MARS relay  ※ 원본이 미들웨어에
                                                          있어 항상 복구 가능
```

---

## 9. 확정/미확정

**확정**:
- 원본 영속 보관 = 미들웨어. MARS 적재·임베딩 능력은 그대로 유지.
- API 4종 시그니처 변경 없음 (미들웨어는 위 계약대로 호출만).

**MARS 쪽 향후 작업 (미들웨어 영향 없음)**:
- `UPLOAD_DIR` 임시 스테이징 TTL 자동 정리 (기본 24h) — 내부 정리라 미들웨어
  연동에 영향 없음.

**미들웨어 확인 필요**:
- 미들웨어 DB에 원본 BLOB 저장 가능 여부 (이 설계의 대전제).
- job 상태를 미들웨어가 미러링할지 (MARS 인메모리 이력의 재시작 취약성 대응).

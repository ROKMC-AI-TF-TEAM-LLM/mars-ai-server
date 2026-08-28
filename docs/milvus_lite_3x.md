# milvus_lite_3x.md — milvus-lite 3.x 전환 검토

**결론: 전환했다.** 검색 품질은 실측상 동일하고, Windows 개발이 운영과 같은
엔진(Milvus Lite)을 쓰게 되어 서버/Lite 동작 차이에서 오는 버그가 사라진다.
Docker Desktop + Milvus 이미지 ~2.9GB 반입도 필요 없어진다.

검토·전환 시점 2026-08-27~28, 측정 환경 WSL2 AlmaLinux 9 / RTX 4050 6GB.
`milvus-lite==2.4.11` → **`3.2.1`**, `pymilvus==2.5.4` 유지.

> ⚠️ **기존 DB는 읽히지 않는다** (저장 포맷 비호환, §2-1).
> 버전을 올린 뒤에는 **전체 재적재**가 필요하다.

---

## 1. 왜 검토했나

Windows에는 `milvus-lite` 휠이 없어 개발 환경이 **Docker Milvus 서버**를 쓴다.
그 결과 개발과 운영이 다른 엔진을 돌게 되고, **운영에서만 터지는 버그**가 생겼다
(`docs/troubleshooting.md` ⑩⑪).

내부망 반입 시 Docker Desktop(~500MB) + Milvus 이미지(~2.4GB)를 함께 들고
들어가야 하는 부담도 있다. 반입 절차가 복잡할수록 이 2.9GB는 실질적인 비용이다.

`milvus-lite` **3.x가 네이티브 C++에서 순수 파이썬으로 재작성되면서
Windows 제약이 사라졌다.** 그래서 검토했다.

| | 2.4.11 (현행) | 3.2.1 |
|---|---|---|
| 구현 | 네이티브 C++ (`milvus` ELF, `libknowhere.so`) | **순수 파이썬** |
| 휠 태그 | `manylinux` / `macosx` | **`py3-none-any`** |
| Windows | ❌ 불가 | ✅ **가능** |

---

## 2. 버전을 교체하면 바뀌는 것

### 2-1. 저장 포맷이 호환되지 않는다 (전체 재적재 필요)

2.4.11은 **단일 파일**, 3.x는 **디렉터리**(세그먼트·WAL·매니페스트 구조)다.
기존 DB를 그대로 열면 즉시 실패한다.

```python
os.makedirs(data_dir, exist_ok=True)
FileExistsError: [Errno 17] File exists: '/root/mars/data/milvus_ax.db'
```

**재적재가 필수다.** 다만 문서 원본과 임베딩 서버만 있으면 되고 자동으로 진행된다
(15문서 1,519청크 기준 **104초**). 운영 코퍼스가 커지면 그만큼 늘어난다.

### 2-2. 컬렉션 load를 명시적으로 호출해야 한다 ★

Milvus는 컬렉션이 `released` 상태면 search/query를 거부한다.

```
MilvusException: (code=101, message=Collection 'company_docs' is in state
'released'; call load() before search/get/query)
```

**2.4.11은 자동 로드해 줘서 코드가 `load_collection()`을 한 번도 부르지 않았다.**
3.x는 진짜 Milvus 시맨틱을 따라 명시적 로드를 요구한다.

실측에서 **적재 4배치 중 3개가 이걸로 실패**했다. 도메인별로 `bulk_ingest.py`를
따로 실행해 프로세스가 4개였는데:

| 배치 | 결과 | 이유 |
|---|---|---|
| ① FINANCE_LEGAL | 성공 | `create_collection`이 실제로 생성 → **생성 직후는 loaded** |
| ② DIRECTIVE | 실패 | 컬렉션이 이미 있어 조기 반환 → **released 상태 그대로** |
| ③ GENERAL | 실패 | 〃 |
| ④ HR | 실패 | 〃 |

터진 지점은 `embed_and_upsert` → `rebuild_bm25()` → `fetch_all_children()` →
`query_iterator()`다. **BM25는 부분 갱신이 안 돼 매번 전체 청크를 조회**하는데,
그 조회가 released 컬렉션에 날아갔다.

**자식·부모 두 컬렉션 모두** 해당한다 — 자식만 고쳤을 때 `document_parents`에서
같은 오류가 났다.

> **이 수정은 이미 반영했다** (커밋 `176865d`, `ensure_loaded()`).
> **3.x 전용이 아니라 원래 맞는 코드다** — Milvus 서버도 재시작 후 released가
> 되므로 현행 Docker 환경에서도 필요하다.

### 2-3. 의존성이 늘어난다 — `faiss-cpu` 하나

`milvus-lite` 3.x는 `faiss-cpu`·`pyarrow`·`numpy`·`grpcio`를 요구한다.
이 중 **실제로 새로 반입해야 하는 건 `faiss-cpu` 하나뿐이다.**

| 패키지 | 크기 | 신규 여부 |
|---|---:|---|
| `faiss-cpu` | 67 MB | **★ 신규** |
| `pyarrow` | 157 MB | 이미 있음 — `datasets`가 끌고 온다 (lock에 `25.0.0`) |
| `numpy` | 74 MB | 이미 있음 |
| `grpcio` | 16 MB | 이미 있음 (`pymilvus`) |
| `milvus_lite` | 4 MB | 교체 |

Windows 휠도 존재한다: `faiss_cpu-1.15.0-cp311-cp311-win_amd64.whl`.

### 2-4. 경로는 여전히 `.db`로 끝나야 한다

3.x는 그 경로에 **디렉터리**를 만들지만, `pymilvus`가 URI 형식을 먼저 검증한다.

```
ConnectionConfigException: uri: .../probe_db is illegal,
needs start with [unix, http, https, tcp] or a local file endswith [.db]
```

`MILVUS_LITE_PATH=./data/milvus_ax.db` 처럼 **확장자를 유지한다.**
이름은 파일 같지만 실제로는 디렉터리다 — 백업·삭제 시 `rm -rf`가 필요하다.

### 2-5. gRPC 미구현 로그가 쏟아진다 (무해하지만 반드시 지운다)

매 작업마다 ERROR 트레이스백이 나온다.

```
ERROR grpc._server: Exception calling application: Method not implemented!
  File ".../pymilvus/grpc_gen/milvus_pb2_grpc.py", line 991, in AllocTimestamp
    raise NotImplementedError('Method not implemented!')
WARNING [__setup_ts_by_request]: failed to get mvccTs from milvus server,
        use client-side ts instead
```

**원인** — `pymilvus`는 분산 Milvus 서버를 전제로 `AllocTimestamp`(전역 타임스탬프
할당)를 호출한다. 임베디드 milvus-lite에는 조율할 노드가 없어 구현이 없고,
클라이언트 타임스탬프로 폴백한다.

**정합성 영향 없음 (실측)** — MARS는 적재 직후 BM25 재빌드를 위해 전체 조회를
하므로 방금 insert한 청크가 반드시 보여야 한다. 50행씩 5회 반복 검증:

| 경로 | 결과 |
|---|---|
| `flush()` 후 조회 (앱의 실제 경로) | **5/5 정확** |
| `flush()` 없이 조회 | **정확** (300/300) |

단일 프로세스 임베디드라 분산 타임스탬프 조율 자체가 불필요하다.

**해결** — `logging_setup.MilvusLiteNoiseFilter` 로 **이 두 메시지만** 지운다.
지우지 않으면 ERROR 트레이스백이 로그를 덮어 진짜 장애를 놓친다.

> ⚠️ **로거에 건 필터는 하위 로거에서 전파돼 온 레코드에 적용되지 않는다.**
> `mvccTs` 경고는 `pymilvus.orm.iterator` 가 내고 `pymilvus` 핸들러로 전파되므로,
> `pymilvus` 로거에만 걸면 그대로 남는다. **로거와 핸들러 양쪽에** 걸어야 한다.

### 2-6. search 결과의 PK 위치가 바뀐다

| 환경 | PK 위치 |
|---|---|
| Milvus 서버 (개발 Docker) | `hit["chunk_id"]` |
| Milvus Lite **2.4.11** | `hit["id"]` |
| Milvus Lite **3.2.1** | **`hit["chunk_id"]`** (서버와 같아짐) |

**코드 수정은 필요 없다.** `_primary_key()`가 이미 양쪽을 모두 읽는다
(`docs/troubleshooting.md` ⑪). 한쪽만 읽지 않기로 한 판단이 결과적으로 맞았다.

### 2-7. `pymilvus 2.5.4`와 일부 호출이 맞지 않는다 (앱 경로 아님)

`list_collections()`는 gRPC 스키마가 달라 실패한다.

```
MilvusException: Protocol message ShowCollectionsResponse has no "shards_num" field.
```

**앱은 이 함수를 쓰지 않는다.** 앱이 실제로 쓰는 호출은 전부 정상 동작한다:

```
create_collection, create_schema, prepare_index_params, has_collection,
insert, flush, delete, drop_collection, query, query_iterator, search
```

> ⚠️ **최초 검토에서 이걸 "앱이 못 돈다"는 근거로 잘못 판단했다.**
> 진단 스크립트가 호출한 함수였을 뿐인데 앱의 문제로 보고했다.
> 같은 착각을 반복하지 않도록 기록해 둔다 — **오류를 발견하면 그 함수를
> 앱이 실제로 호출하는지 먼저 확인한다.**

---

## 3. 테스트 결과

같은 코퍼스(15문서 / 1,519청크), 같은 질의, 같은 임계값(`RERANK_SCORE_THRESHOLD=0.1`)
으로 두 버전을 각각 측정했다.

### 3-1. 검색 품질 — 동일

| 지표 | 2.4.11 | **3.2.1** | 판정 |
|---|---:|---:|---|
| `hit@n` | 0.923 | **0.923** | **동일** |
| `hit@fuse` | 1.0 | **1.0** | **동일** |
| `empty_rate` | 0.077 | **0.077** | **동일** |

### 3-2. 적재 — 정상

| | 2.4.11 | 3.2.1 |
|---|---:|---:|
| 문서 | 15건 | **15건** |
| 자식 청크 | 1,519행 | **1,519행** |
| 적재 배치 | 4/4 | **4/4** (load 수정 후) |
| 소요 | — | 104초 |

도메인별 분포도 일치한다 (FINANCE_LEGAL 398 / DIRECTIVE 430 / GENERAL 686 / HR 5).

### 3-3. 검색 속도 — 2.3배 느림

`SEARCH_TOP_K=20`, ACL 필터 적용, 워밍업 후 30회.

| 지표 | 2.4.11 | **3.2.1** | 배수 |
|---|---:|---:|---:|
| 중앙값 | 1.5 ms | **3.5 ms** | 2.3× |
| 평균 | 1.6 ms | 3.7 ms | 2.3× |
| p95 | 1.8 ms | 4.7 ms | 2.6× |
| 최대 | 2.2 ms | 5.8 ms | 2.6× |

**절대 차이는 2ms다.** MARS 한 질의의 전체 지연에서 LLM 생성이 수 초,
리랭킹이 수백 ms를 차지하므로 **체감 차이는 사실상 없다.**

다만 이건 **1,519청크에서의 수치다.** 파이썬 구현은 코퍼스가 커질수록 C++ 대비
불리해질 수 있다. `AUTOINDEX`가 규모에 따라 인덱스를 바꾸므로 선형 악화는
아니겠지만, **운영 규모에서 재측정 없이 단정할 수 없다.**

### 3-4. 유닛 테스트 — 통과

두 버전 모두 전량 통과했다.

### 3-5. 기능 확인 (Windows, 격리 venv)

`win32`에서 직접 확인한 항목:

| 항목 | 결과 |
|---|---|
| 임베디드 접속 | ✅ |
| `AUTOINDEX` 컬렉션 생성 | ✅ (⑩이 사라진다) |
| 한국어 본문 적재 | ✅ |
| ACL 스칼라 필터 검색 | ✅ |
| `query` 필터 | ✅ |
| `query_iterator` | ✅ |

---

## 4. 얻는 것

| | 현행 (2.4.11 + Docker) | 3.x |
|---|---:|---:|
| **Windows 반입량** | ~2.9 GB | **+67 MB** (faiss-cpu) |
| Docker Desktop | 필요 | **불필요** |
| Milvus 이미지 | 필요 | **불필요** |
| Windows 개발 엔진 | Milvus **서버** | **Lite** (운영과 동일) |

**마지막 항목이 가장 크다.** 지금은 Windows 개발이 서버 모드라 운영(Lite)과 다르고,
그 격차가 ⑩(HNSW 거부)·⑪(PK 키 차이)을 **운영에서만 드러나게** 만들었다.
3.x로 가면 Windows에서 돌리는 것이 곧 운영과 같은 엔진이 되어 이 격차가
구조적으로 사라진다.

부수적으로, 내부망에 Docker Desktop을 설치·유지·갱신하는 운영 부담도 없어진다.

---

## 5. 전환 내역

| 파일 | 변경 |
|---|---|
| `requirements.txt` | `milvus-lite` 2.4.11 → **3.2.1**, `faiss-cpu==1.15.0` 추가 |
| `requirements-linux-app.txt` | 〃 |
| `requirements-dev-windows.txt` | **`milvus-lite`·`faiss-cpu` 추가** (제외 사유 소멸) |
| `requirements-dev-windows.lock` | 〃 |
| `scripts/dev_setup.ps1` | **Docker 단계 제거** (5단계 → 4단계), Docker 사전 검사 제거 |
| `.env.dev.example` | `MILVUS_LITE_PATH`를 파일 경로로 |
| `vectorstore.py` / `parent_store.py` | `ensure_loaded()` (커밋 `176865d`) |
| `logging_setup.py` | `MilvusLiteNoiseFilter` — gRPC 미구현 소음 제거 |

`pymilvus`는 **2.5.4를 유지한다.** 앱이 쓰는 호출은 전부 정상 동작하며(§2-6),
함께 올리면 검증 범위가 불필요하게 넓어진다.

### 기존 환경에서 올릴 때

**기존 DB를 지우고 재적재해야 한다.** 이름은 `.db`지만 3.x에서는 디렉터리다.

```bash
rm -rf data/milvus_ax.db data/bm25_index     # ★ -rf (디렉터리다)
PYTHONPATH=src python scripts/bulk_ingest.py --dir <도메인별 디렉터리> --domain <도메인> ...
PYTHONPATH=src python scripts/eval_retrieval.py    # hit@n 확인
```

`--domain`은 **문서 성격에 맞게 디렉터리를 나눠** 지정한다
(`docs/troubleshooting.md` ⑭).

### 남은 공백

- **운영 규모 코퍼스에서의 검색 속도** — 이번 측정은 1,519청크 기준이다.
  파이썬 구현이라 규모가 커질수록 C++ 대비 불리해질 수 있다.
  L40 배포 후 실제 코퍼스로 재측정한다.
- `serving/milvus-dev/`(Docker compose·etcd 설정)는 **당장 지우지 않았다.**
  3.x 운영이 안정된 뒤 정리한다 — 되돌릴 필요가 생길 수 있다.

---

## 재현 방법

```bash
# 격리 venv (현재 환경을 건드리지 않는다)
python3.11 -m venv /tmp/venv-lite3
/tmp/venv-lite3/bin/pip install "setuptools==75.6.0" "pymilvus==2.5.4" "milvus-lite==3.2.1"

# 기존 DB는 열리지 않는다 (포맷 비호환 확인)
/tmp/venv-lite3/bin/python -c "
from pymilvus import MilvusClient
MilvusClient('data/milvus_ax.db')"      # FileExistsError

# 전환 후 검증
PYTHONPATH=src venv-app/bin/python scripts/eval_retrieval.py
PYTHONPATH=src venv-app/bin/python -m pytest tests/unit_tests -q
```

# troubleshooting.md — 내부망 이관 중 발생한 문제와 해결

개발 환경을 **Windows + Docker Milvus + llama.cpp** 에서
**WSL2 AlmaLinux 9 + Milvus Lite + vLLM** 으로 옮기면서 겪은 문제를 순서대로 기록한다.
목적은 두 가지다.

1. **내부망(L40)에서 같은 문제를 다시 겪지 않는다** — 에어갭에서는 검색으로 해결할 수 없다
2. 개발 환경을 재구축할 때 참조한다

> 실측 수치와 가설 검증은 `docs/experiments.md`, 배포 절차는 `docs/deploy_l40.md`가 소유한다.
> 이 문서는 **문제 상황과 해결**만 다룬다.

## 왜 환경을 옮겼나

기존 개발 환경은 운영과 세 군데가 달랐다.

| | 기존 개발 | 운영 L40 |
|---|---|---|
| OS | Windows | RedHat |
| 벡터DB | **Docker Milvus 서버** | **Milvus Lite** |
| LLM 서빙 | **llama.cpp** | **vLLM** |

이 차이 때문에 **운영에서만 터지는 버그가 개발에서 보이지 않았다.**
아래 ⑤⑥⑦⑧ 네 건이 전부 그런 경우로, 그대로 배포했으면 내부망에서
디버깅해야 했다. 에어갭에서 그건 매우 비싸다.

---

# A. 라이브러리 반입 준비

## ① `pip download -r requirements.txt` 가 ResolutionImpossible

**증상**

```
ERROR: Cannot install -r requirements.txt (line N) and transformers==4.44.2
because these package versions have conflicting dependencies.
```

**원인**

`vllm==0.11.0` 은 `transformers>=4.55` 를 요구하는데,
`FlagEmbedding`(임베딩·리랭커)은 `transformers==4.44.2` 에 묶여 있다.
**한 venv에 공존할 수 없다.**

**해결**

요구사항 파일을 용도별로 쪼개고 venv를 분리한다.

| 파일 | venv | transformers |
|---|---|---|
| `requirements-linux-llm.txt` | `venv-llm` | 4.57.1 (vLLM용) |
| `requirements-linux-app.txt` | `venv-app` | 4.44.2 (FlagEmbedding용) |

**재발 방지** — 두 venv는 운영에서도 분리한다. 합치려는 시도를 하지 않는다.

## ② `triton==3.5.0` 이 설치되지 않는다

**증상** — 해당 버전의 휠이 존재하지 않아 다운로드 실패.

**원인**

`requirements.txt` 에 `triton==3.5.0` 으로 적혀 있었으나,
`torch==2.8.0` 이 실제로 요구하는 버전은 **`triton==3.4.0`** 이다.

**Windows에는 triton이 아예 없어서 이 오류가 그동안 드러나지 않았다.**

**해결** — `triton==3.4.0` 으로 정정 (커밋 `cc4cb22`).

**재발 방지** — torch를 올릴 때 triton 핀을 함께 확인한다. 둘은 한 쌍이다.

## ③ lock 파일에 `setuptools`·`wheel` 이 빠졌다

**증상** — 에어갭 설치가 sdist 빌드 단계에서 실패한다.

```
ERROR: Could not find a version that satisfies the requirement wheel
       (from versions: none)
```

**원인**

`pip freeze` 는 **`setuptools`·`pip`·`wheel` 을 기본으로 제외한다.**
(`pip list --format=freeze` 는 포함한다 — 둘의 동작이 다르다.)

lock에 없으면 에어갭에서 **아예 설치가 되지 않는다** — `pip` 은 자기 자신 외에
`setuptools`·`wheel` 을 제공하지 않으므로, sdist 빌드에 필요한 두 패키지가
반입 목록에서 통째로 빠진다.

> **버전 상한은 필요 없다** (2026-08-28 정정). 예전엔 `pymilvus 2.5.4` 가
> `pkg_resources` 를 요구해 `setuptools<81` 로 묶었으나, `pymilvus 3.x` 는 쓰지
> 않는다. `setuptools 84.0.0`(`pkg_resources` 제거본)에서 sdist 빌드와 테스트
> 386개 전부 통과함을 실측했다. **필요한 건 "상한"이 아니라 "목록에 있는 것"이다.**

**왜 sdist가 등장하나** — `FlagEmbedding` 은 wheel이 없어(1.4.0부터 제공)
**sdist로 설치되고**, 그 빌드에 `setuptools` 와 `wheel` 이 모두 필요하다.

**해결** — lock 생성 시 `pip list --format=freeze` 를 쓰거나 두 패키지를 명시한다.

`setuptools` 만 넣고 `wheel` 을 빠뜨리면 **거기서 다시 멈춘다** (실제로 겪었다).
`--no-index` 예행연습을 하지 않으면 내부망에 들어가서야 발견한다.

### ③-1. 목록에 넣는 것만으로는 부족하다 — 설치 순서와 플래그

두 패키지를 반입해도 **한 명령에 몰아 깔면 의도대로 되지 않는다** (WSL 실측).

| 증상 | 원인 | 해결 |
|---|---|---|
| 반입한 `setuptools 84.0.0`이 아니라 venv 기본값(65.5.1)이 남는다 | 이미 있으면 pip이 업그레이드하지 않는다 | `--upgrade` 로 **먼저** 설치 |
| `kiwipiepy_model` 이 legacy `setup.py install` 경로 | 빌드 시점에 `wheel` 이 아직 없다 | 〃 |
| `FlagEmbedding` 이 legacy 경로 | **설치에도 `--no-binary` 를 줬다** | 설치에는 주지 않는다 |

```bash
pip install --no-index --find-links wheels/ --upgrade setuptools wheel   # ① 먼저
pip install --no-index --find-links wheels/ -r requirements-linux-app.txt  # ② --no-binary 없이
```

**`--no-binary` 는 다운로드 전용 플래그다.** `FlagEmbedding` sdist를 *받기* 위해
필요하고, *설치* 때는 폴더에 sdist밖에 없으므로 pip이 알아서 쓴다.
설치에도 주면 PEP 517 빌드가 막혀 legacy 경로를 탄다.

셋 다 고치면 경고가 완전히 사라진다 (실측 확인).

**재발 방지** — lock에 `setuptools` 와 `wheel` 이 **둘 다** 있는지 확인하고,
반입 전 반드시 `--no-index` 로 설치 예행연습을 한다 (`deploy_l40.md` §1-2).

```bash
grep -iE "^setuptools|^wheel" requirements-*.lock
```

---

# B. WSL 배포판 선택

## ④ Ubuntu 26.04의 Python이 3.14라 휠이 안 맞는다

**증상** — 받아 둔 `cp311` 휠이 설치되지 않음.

**원인** — Ubuntu 26.04 기본 Python은 3.14. 프로젝트는 **3.11** 고정이다.

**해결**

**AlmaLinux 9** 로 교체했다. 이유가 하나 더 있다 —
AlmaLinux는 **RHEL 클론**이라 운영 RedHat과 계열이 같다.
`dnf` 패키지명, glibc 버전, 시스템 경로가 전부 운영과 일치한다.

```powershell
wsl --install AlmaLinux-9
```

**재발 방지** — 개발 배포판은 **운영과 같은 계열**로 고른다. 편의보다 일치가 우선이다.

---

# C. vLLM 기동

## ⑤ vLLM이 gcc 없이는 뜨지 않는다 ★ 운영 블로커

**증상** — 기동 중 C 컴파일러를 찾지 못하고 실패.

**원인**

vLLM은 런타임에 **JIT 컴파일**을 한다. 파이썬 패키지만 반입하면 부족하고
**C 컴파일러가 OS에 있어야 한다.**

**해결**

```bash
dnf install -y gcc gcc-c++
```

**재발 방지** — `docs/deploy_l40.md` 의 에어갭 전제에 gcc를 필수로 명시했다 (커밋 `931d468`).
**내부망에는 RPM도 함께 반입해야 한다.**

## ⑥ 헤더가 없으면 기동은 되고 첫 요청에 죽는다 ★ 운영 블로커 (가장 위험)

**증상**

```
Application startup complete.        ← 정상으로 보인다
```

기동은 성공하고 `/v1/models` 도 응답한다. 그런데
**`response_format=json_schema` 를 쓰는 첫 요청에서 엔진 프로세스가 사망**한다.

**원인**

`python3.11-devel` (Python C 헤더)이 없어 구조화 출력 백엔드의
JIT 컴파일이 실패한다. **기동 시점에는 컴파일하지 않으므로 그때는 멀쩡하다.**

**해결**

```bash
dnf install -y python3.11-devel
```

**왜 가장 위험한가** — 헬스체크가 통과한다. 배포 검증을 `/health` 로만 하면
**정상으로 판정되고, 실사용 첫 질의에서 서비스가 죽는다.**

**재발 방지** — 배포 검증에 **실제 `json_schema` 요청 1건**을 반드시 포함한다.
포트 응답 확인만으로는 부족하다.

## ⑦ `pkill -f "vllm serve"` 가 자기 자신을 죽인다

**증상** — 명령을 실행하면 셸까지 함께 종료된다.

**원인**

`pkill -f` 는 **명령줄 전체**를 대상으로 매칭한다.
`pkill -f "vllm serve"` 라는 명령줄 자체가 `"vllm serve"` 를 포함하므로 자신이 걸린다.

**해결**

```bash
pkill -f "bin/vllm"      # 실행 파일 경로로 매칭
```

## ⑧ `--gpu-memory-utilization 0.85` 가 실패한다 (개발 노트북 한정)

**증상**

```
ValueError: Free memory on device (4.94/6.0 GiB) on startup is less than
desired GPU memory utilization (0.85, 5.1 GiB).
```

**원인**

`--gpu-memory-utilization` 은 **전체 VRAM 기준 비율**이다. 6.0 GiB × 0.85 = 5.1 GiB.
그런데 Windows 디스플레이가 ~1 GiB를 이미 쓰고 있어 **가용은 4.94 GiB뿐**이다.

**해결** — RTX 4050 6GB에서는 **`0.78` 이 상한**이다.

**운영에서는 무관** — L40 48GB는 여유가 충분하다.

## ⑨ 중복 기동으로 VRAM이 부족해진다

**증상** — ⑧과 같은 오류인데 가용 메모리가 더 적게(2.62 GiB) 나온다.

**원인** — **이미 떠 있는 vLLM을 못 보고 또 띄웠다.**

**해결** — 띄우기 전에 항상 확인한다.

```bash
curl -s localhost:8000/v1/models >/dev/null && echo "이미 떠 있음" || echo "비어 있음"
```

**기동 순서 규칙** — vLLM을 **먼저** 올린다.
vLLM은 큰 연속 블록을 통째로 잡아야 해서, 임베딩·리랭커가 먼저 자리를 차지하면
파편화로 실패한다.

```
① vLLM (8000)  →  ② 임베딩 (8001)  →  ③ 리랭커 (8002)
```

---

# D. Milvus Lite

## ⑩ Milvus Lite가 HNSW 인덱스를 거부한다 ★ 운영 블로커

**증상** — `create_collection` 단계에서 실패. **서버가 아예 뜨지 않는다.**

**원인**

Milvus Lite는 로컬 모드에서 **`FLAT`·`IVF_FLAT`·`AUTOINDEX` 만** 지원한다.
HNSW는 **Docker Milvus 서버에서만** 쓸 수 있다.

개발이 Docker Milvus였기 때문에 이 제약이 드러나지 않았다.

**해결** — `AUTOINDEX` 로 변경 (커밋 `9b895f5`, [vectorstore.py](../src/ax_rag/shared/vectorstore.py)).

```python
index_params.add_index(
    field_name="embedding",
    index_type="AUTOINDEX",
    metric_type="COSINE",
)
```

**검색 품질 영향 없음** — 근사 검색으로 인한 recall 손실을 의심해
`FLAT`(완전 탐색)과 비교했으나 **결과가 완전히 동일**했다 (`docs/experiments.md` 기각 가설).

**기존 컬렉션 마이그레이션** — `scripts/migrate_index_type.py`.
임베딩을 읽어 재적재하므로 **재임베딩이 필요 없고**, `chunk_id`·`parent_id` 를
보존해 부모 컬렉션과 BM25 인덱스를 다시 만들 필요가 없다.

## ⑪ dense 검색이 `KeyError: 'chunk_id'` 로 죽는다 ★ 운영 블로커

**증상** — dense 검색 전체가 예외로 실패.

**원인**

**Milvus 서버와 Lite의 search 결과 구조가 다르다.**

| 환경 | PK 위치 |
|---|---|
| 서버 (개발 Docker) | `hit["chunk_id"]` — 실제 필드명 키 |
| **Lite (운영 L40)** | `hit["id"]` / `entity["chunk_id"]` — **PK가 `"id"` 로 온다** |

한쪽만 읽으면 다른 환경에서 통째로 죽는다.

**해결** — 양쪽을 모두 읽는다 (커밋 `e569252`, [dense_retrieve.py:75](../src/ax_rag/query_graph/nodes/dense_retrieve.py#L75)).

```python
def _primary_key(hit: dict, entity: dict) -> str:
    return str(hit.get("chunk_id") or entity.get("chunk_id") or hit.get("id") or "")
```

**재발 방지** — 유닛 테스트 `tests/unit_tests/test_dense_primary_key.py` 가
**양쪽 형태를 모두** 검증한다.

## ⑫ Milvus Lite는 WSL과 Windows로 분리할 수 없다

**질문** — "vLLM과 Milvus만 WSL에 두고 앱은 Windows에서 돌릴 수 있나?"

**답: vLLM은 되고, Milvus Lite는 안 된다.**

| 구성 요소 | 분리 | 이유 |
|---|---|---|
| vLLM (8000) | ✅ | TCP 포트 — WSL2가 Windows `localhost` 로 자동 포워딩 |
| 임베딩·리랭커 (8001/8002) | ✅ | 동일 |
| **Milvus Lite** | ❌ | **유닉스 도메인 소켓** + Windows 빌드 없음 |

**원인**

Milvus Lite는 서버가 아니라 **임베디드 라이브러리**다.
앱 프로세스가 직접 링크해서 로컬 바이너리와 **유닉스 도메인 소켓**으로 통신한다.

```python
return self._servers[str(path)].uds_path     # TCP 포트가 아니다
```

- **포워딩할 TCP 포트가 존재하지 않는다**
- 동봉 바이너리(`milvus`, `libknowhere.so`)가 **리눅스 ELF** — Windows 빌드가 없다

**결론** — 앱이 Windows에 있으면 Docker Milvus로 돌아가야 하고,
그건 ⑩⑪을 다시 보이지 않게 만든다. **전부 WSL에 두는 것이 맞다.**

> 단, 이건 **`milvus-lite 2.4.11` 기준**이다. 3.x는 순수 파이썬으로 재작성돼
> Windows에서도 동작한다 — 전환 검토와 실측은 `docs/milvus_lite_3x.md` 참조.

## ⑬ 컬렉션이 released 상태면 조회가 거부된다

**증상**

```
MilvusException: (code=101, message=Collection 'company_docs' is in state
'released'; call load() before search/get/query)
```

**원인**

Milvus는 컬렉션이 `released`면 search/query를 거부한다. 로드는 **컬렉션 생성
직후에만 유지**되고, 새 프로세스가 기존 컬렉션을 열면 released다.

Milvus Lite 2.4.11이 이걸 자동으로 해 줘서 코드가 `load_collection()`을 한 번도
부르지 않았다. **Milvus 서버는 재시작하면 released가 되고**, milvus-lite 3.x도
서버와 같은 시맨틱을 따른다.

**해결** — `ensure_loaded()`로 조회 전 로드를 보장한다 (커밋 `176865d`).
**자식·부모 두 컬렉션 모두** 필요하다 — 자식만 고치면 `document_parents`에서 같은
오류가 난다.

**터지는 지점** — `rebuild_bm25()` → `fetch_all_children()` → `query_iterator()`.
BM25는 부분 갱신이 안 돼 매번 전체 청크를 조회하기 때문이다.

---

# E. 데이터 적재

## ⑭ 도메인을 일괄 지정하면 검색 정확도가 떨어진다

**증상** — `hit@fuse` 가 100% 여야 하는데 **88.5%** 로 측정됨.

**원인**

적재 시 모든 문서에 같은 도메인을 지정했다.

```bash
# ✗ 틀렸다 — 문서마다 도메인이 다르다
python scripts/bulk_ingest.py --dir sample_docs --domain HR
```

도메인은 **문서별 속성**이라 일괄 지정하면 훈령·법령 문서가 `HR` 로 적재되고,
도메인 한정 검색에서 배제된다.

**해결** — 문서별로 올바른 도메인을 지정해 재적재 → **100% 회복**.

**참고** — 이 오류를 근사 검색 탓으로 의심해 `FLAT` 으로 바꿔 봤으나
결과가 동일했다. **인덱스가 아니라 적재 데이터 문제였다.**
가설을 세울 때 데이터를 먼저 의심한다.

## ⑮ 마이그레이션 백업이 `numpy.float32` 직렬화로 실패한다

**증상** — `TypeError: Object of type float32 is not JSON serializable`

**원인** — Milvus가 반환한 임베딩이 `numpy.float32` 배열인데 `json.dump` 가 처리하지 못한다.

**해결**

```python
json.dump(data, f, default=float)
```

**백업이 drop보다 먼저 실행되는 순서였기에 데이터 손실은 없었다.**
마이그레이션 스크립트는 **백업 → 검증 → drop** 순서를 반드시 지킨다.

---

# 요약 — 내부망 반입 체크리스트

이 문서에서 도출된, **L40 배포 전에 반드시 확인할 항목**이다.

| | 항목 | 확인 방법 |
|---|---|---|
| ☐ | `gcc`·`gcc-c++` RPM 반입 | `gcc --version` |
| ☐ | `python3.11-devel` RPM 반입 | `rpm -q python3.11-devel` |
| ☐ | venv 2개 분리 (`venv-llm`/`venv-app`) | `transformers` 버전이 각각 4.57.1 / 4.44.2 |
| ☐ | `triton==3.4.0` (3.5.0 아님) | `pip show triton` |
| ☐ | lock에 `setuptools` 포함 | `grep setuptools requirements-*.lock` |
| ☐ | 벡터 인덱스가 `AUTOINDEX` | 컬렉션 생성 성공 여부 |
| ☐ | **`json_schema` 요청 1건 실제 성공** | 아래 명령 |
| ☐ | 문서별 도메인이 올바른가 | `scripts/eval_retrieval.py` |

**`/health` 통과만으로 배포 검증을 끝내지 않는다** — ⑥ 때문이다.

```bash
curl -s localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"<서빙명>","messages":[{"role":"user","content":"테스트"}],
       "response_format":{"type":"json_schema","json_schema":{"name":"t","schema":
       {"type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"]}}}}'
```

이 요청이 성공해야 실제로 배포가 끝난 것이다.

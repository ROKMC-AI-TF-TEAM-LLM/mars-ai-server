"""query_graph/acl.py 유닛 테스트 — 보안 테스트 (roadmap 3단계 DoD 필수)."""

from __future__ import annotations

from ax_rag.query_graph.acl import build_acl_filter_expr, filter_by_acl

_CANDIDATES = [
    {"chunk_id": "a", "domain": "HR", "owning_department": "HR_TEAM", "visibility": "ALL"},
    {"chunk_id": "b", "domain": "HR", "owning_department": "HR_TEAM", "visibility": "DEPT_ONLY"},
    {"chunk_id": "c", "domain": "TECH", "owning_department": "TECH_TEAM", "visibility": "ALL"},
    {
        "chunk_id": "d",
        "domain": "FINANCE_LEGAL",
        "owning_department": "FIN_TEAM",
        "visibility": "DEPT_ONLY",
    },
]


class TestFilterByAcl:
    def test_타_부서_DEPT_ONLY_청크는_반드시_제거된다(self) -> None:
        """보안 핵심: TECH_TEAM 사용자는 타 부서 DEPT_ONLY(b, d)를 볼 수 없다."""
        result = filter_by_acl(_CANDIDATES, "GENERAL", "TECH_TEAM")
        ids = {c["chunk_id"] for c in result}
        assert "b" not in ids
        assert "d" not in ids
        assert ids == {"a", "c"}

    def test_자기_부서_DEPT_ONLY는_통과한다(self) -> None:
        result = filter_by_acl(_CANDIDATES, "GENERAL", "HR_TEAM")
        assert {c["chunk_id"] for c in result} == {"a", "b", "c"}

    def test_부서_누락_시_ALL만_통과한다(self) -> None:
        """user_department 누락 → 가장 제한적 폴백 (interfaces.md §5)."""
        result = filter_by_acl(_CANDIDATES, "GENERAL", "")
        assert {c["chunk_id"] for c in result} == {"a", "c"}

    def test_도메인_지정_시_타_도메인은_제거된다(self) -> None:
        result = filter_by_acl(_CANDIDATES, "HR", "HR_TEAM")
        assert {c["chunk_id"] for c in result} == {"a", "b"}

    def test_GENERAL은_전_도메인을_검색한다(self) -> None:
        result = filter_by_acl(_CANDIDATES, "GENERAL", "FIN_TEAM")
        assert {c["chunk_id"] for c in result} == {"a", "c", "d"}

    def test_미지의_visibility_값은_fail_closed로_배제된다(self) -> None:
        weird = [{"chunk_id": "x", "domain": "HR", "owning_department": "HR_TEAM"}]  # 필드 누락
        assert filter_by_acl(weird, "GENERAL", "HR_TEAM") == []

        weird2 = [
            {
                "chunk_id": "y",
                "domain": "HR",
                "owning_department": "HR_TEAM",
                "visibility": "SECRET",
            }
        ]
        assert filter_by_acl(weird2, "GENERAL", "HR_TEAM") == []


class TestBuildAclFilterExpr:
    def test_부서가_있으면_ALL_또는_자기_부서_DEPT_ONLY(self) -> None:
        expr = build_acl_filter_expr("HR", "HR_TEAM")
        assert 'domain == "HR"' in expr
        assert 'visibility == "ALL"' in expr
        assert 'owning_department == "HR_TEAM"' in expr

    def test_부서_누락_시_ALL만(self) -> None:
        expr = build_acl_filter_expr("HR", "")
        assert expr == 'domain == "HR" and (project_id == "" and visibility == "ALL")'
        assert "DEPT_ONLY" not in expr

    def test_GENERAL은_도메인_절이_없다(self) -> None:
        expr = build_acl_filter_expr("GENERAL", "HR_TEAM")
        assert "domain ==" not in expr

    def test_표현식_인젝션_문자는_제거된다(self) -> None:
        """따옴표/or 삽입으로 필터를 무력화할 수 없어야 한다."""
        expr = build_acl_filter_expr("HR", 'X" or visibility != "')
        assert '" or ' not in expr.replace('visibility == "ALL" or', "")
        assert "Xorvisibility" in expr  # 허용 문자만 남는다


# ---------- 프로젝트 격리 (보안 회귀 방지선) ----------

_PROJECT_CANDIDATES = [
    # 전사 공용
    {"chunk_id": "pub", "domain": "HR", "owning_department": "HR_TEAM", "visibility": "ALL"},
    # 프로젝트 A 전용
    {
        "chunk_id": "a1",
        "domain": "HR",
        "owning_department": "HR_TEAM",
        "visibility": "ALL",
        "project_id": "proj-a",
    },
    # 프로젝트 B 전용
    {
        "chunk_id": "b1",
        "domain": "HR",
        "owning_department": "HR_TEAM",
        "visibility": "ALL",
        "project_id": "proj-b",
    },
]


class TestProjectScope:
    def test_일반_채팅은_프로젝트_문서를_못_본다(self) -> None:
        """★ 이 격리가 깨지면 프로젝트 자료가 전 사용자에게 노출된다."""
        result = filter_by_acl(_PROJECT_CANDIDATES, "GENERAL", "HR_TEAM")
        assert {c["chunk_id"] for c in result} == {"pub"}

    def test_프로젝트_채팅은_전사와_자기_프로젝트만_본다(self) -> None:
        result = filter_by_acl(_PROJECT_CANDIDATES, "GENERAL", "HR_TEAM", "proj-a")
        assert {c["chunk_id"] for c in result} == {"pub", "a1"}  # b1은 남의 프로젝트

    def test_project_id_메타데이터가_없는_후보는_전사로_취급된다(self) -> None:
        """구형 BM25 corpus 대비 — 필드가 없으면 전사 공용으로 본다."""
        legacy = [{"chunk_id": "old", "domain": "HR", "visibility": "ALL"}]
        assert len(filter_by_acl(legacy, "GENERAL", "HR_TEAM")) == 1

    def test_dense_표현식도_프로젝트를_한정한다(self) -> None:
        without = build_acl_filter_expr("GENERAL", "HR_TEAM")
        assert 'project_id == ""' in without
        assert " or project_id ==" not in without  # 전사만

        with_project = build_acl_filter_expr("GENERAL", "HR_TEAM", "proj-a")
        assert 'project_id == ""' in with_project  # 전사도 포함
        assert 'project_id == "proj-a"' in with_project

    def test_프로젝트_id_인젝션은_무해한_리터럴이_된다(self) -> None:
        """따옴표·연산자가 제거되어 '존재하지 않는 프로젝트명'으로 떨어진다.
        전사 문서 절(project_id == "")은 그대로 남아 범위가 넓어지지 않는다."""
        expr = build_acl_filter_expr("GENERAL", "HR_TEAM", '" or project_id != "')
        assert "!=" not in expr  # 연산자 삽입 실패
        assert 'project_id == "orproject_id"' in expr  # 허용 문자만 남은 리터럴
        assert 'project_id == ""' in expr  # 전사 절 유지

    def test_허용_문자가_하나도_없으면_프로젝트_절이_안_붙는다(self) -> None:
        """정제 후 비면 전사 문서만 남는다 — 범위가 좁아지는 방향 (fail-closed)."""
        expr = build_acl_filter_expr("GENERAL", "HR_TEAM", '"""')
        assert " or project_id ==" not in expr
        assert 'project_id == ""' in expr

    def test_dense와_bm25가_같은_판정을_한다(self) -> None:
        """두 축의 판정이 어긋나면 한쪽으로만 문서가 샌다."""
        expr = build_acl_filter_expr("GENERAL", "HR_TEAM", "proj-a")
        allowed = {
            c["chunk_id"]
            for c in filter_by_acl(_PROJECT_CANDIDATES, "GENERAL", "HR_TEAM", "proj-a")
        }
        # 표현식이 허용하는 것과 후처리가 허용하는 것이 같아야 한다
        assert ("proj-a" in expr) and ("proj-b" not in expr)
        assert allowed == {"pub", "a1"}

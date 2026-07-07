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
        assert expr == 'domain == "HR" and visibility == "ALL"'
        assert "DEPT_ONLY" not in expr

    def test_GENERAL은_도메인_절이_없다(self) -> None:
        expr = build_acl_filter_expr("GENERAL", "HR_TEAM")
        assert "domain ==" not in expr

    def test_표현식_인젝션_문자는_제거된다(self) -> None:
        """따옴표/or 삽입으로 필터를 무력화할 수 없어야 한다."""
        expr = build_acl_filter_expr("HR", 'X" or visibility != "')
        assert '" or ' not in expr.replace('visibility == "ALL" or', "")
        assert "Xorvisibility" in expr  # 허용 문자만 남는다

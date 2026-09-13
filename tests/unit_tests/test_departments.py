"""질의·적재·두 검색 축의 부서 식별자 검증은 같은 계약이어야 한다."""

import pytest
from fastapi.testclient import TestClient

from ax_rag.api import create_app
from ax_rag.api.normalize import validate_upload
from ax_rag.query_graph.acl import build_acl_filter_expr, filter_by_acl
from ax_rag.shared.departments import validate_department


@pytest.mark.parametrize("bad", ["HR TEAM", "HR.TEAM", 'HR"TEAM', "A" * 33, "가" * 11])
def test_다른_부서로_변환될_수_있는_값은_모든_경계에서_거부한다(bad: str) -> None:
    with pytest.raises(ValueError):
        validate_department(bad)
    with pytest.raises(ValueError):
        validate_upload("규정.md", "HR", "DEPT_ONLY", bad)
    with pytest.raises(ValueError):
        build_acl_filter_expr("HR", bad)
    with pytest.raises(ValueError):
        filter_by_acl([], "HR", bad)
    client = TestClient(create_app())
    assert (
        client.post("/query", json={"question": "휴가 규정", "user_department": bad}).status_code
        == 422
    )
    response = client.post(
        "/documents",
        params={"name": "규정.md", "domain": "HR", "department": bad},
        content="본문".encode(),
    )
    assert response.status_code == 400


def test_허용된_부서는_업로드와_검색에서_동일하게_정규화한다() -> None:
    assert validate_department(" 인사_hr-1 ") == "인사_HR-1"
    _, _, _, department, _ = validate_upload("규정.md", "HR", "DEPT_ONLY", " hr_team ")
    candidate = {"visibility": "DEPT_ONLY", "owning_department": department, "domain": "HR"}
    assert filter_by_acl([candidate], "HR", " hr_team ") == [candidate]
    assert f'owning_department == "{department}"' in build_acl_filter_expr("HR", " hr_team ")
    assert validate_department("") == ""

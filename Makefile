.PHONY: test test-all lint format dev

# 유닛 테스트만 (integration 마커는 pyproject.toml addopts로 기본 제외)
test:
	pytest

# 통합 테스트 포함 (로컬 서비스 필요). -m "" 로 addopts의 마커 필터를 해제
test-all:
	pytest -m ""

lint:
	ruff check .

format:
	ruff format .

# 그래프 시각화 디버깅 (개발 노트북 전용)
dev:
	LANGGRAPH_CLI_NO_ANALYTICS=1 langgraph dev

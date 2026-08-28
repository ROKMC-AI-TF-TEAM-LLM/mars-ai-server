from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="google/gemma-4-26B-A4B-it",  # 원하는 모델 이름
    local_dir="C:/Users/User/Desktop/3차 라이브러리 정리/model/gemma-4-26B-A4B-it"  # 원하는 저장 경로
)
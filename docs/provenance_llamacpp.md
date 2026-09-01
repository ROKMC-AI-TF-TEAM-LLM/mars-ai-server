# provenance_llamacpp.md — llama.cpp 반입물 출처

내부망 반입 승인용 출처 확인서. **llama.cpp 바이너리 2종**을 다룬다.

## 1. 무엇에 쓰는가

개발 노트북(Windows)에서 **LLM 서빙**에 쓴다. vLLM이 Windows를 지원하지 않아
프롬프트·로직 검증용으로 llama.cpp를 대신 쓴다.

- 실행: `serving/start_llm_dev.ps1` → `localhost:8000` (OpenAI 호환 API)
- **운영 L40에는 반입하지 않는다.** L40은 vLLM을 쓴다 (`serving/start_vllm.sh`)

## 2. 반입 파일

| 파일 | 크기 | SHA256 |
|---|---:|---|
| `llama-b9870-bin-win-cuda-12.4-x64.zip` | 253.8 MB | `10CED0B05EB1FDF47981DFE39E820A9465804B9250811F1173D935A22D336D6F` |
| `cudart-llama-bin-win-cuda-12.4-x64.zip` | 373.3 MB | `8C79A9B226DE4B3CACFD1F83D24F962D0773BE79F1E7B75C6AF4DED7E32AE1D6` |

**합계 627.1 MB.** 해시는 SHA-256이며, 반입 직후 대조해 무결성을 확인한다.

## 3. 출처

### 3-1. 배포처

| 항목 | 내용 |
|---|---|
| 프로젝트 | llama.cpp |
| 소유 | ggml-org (Georgi Gerganov 외) |
| 저장소 | `https://github.com/ggml-org/llama.cpp` |
| 배포 경로 | GitHub Releases (공식 릴리스 자산) |
| 라이선스 | **MIT** |

**공식 저장소의 릴리스 자산이다.** 미러·재배포본이 아니다.

### 3-2. 정확한 URL

릴리스 태그 **`b9870`** 으로 고정한다. `latest`를 쓰지 않는다 — 재현성이 깨진다.

```
https://github.com/ggml-org/llama.cpp/releases/download/b9870/llama-b9870-bin-win-cuda-12.4-x64.zip
https://github.com/ggml-org/llama.cpp/releases/download/b9870/cudart-llama-bin-win-cuda-12.4-x64.zip
```

### 3-3. 왜 이 변형인가

파일명이 대상 환경을 그대로 담고 있다.

| 조각 | 뜻 |
|---|---|
| `b9870` | 릴리스 태그 (검증 완료본으로 고정) |
| `bin` | 사전 빌드 바이너리 (소스 컴파일 불필요) |
| `win` | Windows |
| `cuda-12.4` | CUDA 12.4 런타임 |
| `x64` | x86_64 |

`cudart-*`는 **CUDA 런타임 DLL 묶음**이다. NVIDIA CUDA Toolkit을 따로 설치하지
않아도 되도록 llama.cpp가 함께 배포하는 것으로, 없으면 GPU 가속이 동작하지 않는다.
**두 파일은 한 쌍으로 반입한다.**

## 4. 어떻게 받았는가

`scripts/dev_setup.ps1` [4/4] 단계가 자동으로 받는다. 수동으로 받을 때는 동일하게:

```powershell
$tag = "b9870"
curl.exe -sS -L --ssl-no-revoke --fail -o "tools\llama-cuda.zip" `
    "https://github.com/ggml-org/llama.cpp/releases/download/$tag/llama-$tag-bin-win-cuda-12.4-x64.zip"
curl.exe -sS -L --ssl-no-revoke --fail -o "tools\cudart.zip" `
    "https://github.com/ggml-org/llama.cpp/releases/download/$tag/cudart-llama-bin-win-cuda-12.4-x64.zip"
```

- `-L` : GitHub이 자산을 CDN으로 리디렉션하므로 따라가야 한다
- `--fail` : HTTP 오류 시 0바이트 파일을 남기지 않고 실패시킨다
- `--ssl-no-revoke` : 사내망 프록시에서 인증서 폐기 목록 조회가 막힐 때 필요

**받은 뒤 반드시 해시를 대조한다.**

```powershell
Get-FileHash llama-b9870-bin-win-cuda-12.4-x64.zip -Algorithm SHA256
Get-FileHash cudart-llama-bin-win-cuda-12.4-x64.zip -Algorithm SHA256
```

## 5. 내부망 설치

```powershell
Expand-Archive llama-b9870-bin-win-cuda-12.4-x64.zip  -DestinationPath tools\llama.cpp -Force
Expand-Archive cudart-llama-bin-win-cuda-12.4-x64.zip -DestinationPath tools\llama.cpp -Force
```

**두 압축을 같은 폴더에 푼다.** `llama-server.exe`가 CUDA DLL을 같은 디렉터리에서
찾기 때문이다.

확인:

```powershell
tools\llama.cpp\llama-server.exe --version
```

## 6. 네트워크 관련 확인 사항

에어갭 반입 대상이므로 실행 중 외부 통신 여부를 확인해 둔다.

| 항목 | 내용 |
|---|---|
| 모델 자동 다운로드 | **하지 않는다.** `-m` 로 지정한 로컬 GGUF만 읽는다 |
| 텔레메트리 | 없다 |
| 바인딩 | `serving/start_llm_dev.ps1` 이 `--host 127.0.0.1` 로 **루프백에 한정**한다 |
| 업데이트 확인 | 없다 (`--version` 은 빌드된 문자열을 출력할 뿐) |

`llama.cpp`에는 HuggingFace에서 모델을 받는 옵션(`-hf`)이 있으나 **본 프로젝트는
쓰지 않는다.** 기동 스크립트가 로컬 경로만 넘긴다.

## 7. 함께 필요한 것

llama.cpp만으로는 서빙이 되지 않는다. **모델 파일이 별도로 필요하다.**

| 항목 | 비고 |
|---|---|
| `A.X-4.0-Light-Q4_K_M.gguf` | 약 4.1 GB, 별도 출처 문서 |

## 8. 버전을 올릴 때

1. 새 태그의 두 파일을 같은 규칙으로 받는다 (`llama-<tag>-bin-win-cuda-12.4-x64.zip`, `cudart-...`)
2. SHA-256을 기록해 이 문서를 갱신한다
3. `scripts/dev_setup.ps1` 의 `$tag` 를 함께 바꾼다 — **한쪽만 바꾸면 어긋난다**
4. CUDA 변형(12.4)은 노트북 드라이버와 맞춰야 한다. 드라이버가 낮으면 기동 실패

> 태그를 `latest` 로 두지 않는 이유: 반입물과 저장소 기록이 어긋나면 내부망에서
> 무엇이 설치됐는지 추적할 수 없다.

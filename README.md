# Recommendation System

Matrix Factorization 기반 상품-유저 추천 시스템입니다.
Vespa Engine을 활용하여 벡터 유사도 기반의 추천 API 를 제공합니다.

---

## 1. 개발 환경

### System Requirements

| 항목           | 버전/사양 |
| -------------- | --------- |
| Docker         | 28.5.1    |
| Docker Compose | v2.40.3   |
| Python         | 3.12.12   |

### Architecture

```
+-----------------------------------------------------------------------------------------+
|                                  Host Machine (Local)                                   |
|                                                                                         |
|  +---------------------------------------------------+   Port Forwarding                |
|  |                    VS Code                        |   - 6379:6379   (Redis)          |
|  |  +---------------------------------------------+  |   - 8081:8081   (FastAPI)        |
|  |  |       Dev Containers (Extension)            |  |   - 8080:8080   (Vespa Query)    |
|  |  +---------+-----------------------------------+  |   - 19071:19071 (Vespa Admin)    |
|  +------------+--------------------------------------+                                  |
|               | (Connect)                                                               |
|  +------------|----------------------------------------------------------------------+  |
|  |            |               Docker Compose Environment                             |  |
|  |            |                                                                      |  |
|  |  +---------v---------+ Vespa  +-------------------+  HTTP  +-------------------+  |  | HTTP +-------------------+
|  |  |      Develop      | Deploy |       Vespa       |  Req   |        API        |  |  | Req  |      Client       |
|  |  |                  -+--------+>                 <+--------+-                 <+--+--+------+-  (Browser/App)   |
|  |  |  - Python 3.12    | Feed   |  - Vespa Engine   |        |  - FastAPI        |  |  |      |                   |
|  |  |  - Jupyter        |        |                  -+--------+> - pyvespa       -+--+--+------+> GET /recommend   |
|  |  +---------+---------+        +--------+-+--------+  Res   |                   |  |  | Res  |  - /user/{pid}    |
|  |            | (Read)                    | | (Read / Write)  +---------+---------+  |  |      |  - /product/{uid} |
|  |  +---------v-----------------------------v------+                    | (Read)     |  |      |                   |
|  |  |               vespa-logs (Vol)               |          +---------v---------+  |  |      +-------------------+
|  |  +----------------------------------------------+          |       Redis       |  |  |
|  |                         (Read / Write) |                   |                   |  |  |
|  |  +-------------------------------------v--------+          |  - Redis Session  |  |  |
|  |  |               vespa-data (Vol)               |          |                   |  |  |
|  |  +----------------------------------------------+          +--------+-+--------+  |  |
|  |                                                                                   |  |
|  +-----------------------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------------------+
```

### Project Structure

```
recommendation-system/
│
├── .devcontainer/                    # VS Code Dev Container 설정
│   ├── devcontainer.json             # Container 설정 (Python, Packages, Extensions)
│   └── Dockerfile                    # Develop Container 이미지 정의
│
├── api/                              # FastAPI 서버
│   ├── config.py                     # API 서버 환경 설정
│   ├── main.py                       # 앱 진입점
│   ├── vespa_client.py               # API 서버 Vespa 클라이언트
│   ├── redis_client.py               # Session Redis 클라이언트
│   ├── Dockerfile                    # API 서버 Container 이미지 정의
│   ├── requirements.txt              # API 서버 Python 패키지 의존성
│   ├── routers/                      # API 라우터
│   │   ├── health.py
│   │   └── recommendation.py
│   └── services/                     # API 비즈니스 로직
│       └── recommendation.py
│
├── workspace/                        # Develop Container Bind Mount 디렉토리
│   ├── modules/                      # 데이터 처리 및 모델링 유틸리티 모듈
│   │   ├── data_splitters.py
│   │   ├── model_evaluators.py
│   │   └── weight_transformers.py
│   ├── notebooks/                    # Vespa Feed 데이터 생성 및 분석 Notebook
│   │   ├── performance_benchmark_v0.2.ipynb
│   │   ├── train_model_v0.1.ipynb
│   │   └── train_model_v0.2.ipynb
│   └── vespa/                        # Vespa Application Package 및 운영 스크립트
│       ├── create_package.py
│       ├── deploy_vespa.sh           # Vespa 배포 스크립트
│       ├── feed_vespa.sh             # Vespa 데이터 Feed 스크립트
│       └── definitions/              # Vespa 스키마 정의
│           ├── common.py
│           ├── product.py
│           └── user.py
│
├── docker-compose.yml                # Vespa 및 API 서비스 Container 실행 설정
├── docker-compose.dev.yml            # Develop Container 추가 설정
├── pyproject.toml                    # 프로젝트 설정
└── README.md
```

---

## 2. Setup

### 2.1 저장소 클론

```bash
git clone <repository-url>
cd recommendation-system
git checkout v0.3
```

### 2.2 데이터 준비

`workspace/data/fine_data` 디렉토리에 원본 데이터 압축 해제:

```bash
mkdir -p workspace/data/fine_data
tar -xzf fine-data.tar.gz -C workspace/data/fine_data/
```

압축 해제 후 `fine_data` 폴더에 다음 파일이 생성됩니다:

- `fine_users.jsonl`
- `fine_products.jsonl`
- `fine_interactions.jsonl`

### 2.3 환경 변수 설정

`workspace/.env` 파일 생성 :

```bash
cp workspace/.env.template workspace/.env
```

또는,

```bash
touch workspace/.env
```

```env
### Matrix Factorization Hyperparameters
VECTOR_DIMENSION=256

### Data Paths (Develop Container Absolute Paths)
FINE_DATA_DIR=/home/vscode/workspace/data/fine_data
FEED_DATA_DIR=/home/vscode/workspace/data/feed_data
APP_PACKAGE_DIR=/home/vscode/workspace/app_package_out

### Vespa Configuration
VESPA_APP_NAME=recommendation
```

`api/.env` 파일 생성 :

```bash
cp api/.env.template api/.env
```

또는,

```bash
touch api/.env
```

```env
### Vespa Configuration
VESPA_HOST=vespa
VESPA_PORT=8080

### Redis Configuration
REDIS_HOST=redis
REDIS_PORT=6379

### FastAPI Metadata
API_TITLE=Recommendation Service API
API_VERSION=0.1.0
API_DESCRIPTION=API for User & Product Recommendation backed by Vespa

### Recommendation Tuning Parameters
RECOMMEND_HITS=5
RECOMMEND_TARGET_HITS=20

RECOMMEND_ALPHA=0.2
RECOMMEND_BETA=0.8

### Latest Model Version
LATEST_MODEL_VERSION=v0.3
```

### 2.4 Docker 컨테이너 실행

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d
```

컨테이너 상태 확인:

```bash
docker compose ps
```

Vespa Config Server Health Check:

```bash
curl http://localhost:19071/state/v1/health
```

> **Note**: Container API (8080 포트)는 Application Package 배포 이후에 확인 가능합니다. (4.1 참고)

### 2.5 develop 컨테이너 접속

#### Option 1: Docker CLI

```bash
docker exec -it develop bash
```

#### Option 2: VS Code Dev Containers (권장)

1. VS Code에서 **Dev Containers** Extension 설치

   - Extension ID: `ms-vscode-remote.remote-containers`

2. 프로젝트 폴더를 VS Code로 열기

3. Command Palette (`F1` 또는 `Cmd+Shift+P`) 실행 후:

   - `Dev Containers: Rebuild and Reopen in Container` 선택
   - 또는 좌측 하단 `><` 아이콘 클릭 → `Reopen in Container`

4. 컨테이너 빌드 완료 후 자동으로 `/home/vscode/workspace` 디렉토리에 접속됩니다.

> **Note**: Dev Container 사용 시 Python 3.12, Jupyter, pyvespa 등 필요한 패키지 및 Extension 이 자동 설치됩니다.

---

## 3. 벡터 데이터 생성

### 3.1 Jupyter Notebook 실행

Dev Container 접속 후 VS Code에서:

1. `workspace/notebooks/train_model_v0.3.ipynb` 파일 열기
2. 우측 상단 **Select Kernel** 클릭 → Python 3.12 선택
3. 셀 실행 (`Shift+Enter`)

> **Note**: Jupyter Extension이 Dev Container에 자동 설치되어 있어 별도 서버 실행 없이 바로 사용 가능합니다.

### 3.2 모델 학습

`notebooks/train_model_v0.3.ipynb` 노트북을 순차적으로 실행합니다.

**주요 단계:**

| Step   | Description                                   |
| ------ | --------------------------------------------- |
| Step 1 | Set Environment & Load Data                   |
| Step 2 | Data Cleaning & Preprocessing                 |
| Step 3 | ID Mapping                                    |
| Step 4 | Data Splitting (Train/Test Split)             |
| Step 5 | Feature Engineering (Weighting & Aggregation) |
| Step 6 | Sparse Matrix Construction                    |
| Step 7 | Latent Factor Model Training(SVD)             |
| Step 8 | Model Evaluation                              |
| Step 9 | Export to Vespa Feed                          |

**출력 파일** (`/home/vscode/workspace/data/feed_data/`):

#### v0.1 이하

| 파일명                     | 설명      |
| -------------------------- | --------- |
| `vespa_user_feed.jsonl`    | 유저 벡터 |
| `vespa_product_feed.jsonl` | 상품 벡터 |

#### v0.2 이하

| 파일명                          | 설명                     |
| ------------------------------- | ------------------------ |
| `vespa_user_feed.jsonl`    | 유저 메타데이터 (Parent) |
| `vespa_product_feed.jsonl` | 상품 메타데이터 (Parent) |
| `vespa_user_vector_feed.jsonl`         | 유저 벡터 (Child)        |
| `vespa_product_vector_feed.jsonl`      | 상품 벡터 (Child)        |
| `vespa_user_cold_start_feed.jsonl`        | 유저 Cold Start 데이터 (Child) |
| `vespa_product_cold_start_feed.jsonl`     | 상품 Cold Start 데이터 (Child) |

#### v0.3 이상

| 파일명                          | 설명                     |
| ------------------------------- | ------------------------ |
| `vespa_user_feed.jsonl`    | 유저 메타데이터 (Parent) |
| `vespa_product_feed.jsonl` | 상품 메타데이터 (Parent) |
| `vespa_user_vector_feed.jsonl`         | 유저 벡터 (Child)        |
| `vespa_product_vector_feed.jsonl`      | 상품 벡터 (Child)        |
| `vespa_user_segment_feed.jsonl`        | 유저 Segment |

---

## 4. 스키마 생성 및 Vespa Feed

### 4.1 Application Package 배포

Develop Container 머신에서:

```bash
/bin/bash workspace/vespa/deploy_vespa.sh
```

이 스크립트는 다음을 수행합니다:

1. `create_package.py` 실행하여 Vespa 스키마 생성
2. Vespa Config Server에 Vespa cli 를 통한 Application Package 배포

배포 완료 후 Container API 상태 확인 (Develop Container 에서):

```bash
curl http://vespa:8080/state/v1/health
```

### 4.2 데이터 Feed

> **Note**: 데이터 Feed 전에 반드시 Container API health check가 정상인지 확인하세요. Deploy 이후 약 1~2분이 소요됩니다.

```bash
/bin/bash workspace/vespa/feed_vespa.sh
```

이 스크립트는 다음 순서로 데이터를 Feed합니다:

1. `user_data` (Parent) → `user` (Child) → `user_segment`
2. `product_data` (Parent) → `product` (Child)

### 4.3 데이터 확인

```bash
# 문서 수 확인
curl 'http://vespa:8080/search/?yql=select%20*%20from%20user%20where%20true'
curl 'http://vespa:8080/search/?yql=select%20*%20from%20product%20where%20true'
```

---

## 5. FastAPI API 서버

### 5.1 API 서버 실행

API Container 머신에서:
(컨테이너 실행 시 자동으로 실행됩니다.)

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8081
```

### 5.2 API 문서

API 서버 실행 후 Swagger UI에서 API 명세를 확인할 수 있습니다:

- **Swagger UI**: http://localhost:8081/docs
- **ReDoc**: http://localhost:8081/redoc

### 5.3 API 엔드포인트

#### 상품 추천 (User → Product)

사용자 ID를 입력받아 추천 상품 목록을 반환합니다.

```bash
GET /recommend/product/{uid}
```

**요청 예시:**

```bash
curl http://localhost:8081/recommend/product/12345
```

**응답 예시:**

```json
{
  "uid": "1",
  "recommendations": [
    {
      "pid": "37871",
      "name": "CENTRAL-12K",
      "categories": [
        "3",
        "449",
        "334"
      ]
    },
    {
      "pid": "225807",
      "name": "Best Minka-41A2",
      "categories": [
        "234",
        "244"
      ]
    },
    ...
  ]
}
```

#### 타겟 유저 추천 (Product → User)

상품 ID를 입력받아 해당 상품에 관심을 가질 만한 유저 목록을 반환합니다.

```bash
GET /recommend/user/{pid}
```

**요청 예시:**

```bash
curl http://localhost:8081/recommend/user/67890
```

**응답 예시:**

```json
{
  "pid": "1",
  "target_users": [
    {
      "uid": "21239",
      "country": "United States",
      "state": "NY",
      "zipcode": "11236"
    },
    {
      "uid": "24674",
      "country": "Mexico",
      "state": "Mexico",
      "zipcode": "56536"
    },
    ...
  ]
}
```

---

## 참고 자료

- [Vespa Documentation](https://docs.vespa.ai/)
- [Vespa Parent-Child Pattern](https://docs.vespa.ai/en/parent-child.html)
- [Vespa HNSW Index](https://docs.vespa.ai/en/approximate-nn-hnsw.html)
- [pyvespa](https://vespa-engine.github.io/pyvespa/)

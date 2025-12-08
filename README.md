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
+---------------------------------------------------------------------------+
|                         Host Machine (Local)                              |
|                                                                           |
|  [User / Developer]                                                       |
|         |                                                                 |
|         | VS Code Dev Container                                           |
|         v                                                                 |
|  +-------------------------------------------+      +------------------+  |      +------------------+
|  |        Docker Compose Environment         |      |  FastAPI Server  |  |      |      Client      |
|  |                                           |      |     (Local)      |  |      |   (Browser/App)  |
|  |  +-----------------+  +----------------+  | HTTP |                  |  | HTTP |                  |
|  |  |     Develop     |  |     Vespa      |  | Req  |   - Port 8081    |  | Req  |  GET /recommend/ |
|  |  |                 |  |                |<-+------+-- - pyvespa      |<-+------+-- product/{uid}  |
|  |  |  - Python 3.12  |  | - Vespa Engine |  |      |                  |  |      |   user/{pid}     |
|  |  |  - Jupyter      |  |                |--+------+->                |--+------+->                |
|  |  +----+------------+  +----+-----------+  | Res  +------------------+  | Res  +------------------+
|  |       |                    |              |                            |
|  |       | Read/Write         | Indexing     |   Port Forwarding          |
|  |       v                    v              |    - 8080:8080 (Query)     |
|  |  +-------------------------------------+  |    - 19071:19071 (Admin)   |
|  |  |      Shared Volume (shared-fs)      |  |                            |
|  |  |   (Preprocessed JSONL, Feed Data)   |  |                            |
|  |  +-------------------------------------+  |                            |
|  +-------------------------------------------+                            |
+---------------------------------------------------------------------------+
```

### Project Structure

```
recommendation-system/
│
├── .devcontainer/                    # VS Code Dev Container 설정
│   └── devcontainer.json             # 컨테이너 설정 (Python, Packages, Extensions)
│
├── api/                              # FastAPI 서버
│   ├── config.py                     # FastAPI 서버 환경 설정
│   ├── main.py                       # 앱 진입점
│   ├── vespa_client.py               # Vespa 클라이언트
│   ├── routers/                      # API 라우터
│   │   ├── health.py
│   │   └── recommendation.py
│   └── services/                     # API 비즈니스 로직
│       └── recommendation.py
│
├── workspace/                        # Develop 컨테이너 Bind Mount 디렉토리
│   ├── data/                         # 원본 데이터
│   │   ├── fine_users.jsonl
│   │   ├── fine_products.jsonl
│   │   └── fine_interactions.jsonl
│   ├── notebooks/                    # Vespa Feed 데이터 생성 노트북
│   │   └── train_model_v0.x.ipynb
│   └── vespa/                        # Vespa Application Package 정의
│       ├── create_package.py
│       └── definitions/
│           ├── common.py
│           ├── product.py
│           └── user.py
│
├── scripts/                          # 배포/운영 스크립트
│   ├── deploy_vespa.sh               # Vespa Application Package 생성 및 Vespa 배포
│   └── feed_vespa.sh                 # Vespa Data Feed (Parent -> Child)
│
├── docker-compose.yml                # Develop, Vespa 컨테이너 설정
├── environment.yml                   # conda 환경 설정
├── requirements.txt                  # Python 패키지 의존성 (pip)
└── README.md
```

---

## 2. Setup

### 2.1 저장소 클론

```bash
git clone <repository-url>
cd recommendation-system
git checkout v0.1
```

### 2.2 데이터 준비

`workspace/data/` 디렉토리에 원본 데이터 압축 해제:

```bash
tar -xzf fine-data.tar.gz -C workspace/data/
```

압축 해제 후 다음 파일이 생성됩니다:
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
VECTOR_DIMENSION=32
RANDOM_STATE=100

### Data Paths (Develop Container Absolute Paths)
FINE_DATA_DIR=/home/vscode/workspace/data
VESPA_FEED_DATA_DIR=/home/vscode/shared/vespa_feed
APP_PACKAGE_DIR=/home/vscode/shared/app_package_out

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
VESPA_HOST=localhost
VESPA_PORT=8080

### FastAPI Metadata
API_TITLE=Recommendation Service API
API_VERSION=0.1.0
API_DESCRIPTION=API for User & Product Recommendation backed by Vespa

### Recommendation Tuning Parameters
RECOMMEND_HITS=5
RECOMMEND_TARGET_HITS=20
```

### 2.3 Docker 컨테이너 실행

```bash
docker compose up -d
```

컨테이너 상태 확인:

```bash
docker compose ps
```

Vespa Health Check:

```bash
# Config Server 상태 확인
curl http://localhost:19071/state/v1/health

# Container API 상태 확인
curl http://localhost:8080/state/v1/health
```

### 2.4 develop 컨테이너 접속

#### Option 1: Docker CLI

```bash
docker exec -it develop bash
```

#### Option 2: VS Code Dev Containers (권장)

1. VS Code에서 **Dev Containers** Extension 설치

   - Extension ID: `ms-vscode-remote.remote-containers`

2. 프로젝트 폴더를 VS Code로 열기

3. Command Palette (`F1` 또는 `Cmd+Shift+P`) 실행 후:

   - `Dev Containers: Reopen in Container` 선택
   - 또는 좌측 하단 `><` 아이콘 클릭 → `Reopen in Container`

4. 컨테이너 빌드 완료 후 자동으로 `/home/vscode/workspace` 디렉토리에 접속됩니다.

> **Note**: Dev Container 사용 시 Python 3.12, Jupyter, pyvespa 등 필요한 패키지 및 Extension 이 자동 설치됩니다.

---

## 3. 벡터 데이터 생성

### 3.1 Jupyter Notebook 실행

Dev Container 접속 후 VS Code에서:

1. `workspace/notebooks/train_model_v0.x.ipynb` 파일 열기
2. 우측 상단 **Select Kernel** 클릭 → Python 3.12 선택
3. 셀 실행 (`Shift+Enter`)

> **Note**: Jupyter Extension이 Dev Container에 자동 설치되어 있어 별도 서버 실행 없이 바로 사용 가능합니다.

### 3.2 모델 학습

`notebooks/train_model_v0.x.ipynb` 노트북을 순차적으로 실행합니다.

**주요 단계:**

| Step   | Description                                |
| ------ | ------------------------------------------ |
| Step 1 | Set Environment & Load Data                |
| Step 2 | Data Cleaning                              |
| Step 3 | ID Mapping                                 |
| Step 4 | Define Time Decay Function & Apply Weights |
| Step 5 | Matrix Creation & Aggregation              |
| Step 6 | Model Training (TruncatedSVD)              |
| Step 7 | Model Evaluation                           |
| Step 8 | Export to Vespa Feed                       |

**출력 파일** (`/home/vscode/shared/vespa_feed/`):

#### v0.1 이하

| 파일명                     | 설명      |
| -------------------------- | --------- |
| `vespa_user_feed.jsonl`    | 유저 벡터 |
| `vespa_product_feed.jsonl` | 상품 벡터 |

#### v0.2 이상

| 파일명                          | 설명                     |
| ------------------------------- | ------------------------ |
| `vespa_user_data_feed.jsonl`    | 유저 메타데이터 (Parent) |
| `vespa_product_data_feed.jsonl` | 상품 메타데이터 (Parent) |
| `vespa_user_feed.jsonl`         | 유저 벡터 (Child)        |
| `vespa_product_feed.jsonl`      | 상품 벡터 (Child)        |

---

## 4. 스키마 생성 및 Vespa Feed

### 4.1 Application Package 배포

호스트 머신에서:

```bash
/bin/bash scripts/deploy_vespa.sh
```

이 스크립트는 다음을 수행합니다:

1. `create_package.py` 실행하여 Vespa 스키마 생성
2. Vespa Config Server에 Application Package 배포

### 4.2 데이터 Feed

```bash
/bin/bash scripts/feed_vespa.sh
```

이 스크립트는 다음 순서로 데이터를 Feed합니다:

#### v0.1 이하

1. `user`
2. `product`

#### v0.2 이상

1. `user_data` (Parent) → `user` (Child)
2. `product_data` (Parent) → `product` (Child)

### 4.3 데이터 확인

```bash
# 문서 수 확인
curl 'http://localhost:8080/search/?yql=select%20*%20from%20user%20where%20true'
curl 'http://localhost:8080/search/?yql=select%20*%20from%20product%20where%20true'
```

---

## 5. API 작동

### 5.1 Python 환경 설정

호스트 머신 (Local) 에서 Python 환경 설정:

#### Option 1: conda (environment.yml)

```bash
conda env create -f environment.yml
conda activate recommendation-system
```

#### Option 2: pip (requirements.txt)

```bash
pip install -r requirements.txt
```

### 5.2 API 서버 실행

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8081
```

### 5.3 API 문서

API 서버 실행 후 Swagger UI에서 API 명세를 확인할 수 있습니다:

- **Swagger UI**: http://localhost:8081/docs
- **ReDoc**: http://localhost:8081/redoc

### 5.4 API 엔드포인트

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
  "uid": "12345",
  "recommendations": [
    {
      "pid": "67890",
      "name": "PRODUCT-A",
      "categories": ["1", "8", "20"]
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
  "pid": "67890",
  "target_users": [
    {
      "uid": "12345",
      "country": "United States",
      "state": "CA",
      "zipcode": "90210"
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

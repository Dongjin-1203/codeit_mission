# 🔢 MNIST 숫자 인식 Streamlit 서비스

ONNX 모델을 활용한 손글씨 숫자 인식 웹 서비스입니다. 사용자가 웹 인터페이스에서 숫자를 그리면 AI가 실시간으로 인식합니다.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red)
![ONNX](https://img.shields.io/badge/ONNX-Runtime-green)
![Docker](https://img.shields.io/badge/Docker-Supported-blue)

---

## 📋 목차
- [프로젝트 개요](#프로젝트-개요)
- [주요 기능](#주요-기능)
- [디렉토리 구조](#디렉토리-구조)
- [설치 및 실행](#설치-및-실행)
  - [방법 1: Docker 사용](#방법-1-docker-사용)
  - [방법 2: 가상환경 사용](#방법-2-가상환경-사용)
- [코드 설명](#코드-설명)
- [사용 방법](#사용-방법)
- [기술 스택](#기술-스택)

---

## 🎯 프로젝트 개요

이 프로젝트는 GitHub ONNX 모델 저장소의 MNIST 모델을 활용하여 손글씨 숫자를 인식하는 웹 서비스입니다. Streamlit을 사용하여 직관적인 UI를 제공하며, 여러 ONNX 모델을 비교할 수 있는 기능을 포함합니다.

### 프로젝트 목표
- ONNX 모델을 활용한 실시간 숫자 인식
- 사용자 친화적인 웹 인터페이스 구현
- Docker를 통한 손쉬운 배포
- 여러 모델 성능 비교 기능

---

## ✨ 주요 기능

### 1. 손글씨 입력 캔버스
- 마우스/터치로 숫자 그리기
- 실시간 전처리 이미지 표시
- 펜 굵기 및 캔버스 크기 조절

### 2. AI 숫자 인식
- MNIST ONNX 모델 활용 (버전 7, 8, 12)
- 0-9 숫자 예측 및 확률 시각화
- 신뢰도 기반 경고 시스템

### 3. 모델 비교 모드
- 최대 3개 모델 동시 실행
- 예측 결과 및 추론 시간 비교
- 모델별 확률 분포 시각화

### 4. 이미지 히스토리
- 최근 20개 예측 결과 저장
- 이미지, 예측 레이블, 신뢰도, 사용 모델 표시

---

## 📁 디렉토리 구조

```
├── app.py                      # Streamlit 메인 애플리케이션
├── models/
│   └── model_loader.py        # ONNX 모델 다운로드 및 로딩
├── utils/
│   ├── image_processor.py     # 이미지 전처리 (28x28 변환, 정규화)
│   └── inferencer.py          # 모델 추론 및 결과 처리
├── requirements.txt           # Python 패키지 의존성
├── Dockerfile                 # Docker 이미지 빌드 파일
├── .dockerignore             # Docker 빌드 제외 파일 목록
└── README.md                 # 프로젝트 문서 (본 파일)
```

---

## 🚀 설치 및 실행

### 방법 1: Docker 사용 🐳

Docker를 사용하면 환경 설정 없이 바로 실행할 수 있습니다.

#### 1-1. Docker 이미지 빌드
```bash
# 저장소 클론
git clone 
cd mnist-web-service

# Docker 이미지 빌드
docker build -t codeit-mission-17:latest .
```

#### 1-2. 컨테이너 실행
```bash
docker run -p 8501:8501 codeit-mission-17:latest
```

#### 1-3. 브라우저 접속
```
http://localhost:8501
```

#### 1-4. Docker Hub에서 실행 (배포 후)
```bash
# 이미지 다운로드 및 실행
docker pull hambur1203/codeit-mission-17:latest
docker run -p 8501:8501 hambur1203/codeit-mission-17:latest
```

---

### 방법 2: 가상환경 사용 🐍

#### 2-1. venv 사용 (Python 기본 가상환경)

**Windows:**
```bash
# 저장소 클론
git clone 
cd mnist-web-service

# 가상환경 생성
python -m venv venv

# 가상환경 활성화
venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt

# Streamlit 실행
streamlit run app.py
```

**macOS/Linux:**
```bash
# 저장소 클론
git clone 
cd mnist-web-service

# 가상환경 생성
python3 -m venv venv

# 가상환경 활성화
source venv/bin/activate

# 패키지 설치
pip install -r requirements.txt

# Streamlit 실행
streamlit run app.py
```

#### 2-2. conda 사용 (Anaconda/Miniconda)
```bash
# 저장소 클론
git clone 
cd mnist-web-service

# conda 환경 생성
conda create -n mnist-service python=3.11 -y

# 환경 활성화
conda activate mnist-service

# 패키지 설치
pip install -r requirements.txt

# Streamlit 실행
streamlit run app.py
```

#### 가상환경 종료
```bash
# venv
deactivate

# conda
conda deactivate
```

---

## 💻 코드 설명

### 1. app.py - 메인 애플리케이션

**주요 기능:**
- Streamlit 웹 인터페이스 구성
- 사용자 입력 처리 (캔버스 그리기)
- 모델 선택 및 로딩 관리
- 예측 결과 시각화
- 이미지 히스토리 관리

**핵심 구조:**
```python
# 페이지 설정 및 CSS 스타일
st.set_page_config(layout="wide")

# 세션 상태 초기화 (모델, 히스토리, 비교모드)
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

# 사이드바 - 모델 선택 및 설정
with st.sidebar:
    - 모델 선택 (단일/비교 모드)
    - 캔버스 설정
    - 히스토리 관리

# 메인 영역 - 3컬럼 레이아웃
col1: 입력 캔버스
col2: 전처리 이미지
col3: 예측 결과

# 예측 로직
if predict_button:
    1. 이미지 전처리
    2. 모델 추론
    3. 결과 시각화
    4. 히스토리 저장
```

---

### 2. models/model_loader.py - 모델 관리

**주요 기능:**
- ONNX 모델 자동 다운로드
- 모델 캐싱 (Streamlit `@st.cache_resource`)
- 여러 모델 버전 지원 (MNIST-7, 8, 12)

**핵심 함수:**

#### `MODEL_CONFIGS`
```python
MODEL_CONFIGS = {
    'MNIST-7': {
        'url': 'GitHub URL',
        'path': 'models/mnist-7.onnx',
        'description': '모델 설명',
        'opset_version': 7
    },
    # MNIST-8, MNIST-12 동일 구조
}
```

#### `load_onnx_model(model_name)`
- 모델 파일 존재 확인
- 없으면 GitHub에서 자동 다운로드
- ONNX Runtime으로 모델 로드
- 입력/출력 이름 및 shape 반환

#### `download_model(url, save_path)`
- HTTP GET 요청으로 모델 다운로드
- 진행률 표시
- 실패 시 부분 파일 삭제

**반환 구조:**
```python
{
    'session': InferenceSession,
    'input_name': str,
    'output_name': str,
    'model_name': str,
    'model_size_mb': float
}
```

---

### 3. utils/image_processor.py - 이미지 전처리

**주요 기능:**
- Canvas 이미지 → ONNX 입력 형식 변환
- 전처리 결과 시각화
- 입력 검증

**전처리 파이프라인:**
```
Canvas RGBA (280x280)
    ↓
1. RGBA → Grayscale 변환 (알파 채널 고려)
    ↓
2. 28x28 리사이즈 (안티앨리어싱)
    ↓
3. 정규화 [0, 255] → [0, 1]
    ↓
4. 차원 변환 (H, W) → (1, 1, H, W)
    ↓
ONNX 입력: (1, 1, 28, 28) float32
```

#### `preprocess_canvas_image(canvas_data)`
**알고리즘:**
1. **RGBA → Grayscale**
   - RGB 가중 평균: `0.299*R + 0.587*G + 0.114*B`
   - 알파 채널로 배경 제거
   - MNIST 형식에 맞게 반전

2. **리사이징**
   - PIL의 LANCZOS 보간법 사용
   - 안티앨리어싱으로 품질 유지

3. **정규화 및 차원 변환**
   - float32로 변환
   - NCHW 형식으로 reshape

#### `visualize_preprocessed_image(preprocessed_array)`
- (1, 1, 28, 28) → (28, 28) 추출
- [0, 1] → [0, 255] 변환
- PIL Image로 반환

#### `validate_preprocessed_shape(preprocessed_array)`
- Shape, dtype, range 검증
- 오류 시 경고 메시지 표시

---

### 4. utils/inferencer.py - 모델 추론

**주요 기능:**
- ONNX 모델 추론 실행
- Softmax 확률 변환
- 결과 포맷팅 및 시각화

**추론 파이프라인:**
```
전처리 이미지 (1, 1, 28, 28)
    ↓
1. ONNX 모델 추론 → Logits (1, 10)
    ↓
2. Softmax 적용 → Probabilities (10,)
    ↓
3. Argmax → Predicted Label
    ↓
4. 결과 포맷팅
```

#### `predict_digit(model_info, preprocessed_image)`
**핵심 로직:**
```python
# 1. 모델 추론
input_dict = {input_name: preprocessed_image}
outputs = session.run([output_name], input_dict)
logits = outputs[0][0]  # (1, 10) → (10,)

# 2. Softmax 적용
probabilities = softmax(logits)

# 3. 예측 레이블
predicted_label = np.argmax(probabilities)
confidence = np.max(probabilities)
```

**반환값:**
```python
{
    'predicted_label': int,      # 0-9
    'confidence': float,          # 0-1
    'probabilities': dict,        # {0: p0, 1: p1, ...}
    'inference_time': float       # 초 단위
}
```

#### `softmax(logits)`
**수식:**
```
softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
```
- `max(x)` 빼기: 수치 안정성 확보 (overflow 방지)

#### `get_top_k_predictions(probabilities, k=3)`
- 확률 내림차순 정렬
- 상위 K개 반환

#### `format_prediction_result(result)`
- 신뢰도별 이모지 선택 (🎯/✅/⚠️/❓)
- 사람이 읽기 쉬운 형식으로 포맷팅

---

## 📖 사용 방법

### 1. 단일 모델 모드

1. **사이드바에서 모델 선택**
   - MNIST-7, MNIST-8, MNIST-12 중 선택
   - 선택한 모델 정보 확인

2. **캔버스에 숫자 그리기**
   - 마우스/터치로 0-9 숫자 그리기
   - 펜 굵기 및 캔버스 크기 조절 가능

3. **예측하기 버튼 클릭**
   - 전처리 이미지 확인 (28x28)
   - 예측 결과 및 확률 확인
   - Top-3 예측 확인

4. **히스토리 확인**
   - 하단 이미지 저장소에서 과거 예측 확인
   - 최근 20개 저장

---

### 2. 모델 비교 모드

1. **비교 모드 활성화**
   - 사이드바에서 "📊 모델 비교 모드" 체크

2. **비교할 모델 선택**
   - 최대 3개 모델 선택 (multiselect)

3. **숫자 그리기 및 예측**
   - 동일한 입력으로 여러 모델 동시 실행

4. **비교 결과 확인**
   - 비교 테이블: 모델별 예측, 신뢰도, 추론시간
   - 일치 여부 확인
   - 탭으로 모델별 상세 확률 분포 확인

---

## 🛠 기술 스택

### 프론트엔드
- **Streamlit** (1.28+): 웹 UI 프레임워크
- **streamlit-drawable-canvas** (0.9.3+): 손글씨 입력 캔버스

### 백엔드
- **Python** (3.11): 메인 프로그래밍 언어
- **ONNX Runtime** (1.16+): 모델 추론 엔진
- **NumPy** (1.24+): 수치 연산
- **Pillow** (10.0+): 이미지 처리
- **Pandas** (2.0+): 데이터 처리 및 시각화

### 배포
- **Docker**: 컨테이너화
- **Docker Hub**: 이미지 배포

### AI 모델
- **MNIST ONNX 모델** (버전 7, 8, 12)
  - 출처: [ONNX Model Zoo](https://github.com/onnx/models)
  - 입력: (1, 1, 28, 28) float32
  - 출력: (1, 10) logits

---

## 📊 성능 및 특징

### 모델 성능
- **정확도**: ~99% (MNIST 데이터셋 기준)
- **추론 시간**: 2-3ms (CPU 기준)
- **모델 크기**: ~25KB (경량)

### 시스템 요구사항
- **Python**: 3.9 이상
- **메모리**: 최소 512MB
- **디스크**: 최소 100MB (모델 포함)
- **브라우저**: Chrome, Firefox, Safari, Edge (최신 버전)

---

## 🐛 트러블슈팅

### 1. 모델 다운로드 실패
**증상**: "Failed to download model" 에러

**해결책**:
- 인터넷 연결 확인
- GitHub 접근 가능 여부 확인
- `models/` 디렉토리 수동 생성

---

### 2. Docker 컨테이너 접속 불가
**증상**: localhost:8501 접속 안 됨

**해결책**:
```bash
# 포트가 올바르게 매핑되었는지 확인
docker ps

# 8501 포트가 이미 사용중인지 확인
# Windows: netstat -ano | findstr :8501
# Mac/Linux: lsof -i :8501

# 다른 포트로 실행
docker run -p 8502:8501 codeit-mission-17:latest
```

---

### 3. 캔버스가 비어보임
**증상**: 그림을 그렸는데 예측이 안 됨

**해결책**:
- 브라우저 새로고침 (Ctrl+F5 또는 Cmd+Shift+R)
- 캔버스 크기 조절 후 다시 시도
- 다른 브라우저로 접속

---

### 4. 낮은 신뢰도 경고
**증상**: "신뢰도가 낮습니다" 메시지

**개선 방법**:
- 숫자를 더 크고 선명하게 그리기
- 캔버스 중앙에 그리기
- 선이 끊기지 않도록 그리기
- 펜 굵기 증가

---

## 👨‍💻 제작자

**지동진**
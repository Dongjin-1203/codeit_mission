# Mission 15 Report: 학생 성적 예측 모델 및 Docker 기반 협업 시스템

---

## 미션 소개
이번 미션은 두 명의 연구자가 협업하는 아래 시나리오를 참고하여 도커 기반 워크플로우를 설계하고, 
필요한 도커파일을 작성하는 미션입니다.

각 연구자에게 부여된 역할은 다음과 같습니다.

- **연구자 1**: 데이터 전처리, 탐색적 데이터 분석(EDA), 모델링 및 모델 파일 추출
- **연구자 2**: 추출된 모델을 활용한 추론

## 데이터셋
데이터셋의 변수 설명이다.

|변수명|설명|
|---|---|
|Hours Studied|각 학생이 공부에 소요한 총 시간|
|Previous Scores|학생들이 이전 시험에서 얻은 점수|
|Extracurricular Activities|학생이 과외 활동에 참여하는지 여부 (예 또는 아니오)|
|Sleep Hours|학생이 하루 평균 수면 시간|
|Sample Question Papers Practiced|학생이 연습한 모의고사 수|
|Performance Index|**목표변수**. 각 학생의 전반적인 성취도를 나타내는 지표 <br> (성취도 지수는 학생의 학업 성취도를 나타내며, 가장 가까운 정수로 반올림됩니다.<br>지수는 10에서 100까지이며, 값이 높을수록 더 나은 성취도를 나타냅니다.)|

## 협업 시나리오

```text
1. [연구자 1]은 `train.csv` 데이터를 기반으로 Jupyter Notebook(`.ipynb`)에서 
   데이터 전처리, 탐색적 데이터 분석(EDA), 그리고 `scikit-learn`을 사용한 회귀 모델링을 수행한다. 
2. 모델 성능은 RMSE로 평가하며, 최종 모델은 `model.pkl` 파일로 저장한다. 
3. 이후, 전처리 - 모델링 - 모델 저장 과정을 하나의 `.py` 스크립트로 정리한다. 
4. [연구자 1]은 이 작업을 자동화하는 도커 이미지를 구축하여 Docker Hub에 업로드한다.
```

```text
1. [연구자 2]는 [연구자 1]이 생성한 도커 이미지와 별도의 Jupyter Notebook 도커 이미지를 `docker-compose`로 구성한다.
2. [연구자 2]는 [연구자 1]의 도커 컨테이너에서 생성된 `model.pkl` 파일과 컨테이너 내부의 `test.csv` 파일을 활용하여 
   Jupyter Notebook 컨테이너에서 추론을 수행하고, 결과를 `result.csv` 파일로 저장한다. 
3. 전체 추론 과정이 담긴 inference.ipynb 파일을 별도로 저장한다.

(참고: [연구자 2]는 사전에 데이터나 모델 파일을 보유하지 않은 상태이며, 
      [연구자 1]의 Docker Hub 이미지를 통해 필요한 파일을 가져와야 한다.)
```

## 데이터 EDA 결과

### 1. 결측값: 없음

### 2. 데이터셋 상세 정보

**train data**
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 7000 entries, 0 to 6999
Data columns (total 6 columns):
 #   Column                            Non-Null Count  Dtype  
---  ------                            --------------  -----  
 0   Hours Studied                     7000 non-null   int64  
 1   Previous Scores                   7000 non-null   int64  
 2   Extracurricular Activities        7000 non-null   object 
 3   Sleep Hours                       7000 non-null   int64  
 4   Sample Question Papers Practiced  7000 non-null   int64  
 5   Performance Index                 7000 non-null   float64
dtypes: float64(1), int64(4), object(1)
memory usage: 328.3+ KB
```

**test data**
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 3000 entries, 0 to 2999
Data columns (total 5 columns):
 #   Column                            Non-Null Count  Dtype 
---  ------                            --------------  ----- 
 0   Hours Studied                     3000 non-null   int64 
 1   Previous Scores                   3000 non-null   int64 
 2   Extracurricular Activities        3000 non-null   object
 3   Sleep Hours                       3000 non-null   int64 
 4   Sample Question Papers Practiced  3000 non-null   int64 
dtypes: int64(4), object(1)
memory usage: 117.3+ KB
```
### 3. 상관 관계
<img width="551" height="494" alt="image" src="https://github.com/user-attachments/assets/e1fb76b7-998d-4029-bb56-51a102c1d7fe" />

### 4. Column별 데이터 분포
<img width="695" height="316" alt="image" src="https://github.com/user-attachments/assets/cc07fbf6-6859-433e-b53e-38471b1e5f96" />
<img width="695" height="316" alt="image" src="https://github.com/user-attachments/assets/10fc2f62-55ab-4dcf-9a0f-a3b0252fa776" />
<img width="710" height="316" alt="image" src="https://github.com/user-attachments/assets/f455f4c4-310f-41cc-b560-972e5f2c4edd" />
<img width="704" height="316" alt="image" src="https://github.com/user-attachments/assets/4adc3211-9f42-4b56-8fc5-81ecdf2de8e1" />
<img width="749" height="316" alt="image" src="https://github.com/user-attachments/assets/b1fded9f-b154-40de-a0e5-bc007307a914" />
<img width="695" height="316" alt="image" src="https://github.com/user-attachments/assets/2295642e-123c-445d-a36d-8026f238a1a6" />

## 데이터 전처리

1. **변수 Standard Scaled**: 변수마다 범위가 다르기 때문에, 변수의 중요도가 변할 수 있다. 같은 범위로 조정해줄 필요가 있다.
2. **독립, 종속 변수 선언**: Target을 `y`로 하여 종속변수와 독립변수를 정한다.
3. **학습 데이터, 검증 데이터 분할**: test 데이터가 별도로 있기때문에 검증 데이터를 8:2로 분할하였음.

## 모델링
### 모델 선택: Linear Regression

본 프로젝트에서는 **선형 회귀(Linear Regression)** 모델을 선택했습니다.

#### 선택 근거

**1) 데이터 특성 분석 결과**
```python
상관관계 분석:
- Previous Scores ↔ Performance Index: 0.91 (강한 선형 관계)
- Hours Studied ↔ Performance Index: 0.37 (중간 선형 관계)
- 나머지 변수들: 0.05 이하 (약한 관계)
```

타겟 변수와 독립 변수 간 **강한 선형 관계**가 관찰되어 복잡한 비선형 모델이 불필요함을 확인했습니다.

**2) 문제의 단순성**
- 독립변수 5개로 구성된 단순한 구조
- 특징 간 복잡한 상호작용이 없음
- 데이터셋 크기(7,000개)가 적당하여 과적합 위험 낮음

**3) 해석 가능성(Interpretability)**
```python
Standardized Coefficients:
- Previous Scores: 17.57  → 1 표준편차 증가 시 17.57점 상승
- Hours Studied: 7.42     → 1 표준편차 증가 시 7.42점 상승
```
회귀 계수를 통해 각 변수의 영향력을 직관적으로 파악 가능합니다.

#### **선택 논리**

**✅ Linear Regression을 선택한 이유:**

1. **오컴의 면도날 원칙** (Occam's Razor)
   - "간단한 모델이 복잡한 모델보다 우선한다"
   - 선형 관계가 명확한 경우 복잡한 모델은 불필요

2. **계산 효율성**
   ```python
   학습 시간 비교 (7,000개 샘플 기준):
   - Linear Regression: ~0.01초
   - Random Forest: ~2초
   - XGBoost: ~5초
   - Neural Network: ~30초
   ```

3. **배포 및 유지보수**
   - 모델 파일 크기: 1-2KB (pkl)
   - 추론 속도: 실시간 가능
   - Docker 이미지 크기 최소화

4. **과적합 방지**
   - Train/Valid split에서 일반화 성능 우수
   - 규제 없이도 안정적인 성능

## 모델 학습 결과
<img width="783" height="413" alt="image" src="https://github.com/user-attachments/assets/469cee86-0ccf-4c9e-9a34-ab8786a87d27" />
### 1. RMSE (Root Mean Squared Error) 개념

**정의:**
```
RMSE = √(Σ(y_pred - y_true)² / n)
```

**의미:**
- 예측값과 실제값 간의 **평균 오차 크기**
- 값이 작을수록 정확한 모델
- **타겟 변수와 같은 단위**를 가짐 (Performance Index 점수)

**특징:**
- 큰 오차에 더 큰 페널티 부여 (제곱 연산)
- MAE보다 이상치에 민감
- 회귀 문제에서 가장 널리 사용되는 지표

---

### 2. 모델 성능 결과

#### **최종 RMSE 점수**

```python
RMSE: 2.0103
```

---

# Docker 워크플로우

### 연구자1: 모델 학습 환경 구축 및 배포

#### **Step 1: 프로젝트 구조 설정**

```bash
mission-result/researcher1/
├── main.py                    # 모델 학습 스크립트
├── Dockerfile                 # 이미지 빌드 파일
├── requirements.txt           # Python 패키지
└── data/
    ├── mission15_train.csv    # 학습 데이터
    └── mission15_test.csv     # 테스트 데이터
```

#### **Step 2: Python 학습 스크립트 작성**

```python
# main.py - 주요 기능
def load_and_preprocess_data():
    """데이터 로드 및 범주형 변수 인코딩"""
    
def normalize_features():
    """StandardScaler로 정규화"""
    
def train_model():
    """LinearRegression 학습"""
    
def save_model():
    """모델 + scaler를 pkl 파일로 저장"""
    
def copy_test_data():
    """테스트 데이터를 models/로 복사 (연구자2 제공)"""
```

**핵심 포인트:**
- 모델과 전처리기를 딕셔너리로 함께 저장
- 테스트 데이터를 models/ 디렉토리에 복사하여 연구자2에게 제공

#### **Step 3: requirements.txt 작성**

```txt
pandas
scikit-learn
numpy
```

버전을 고정하지 않아 최신 버전 사용 (재현성보다 호환성 우선)

#### **Step 4: Dockerfile 작성**

```dockerfile
# Python 3.11 경량 이미지 사용
FROM python:3.11-slim

# 작업 디렉토리 설정
WORKDIR /app

# 의존성 설치 (캐싱 최적화)
COPY requirements.txt .
RUN pip install -r requirements.txt

# 소스 코드 및 데이터 복사
COPY main.py .
COPY data/ ./data/

# 모델 저장 디렉토리 생성
RUN mkdir -p models

# 컨테이너 실행 시 자동으로 학습
CMD ["python", "main.py"]
```

**설계 의도:**
- `requirements.txt`를 먼저 복사하여 레이어 캐싱 활용
- `models/` 디렉토리는 볼륨으로 마운트될 예정
- `CMD`로 실행 시 자동으로 모델 학습

#### **Step 5: Docker 이미지 빌드**

```bash
# researcher1 디렉토리에서 실행
cd mission-result/researcher1/

# 이미지 빌드
docker build -t hambur1203/codeit-mission15-ml-model:latest .

# 이미지 업로드
docker push hambur1203/codeit-mission15-ml-model:latest
```

### 연구자2: 추론 환경 구성 및 실행

#### **Step 1: 작업 디렉토리 생성 및 컨테이너 실행**

```bash
mkdir mission-result/researcher2
cd mission-result/researcher2

# 컨테이너 실행하여 모델 생성
docker run -v $(pwd)/test-output:/app/models \hambur1203/codeit-mission15-ml-model:latest
```

#### **Step 2: docker-compose.yml 작성**

```yaml
services:
  # 연구자1의 학습 이미지 실행
  training:
    image: hambur1203/codeit-mission15-ml-model:latest
    container_name: ml-training
    volumes:
      - shared-models:/app/models
    command: python main.py

  # Jupyter Notebook 환경
  jupyter:
    image: jupyter/minimal-notebook:latest
    container_name: jupyter-inference
    ports:
      - "8888:8888"
    volumes:
      - shared-models:/home/jovyan/models          # 모델 파일 접근
      - ./inference.ipynb:/home/jovyan/inference.ipynb
      - ./:/home/jovyan/work                       # 결과 저장용
    environment:
      - JUPYTER_ENABLE_LAB=yes
    depends_on:
      - training
    command: start-notebook.sh --NotebookApp.token='' --NotebookApp.password=''

volumes:
  shared-models:  # 두 컨테이너 간 파일 공유용 Named Volume
```

**핵심 메커니즘:**
1. **Named Volume (`shared-models`)**
   - `training` 컨테이너가 생성한 파일을 `jupyter` 컨테이너가 접근
   - Docker 내부 볼륨으로 컨테이너 간 데이터 공유

2. **의존성 관리 (`depends_on`)**
   - `jupyter`가 `training` 다음에 시작
   - 모델 파일이 생성된 후 추론 가능

3. **Bind Mount (`./:/home/jovyan/work`)**
   - 로컬 디렉토리와 컨테이너 디렉토리 연결
   - `result.csv`가 호스트에 저장됨
  
#### **Step 3: inference.ipynb 작성**: 리포지토리 확인

#### **Step 4: 컨테이너 실행**

```bash
# docker-compose 실행
docker-compose up
```

#### **Step 5: Jupyter에서 추론 수행**

```bash
# 브라우저에서 접속
http://localhost:8888/lab
```

**노트북 실행:**
1. 왼쪽 파일 탐색기에서 `inference.ipynb` 열기
2. 셀 순서대로 실행 (Shift + Enter)
3. 마지막 셀에서 `main()` 실행

**실행 결과:**
```
모델 로드 완료
테스트 데이터 로드 완료: 3000개 샘플
예측 결과 (처음 10개):
[91.83294433 45.1396592  84.29368324 65.53197585 47.42518663 
 30.90313336 72.62567883 58.83080941 40.07411392 81.82245791]
예측 결과 저장 완료: work/result.csv
```

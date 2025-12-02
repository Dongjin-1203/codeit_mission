# 스프린트 미션 15: Docker Hub 실습

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

## 모델 학습 결과
<img width="783" height="413" alt="image" src="https://github.com/user-attachments/assets/469cee86-0ccf-4c9e-9a34-ab8786a87d27" />

---

# Docker 설정

# Python 3.11.8 베이스 이미지
FROM python:3.11-slim

# 작업 디렉토리 설정
WORKDIR /app

# requirements.txt 복사 및 패키지 설치
COPY requirements.txt .
RUN pip install -r requirements.txt

# 학습 코드와 데이터 복사
COPY main.py .
COPY data/ ./data/

# models 디렉토리 생성
RUN mkdir -p models

# 학습 실행
CMD ["python", "main.py"]
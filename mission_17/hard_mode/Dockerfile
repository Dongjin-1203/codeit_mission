# Python 3.11 slim 베이스 이미지
FROM python:3.11-slim

# 작업 디렉토리 설정
WORKDIR /app

# pip 업그레이드
RUN pip install --upgrade pip

# 의존성 파일 먼저 복사 (캐싱 최적화)
COPY requirements.txt .

# 패키지 설치 (캐시 미사용으로 용량 절약)
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# Streamlit 포트 노출
EXPOSE 8501

# Streamlit 애플리케이션 실행
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
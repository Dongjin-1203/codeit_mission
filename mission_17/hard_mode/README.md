# 코드잇 스프린트 미션17(심화): 💰 연말정산 RAG 챗봇

Google Gemini와 ChromaDB 기반의 연말정산 상담 RAG(Retrieval-Augmented Generation) 시스템입니다. PDF 문서를 업로드하고 질문하면 문서 내용을 기반으로 정확한 답변을 제공합니다.

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-121212?style=flat&logo=chainlink&logoColor=white)](https://www.langchain.com/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)

---

## 🎯 프로젝트 개요

국세청 연말정산 안내 문서를 기반으로 사용자의 질문에 답변하는 지능형 챗봇 시스템입니다. 
복잡한 세법을 쉽게 설명하는 10년 경력 세무 상담사 페르소나를 적용하여 친절하고 정확한 상담을 제공합니다.

### ✨ 주요 기능

#### 1. **API 키 검증 및 모델 선택**
- Gemini API 키 유효성 자동 검증
- 사용 가능한 모델 목록 자동 조회
- Gemini 1.5 Flash, Pro, 2.0 등 최신 모델 지원
- 권장 모델 표시 (⭐)

#### 2. **PDF 문서 처리**
- 드래그 앤 드롭으로 PDF 업로드
- 자동 페이지 분석 및 텍스트 추출
- 문서 정보 표시 (페이지 수, 총 문자 수)

#### 3. **지능형 문서 청킹**
- RecursiveCharacterTextSplitter 사용
- 청크 크기 조정 (200~2000자)
- 중첩 크기 설정 (0~500자)
- 실시간 설정 검증 및 권장사항 제시
- 청크 미리보기 및 통계 제공

#### 4. **벡터 데이터베이스**
- ChromaDB 기반 영속성 벡터스토어
- 한국어 특화 임베딩 (jhgan/ko-sroberta-multitask)
- 자동 캐싱으로 빠른 재로드
- Windows 파일 잠금 문제 자동 처리

#### 5. **고급 검색 시스템**
- **MMR (Maximal Marginal Relevance)**: 다양성 고려 검색
- **Similarity Search**: 유사도 기반 검색
- 검색 문서 개수 조정 (1~10개)
- Lambda 파라미터로 다양성 조절 (0~1)
- 검색 테스트 기능

#### 6. **RAG 답변 생성**
- LangChain LCEL 기반 RAG 파이프라인
- 페르소나 기반 프롬프트 엔지니어링
- 참조 문서 표시 (페이지 번호, 원문)
- 대화 기록 관리

#### 7. **사용자 인터페이스**
- Streamlit 기반 직관적인 웹 UI
- 단계별 진행 상태 표시
- 실시간 피드백 및 에러 메시지
- 반응형 레이아웃

---

## 📁 프로젝트 구조
```
hard_mode/
├── app.py                          # Streamlit 메인 애플리케이션
├── models/
│   ├── __init__.py
│   └── llm_loader.py              # Gemini 모델 로딩 및 API 관리
├── utils/
│   ├── __init__.py
│   ├── document_loader.py         # PDF 로드 및 청킹
│   ├── vectorstore.py             # ChromaDB 벡터스토어 관리
│   └── retriever.py               # 검색 및 답변 생성
├── chroma_db/                      # 벡터 DB 저장소 (자동 생성)
├── requirements.txt                # Python 의존성
├── Dockerfile                      # Docker 이미지 빌드
├── .dockerignore                   # Docker 빌드 제외 파일
└── README.md                       # 프로젝트 문서
```

---

## 🚀 설치 및 실행

### 📋 사전 준비

1. **Python 3.11 이상** 설치
2. **Gemini API 키** 발급
   - [Google AI Studio](https://makersuite.google.com/app/apikey)에서 무료 발급
3. **연말정산 PDF 문서** 준비
   - 국세청 홈페이지에서 다운로드 가능

---

### 🐳 Docker로 실행 (권장)

#### 1. Docker 이미지 빌드
```bash
docker build -t codeit-mission-17-rag:latest .
```

#### 2. 컨테이너 실행
```bash
docker run -p 8501:8501 hambur1203/codeit-mission-17-rag:latest
```

#### 3. 브라우저에서 접속
```
http://localhost:8501
```

---

### 💻 로컬 환경에서 실행

#### venv 사용
```bash
# 1. 가상환경 생성
python -m venv venv

# 2. 가상환경 활성화
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 3. 의존성 설치
pip install -r requirements.txt

# 4. Streamlit 실행
streamlit run app.py
```

---

## 📚 코드 설명

### 1. `app.py` - 메인 애플리케이션 (~500줄)

Streamlit 기반 메인 UI 및 전체 플로우 관리

**주요 구성:**
```python
# 페이지 설정
st.set_page_config(
    page_title="연말정산 RAG 챗봇",
    page_icon="💰",
    layout="wide"
)

# 세션 상태 관리
- model_loaded: 모델 로드 여부
- llm: 로드된 LLM 객체
- pages: PDF 페이지 리스트
- splits: 분할된 청크
- vectorstore: 벡터스토어 객체
- rag_chain: RAG 체인
- chat_history: 대화 기록
```

**사이드바 구성:**
1. API 설정 (키 입력, 모델 선택)
2. 문서 설정 (PDF 업로드)
3. 청킹 설정 (크기, 중첩 조정)
4. 벡터스토어 (생성/재생성)
5. RAG 설정 (검색 방식, 문서 개수)

**메인 영역:**
1. 진행 상태 체크
2. 청크 미리보기
3. 검색 테스트
4. 챗봇 인터페이스
5. 대화 기록 표시

---

### 2. `models/llm_loader.py` - LLM 관리

Gemini API 연동 및 모델 로딩

**핵심 함수:**

#### `validate_api_key(api_key)`
```python
# API 키 유효성 검증
# 모델 리스트 조회로 검증 (gemini-pro deprecated 문제 해결)
genai.configure(api_key=api_key)
models = list(genai.list_models())
```

#### `get_available_models(api_key)`
```python
# generateContent 지원 모델 필터링
for model in models:
    if 'generateContent' in model.supported_generation_methods:
        available_models.append(model.name)
```

#### `load_gemini_model(_api_key, model_name)`
```python
# @st.cache_resource로 캐싱
llm = ChatGoogleGenerativeAI(
    model=model_name,
    temperature=0,
    google_api_key=_api_key,
    convert_system_message_to_human=True
)
```

**모델 정보:**
- gemini-2.0-flash-exp: 실험적 최신 모델 ⭐
- gemini-1.5-flash: 빠른 응답, 권장 ⭐
- gemini-1.5-pro: 복잡한 작업용

---

### 3. `utils/document_loader.py` - 문서 처리

PDF 로드 및 지능형 청킹

**주요 함수:**

#### `load_pdf_from_upload(uploaded_file)`
```python
# 임시 파일로 저장 후 PyPDFLoader로 로드
with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
    tmp_file.write(uploaded_file.getvalue())
    tmp_file_path = tmp_file.name

loader = PyPDFLoader(tmp_file_path)
pages = loader.load()
```

#### `split_documents(pages, chunk_size, chunk_overlap)`
```python
# RecursiveCharacterTextSplitter 사용
splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    separators=["\n\n", "\n", " ", ""]
)
splits = splitter.split_documents(pages)
```

#### `validate_chunk_settings(chunk_size, chunk_overlap)`
```python
# 설정 검증 및 권장사항
- chunk_size: 100~5000 권장
- chunk_overlap < chunk_size
- overlap_ratio ≤ 20% 권장
```

**반환 구조:**
```python
{
    'total_chunks': int,
    'avg_length': float,
    'min_length': int,
    'max_length': int,
    'samples': [...]
}
```

---

### 4. `utils/vectorstore.py` - 벡터스토어 관리

ChromaDB 기반 벡터 데이터베이스

**핵심 함수:**

#### `load_embedding_model(model_name)`
```python
# @st.cache_resource로 캐싱 (한 번만 로드)
embeddings = HuggingFaceEmbeddings(
    model_name="jhgan/ko-sroberta-multitask",  # 768차원
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
```

#### `create_vectorstore(splits, embeddings)`
```python
# Windows 파일 잠금 문제 해결
# 1. 가비지 컬렉션 실행
# 2. 기존 디렉토리 삭제 시도
# 3. 실패 시 타임스탬프로 새 컬렉션 생성

vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory="./chroma_db",
    collection_name="rag_documents"
)
```

#### `test_vectorstore_search(vectorstore, query, k)`
```python
# 유사도 검색 테스트
results = vectorstore.similarity_search(test_query, k=k)
```

**특징:**
- 자동 영속화 (persist_directory)
- 재시작 후 빠른 로드
- 메타데이터 저장 (페이지 번호 등)

---

### 5. `utils/retriever.py` - 검색 및 답변 생성

RAG 파이프라인 구성

**핵심 함수:**

#### `create_retriever(vectorstore, search_type, k, lambda_mult)`
```python
# MMR 검색
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": k,              # 검색 문서 개수
        "lambda_mult": 0.5   # 0: 다양성, 1: 유사도
    }
)
```

#### `create_prompt_template()`
```python
system_prompt = """당신은 10년 경력의 친절한 세무 상담사입니다.
연말정산에 대해 잘 모르는 일반인들에게 복잡한 세법을 쉽게 설명하는 것이 전문입니다.

역할:
- 제공된 문서를 기반으로 정확한 정보 제공
- 어려운 용어는 쉬운 말로 풀어서 설명
- 예시를 들어 이해하기 쉽게 설명
- 놓치기 쉬운 주의사항도 함께 알려주기

답변 시 주의사항:
- 문서에 없는 내용은 "제공된 자료에는 해당 정보가 없습니다"라고 명확히 안내
- 금액, 날짜 등 숫자는 정확하게 인용
- 친근하지만 전문적인 톤 유지

{context}

질문: {question}
"""
```

#### `create_rag_chain(llm, retriever)`
```python
# LangChain LCEL 체인
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)
```

#### `generate_answer_with_sources(llm, retriever, question)`
```python
# 검색 → 프롬프트 → LLM → 파싱
docs = retriever.invoke(question)        # 문서 검색
messages = prompt.format_messages(...)   # 프롬프트 생성
response = llm.invoke(messages)          # LLM 호출

return {
    'answer': response.content,
    'sources': docs,
    'question': question
}
```

---

## 🎮 사용 방법

### 1️⃣ API 키 설정
1. 사이드바에 Gemini API 키 입력
2. "✅ API 키가 유효합니다" 메시지 확인
3. 사용 가능한 모델 목록 확인
4. 원하는 모델 선택 (⭐ 표시 = 권장)
5. "🚀 모델 로드" 버튼 클릭

### 2️⃣ 문서 업로드
1. PDF 파일 업로드 (드래그 앤 드롭)
2. "📂 문서 로드" 버튼 클릭
3. 페이지 수 및 문자 수 확인

### 3️⃣ 문서 분할
1. 청크 크기 조정 (기본값: 1000자)
2. 중첩 크기 조정 (기본값: 200자)
3. 설정 검증 메시지 확인
4. "✂️ 문서 분할" 버튼 클릭
5. 생성된 청크 통계 및 미리보기 확인

### 4️⃣ 벡터스토어 생성
1. 임베딩 모델 정보 확인
2. "🗄️ 벡터스토어 생성" 버튼 클릭
3. **1~3분 대기** (임베딩 생성 중)
4. "✅ 벡터스토어 생성 완료!" 메시지 확인

### 5️⃣ RAG 시스템 준비
1. 검색 방식 선택:
   - **MMR**: 다양성 고려 (권장)
   - **Similarity**: 유사도만 고려
2. 검색 문서 개수 설정 (1~10, 기본값: 5)
3. Lambda 조정 (MMR 선택 시)
   - 0.0: 다양성 우선
   - 1.0: 유사도 우선
4. "🚀 RAG 시스템 준비" 버튼 클릭

### 6️⃣ 질문하기
1. 질문 입력창에 질문 작성
   - 예: "연말정산 신고 기한은 언제까지인가요?"
2. "💬 질문하기" 버튼 클릭
3. 답변 확인
4. "📚 참조 문서" 펼쳐서 근거 확인

### 7️⃣ 대화 기록 관리
- 이전 질문/답변 자동 저장
- 최신 순으로 표시
- "🗑️ 대화 기록 삭제" 버튼으로 초기화

---

## 🛠️ 기술 스택

### Core Technologies
- **Python 3.11**: 주요 개발 언어
- **Streamlit 1.28+**: 웹 UI 프레임워크
- **LangChain 0.1+**: RAG 파이프라인 구축
- **Google Gemini API**: LLM 백엔드

### RAG Components
- **ChromaDB 0.4+**: 벡터 데이터베이스
- **HuggingFace Embeddings**: 임베딩 모델 로더
- **jhgan/ko-sroberta-multitask**: 한국어 특화 임베딩 (768차원)
- **PyPDF 3.17+**: PDF 문서 파싱

### LangChain Modules
- `langchain-google-genai`: Gemini 통합
- `langchain-chroma`: ChromaDB 통합
- `langchain-huggingface`: HuggingFace 통합
- `langchain-community`: 커뮤니티 도구
- `langchain-core`: 핵심 기능

### Development Tools
- **Docker**: 컨테이너화
- **venv/conda**: 가상환경 관리

---

## ⚡ 성능 및 리소스

### 소요 시간 (연말정산 50페이지 PDF 기준)

| 단계 | 소요 시간 | 비고 |
|------|----------|------|
| 문서 로드 | 5-10초 | PDF 크기에 비례 |
| 문서 분할 | 1-2초 | 청크 개수에 비례 |
| **벡터스토어 생성** | **1-3분** | ⚠️ 가장 오래 걸림 |
| 검색 | <1초 | ChromaDB 빠름 |
| 답변 생성 | 3-5초 | LLM 호출 시간 |

### 메모리 사용량

| 구성 요소 | 메모리 |
|----------|--------|
| 임베딩 모델 | ~500MB |
| 벡터 DB (50페이지) | ~50MB |
| Streamlit 앱 | ~200MB |
| **총합** | **~750MB** |

### 디스크 사용량
```
chroma_db/                 # 50~100MB (문서 크기에 비례)
└── 12c3674e.../          # 컬렉션 데이터
    ├── data_level0.bin    # 벡터 데이터
    ├── header.bin         # 메타데이터
    └── length.bin         # 길이 정보
```

---

## 🐛 트러블슈팅

### API 호출 한도 초과

**에러:**
```
ResourceExhausted: API quota exceeded
```

**해결:**
- 무료 티어: 분당 15회, 일당 1,500회
- [Google AI Studio](https://makersuite.google.com/)에서 할당량 확인
- 유료 플랜 고려

---

**지동진**
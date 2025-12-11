# app.py

import streamlit as st
from models.llm_loader import (
    validate_api_key, 
    get_available_models, 
    get_model_info, 
    load_gemini_model,
    is_model_loaded
)
from utils.document_loader import (
    load_pdf_from_upload,
    split_documents,
    validate_chunk_settings,
    preview_chunks,
    get_document_info
)
from utils.vectorstore import (
    load_embedding_model,
    create_vectorstore,
    test_vectorstore_search,
    get_vectorstore_info,
    delete_vectorstore
)
from utils.retriever import (
    create_retriever,
    create_rag_chain,
    generate_answer_with_sources
)


# ==================== 페이지 설정 ====================
st.set_page_config(
    page_title="연말정산 RAG 챗봇",
    page_icon="💰",
    layout="wide"
)


# ==================== 세션 상태 초기화 ====================
# LLM 관련
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'llm' not in st.session_state:
    st.session_state.llm = None
if 'api_key' not in st.session_state:
    st.session_state.api_key = None
if 'current_model' not in st.session_state:
    st.session_state.current_model = None

# 문서 관련
if 'doc_loaded' not in st.session_state:
    st.session_state.doc_loaded = False
if 'pages' not in st.session_state:
    st.session_state.pages = None
if 'splits_ready' not in st.session_state:
    st.session_state.splits_ready = False
if 'splits' not in st.session_state:
    st.session_state.splits = None

# 벡터스토어 관련
if 'vectorstore_ready' not in st.session_state:
    st.session_state.vectorstore_ready = False
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None

# RAG 관련
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'rag_chain' not in st.session_state:
    st.session_state.rag_chain = None


# ==================== 사이드바 ====================
with st.sidebar:
    # ===== 1. API 설정 =====
    st.markdown("### 🔑 Gemini API 설정")
    
    api_key = st.text_input(
        "API Key",
        type="password",
        help="Google AI Studio에서 발급받은 API 키를 입력하세요",
        value=st.session_state.get('api_key', '')
    )
    
    if api_key:
        # API 키 변경 감지
        if api_key != st.session_state.get('api_key'):
            st.session_state.api_key = api_key
            st.session_state.model_loaded = False
            st.session_state.llm = None
        
        # API 키 검증
        is_valid, message = validate_api_key(api_key)
        
        if is_valid:
            st.success(message)
            
            # 모델 목록 가져오기
            with st.spinner("모델 목록 가져오는 중..."):
                available_models = get_available_models(api_key)
            
            if available_models:
                st.markdown("### 🤖 모델 선택")
                
                # 기본 선택 인덱스
                default_index = 0
                if st.session_state.current_model in available_models:
                    default_index = available_models.index(st.session_state.current_model)
                
                # 모델 선택
                selected_model = st.selectbox(
                    "사용할 모델",
                    options=available_models,
                    index=default_index,
                    format_func=lambda x: f"{'⭐ ' if get_model_info(x)['recommended'] else ''}{x}"
                )
                
                # 모델 정보 표시
                model_info = get_model_info(selected_model)
                st.info(f"{model_info['description']}")
                
                # 모델 변경 감지
                if selected_model != st.session_state.get('current_model'):
                    st.session_state.current_model = selected_model
                    st.session_state.model_loaded = False
                
                # 모델 로드 버튼
                if not st.session_state.model_loaded:
                    if st.button("🚀 모델 로드", type="primary", use_container_width=True):
                        with st.spinner(f"{selected_model} 로딩 중..."):
                            try:
                                llm = load_gemini_model(api_key, selected_model)
                                st.session_state.llm = llm
                                st.session_state.model_loaded = True
                                st.success(f"✅ {selected_model} 로드 완료!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"모델 로드 실패: {e}")
                else:
                    st.success(f"✅ {selected_model} 사용 중")
                    if st.button("🔄 모델 재로드", use_container_width=True):
                        st.session_state.model_loaded = False
                        st.rerun()
            else:
                st.warning("사용 가능한 모델이 없습니다.")
        else:
            st.error(message)
    else:
        st.info("👆 API 키를 입력하여 시작하세요")
    
    st.divider()
    
    # 모델 상태 표시
    if is_model_loaded():
        st.markdown("### 📊 모델 상태")
        st.success("🟢 연결됨")
        st.caption(f"모델: {st.session_state.current_model}")
    
    # ===== 2. 문서 설정 =====
    st.divider()
    st.markdown("### 📄 문서 설정")
    
    # PDF 업로드
    uploaded_file = st.file_uploader(
        "PDF 업로드",
        type=['pdf'],
        help="연말정산 관련 PDF 문서를 업로드하세요"
    )
    
    if uploaded_file:
        if st.button("📂 문서 로드", use_container_width=True):
            with st.spinner("PDF 로딩 중..."):
                pages = load_pdf_from_upload(uploaded_file)
                
                if pages:
                    st.session_state.pages = pages
                    st.session_state.doc_loaded = True
                    
                    # 문서 정보 표시
                    doc_info = get_document_info(pages)
                    st.success(f"✅ {doc_info['total_pages']}페이지 로드 완료")
                    st.info(f"총 {doc_info['total_chars']:,}자")
                    st.rerun()
    
    # ===== 3. 청킹 설정 =====
    if st.session_state.get('doc_loaded', False):
        st.divider()
        st.markdown("### ✂️ 청킹 설정")
        
        # 청크 크기 설정
        chunk_size = st.slider(
            "청크 크기",
            min_value=200,
            max_value=2000,
            value=1000,
            step=100,
            help="각 청크의 최대 문자 수"
        )
        
        # 중첩 크기 설정
        chunk_overlap = st.slider(
            "중첩 크기",
            min_value=0,
            max_value=500,
            value=200,
            step=50,
            help="인접 청크 간 겹치는 문자 수"
        )
        
        # 설정 검증
        is_valid, message = validate_chunk_settings(chunk_size, chunk_overlap)
        if is_valid:
            if "✅" in message:
                st.success(message)
            else:
                st.warning(message)
        else:
            st.error(message)
        
        # 청킹 실행
        if st.button("✂️ 문서 분할", disabled=not is_valid, use_container_width=True):
            with st.spinner("문서 분할 중..."):
                splits = split_documents(
                    st.session_state.pages,
                    chunk_size,
                    chunk_overlap
                )
                
                if splits:
                    st.session_state.splits = splits
                    st.session_state.splits_ready = True
                    
                    # 미리보기
                    preview = preview_chunks(splits, num_preview=3)
                    st.success(f"✅ {preview['total_chunks']}개 청크 생성")
                    st.caption(f"평균 {preview['avg_length']:.0f}자")
                    st.rerun()
    
    # ===== 4. 벡터스토어 =====
    if st.session_state.get('splits_ready', False):
        st.divider()
        st.markdown("### 🗄️ 벡터스토어")
        
        # 임베딩 모델 정보
        st.info("📊 임베딩: jhgan/ko-sroberta-multitask")
        
        # 벡터스토어 생성
        if not st.session_state.get('vectorstore_ready', False):
            if st.button("🗄️ 벡터스토어 생성", type="primary", use_container_width=True):
                with st.spinner("벡터스토어 생성 중... (시간이 걸릴 수 있습니다)"):
                    try:
                        # 임베딩 모델 로드
                        embeddings = load_embedding_model()
                        st.session_state.embeddings = embeddings
                        
                        # 벡터스토어 생성
                        vectorstore = create_vectorstore(
                            st.session_state.splits,
                            embeddings
                        )
                        
                        if vectorstore:
                            st.session_state.vectorstore = vectorstore
                            st.session_state.vectorstore_ready = True
                            
                            st.success("✅ 벡터스토어 생성 완료!")
                            st.rerun()
                    except Exception as e:
                        st.error(f"생성 실패: {e}")
        else:
            # 벡터스토어 정보 표시
            info = get_vectorstore_info(st.session_state.vectorstore)
            if info:
                st.success("✅ 벡터스토어 준비됨")
                st.caption(f"문서: {info['total_documents']}개")
            
            # 재생성 버튼
            if st.button("🔄 벡터스토어 재생성", use_container_width=True):
                st.session_state.vectorstore_ready = False
                st.session_state.vectorstore = None
                st.rerun()
    
    # ===== 5. RAG 설정 =====
    if st.session_state.get('vectorstore_ready', False):
        st.divider()
        st.markdown("### 🔧 RAG 설정")
        
        # 검색 방식
        search_type = st.selectbox(
            "검색 방식",
            ["mmr", "similarity"],
            help="MMR: 다양성 고려, Similarity: 유사도만 고려"
        )
        
        # 검색 문서 개수
        search_k = st.slider("검색 문서 개수", min_value=1, max_value=10, value=5)
        
        if search_type == "mmr":
            lambda_mult = st.slider(
                "Lambda (다양성)",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.1,
                help="0: 다양성 우선, 1: 유사도 우선"
            )
        else:
            lambda_mult = 0.5
        
        # RAG 체인 생성
        if st.button("🚀 RAG 시스템 준비", type="primary", use_container_width=True):
            with st.spinner("RAG 시스템 준비 중..."):
                try:
                    # Retriever 생성
                    retriever = create_retriever(
                        st.session_state.vectorstore,
                        search_type=search_type,
                        k=search_k,
                        lambda_mult=lambda_mult
                    )
                    
                    if retriever:
                        st.session_state.retriever = retriever
                        
                        # RAG 체인 생성
                        rag_chain = create_rag_chain(
                            st.session_state.llm,
                            retriever
                        )
                        
                        if rag_chain:
                            st.session_state.rag_chain = rag_chain
                            st.success("✅ RAG 시스템 준비 완료!")
                            st.rerun()
                except Exception as e:
                    st.error(f"RAG 시스템 준비 실패: {e}")


# ==================== 메인 영역 ====================
st.title("💰 연말정산 RAG 챗봇")
st.markdown("연말정산 관련 질문을 해보세요!")

# 진행 상태 체크
if not is_model_loaded():
    st.warning("⚠️ 먼저 사이드바에서 API 키를 입력하고 모델을 로드해주세요.")
    st.stop()

if not st.session_state.get('doc_loaded', False):
    st.info("📄 사이드바에서 PDF 문서를 업로드해주세요.")
    st.stop()

if not st.session_state.get('splits_ready', False):
    st.info("✂️ 사이드바에서 문서를 분할해주세요.")
    st.stop()

# ===== 청크 미리보기 =====
if not st.session_state.get('vectorstore_ready', False):
    st.markdown("### 📋 청크 미리보기")
    
    preview = preview_chunks(st.session_state.splits, num_preview=5)
    
    # 통계
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("총 청크", preview['total_chunks'])
    with col2:
        st.metric("평균 길이", f"{preview['avg_length']:.0f}자")
    with col3:
        st.metric("최소 길이", f"{preview['min_length']}자")
    with col4:
        st.metric("최대 길이", f"{preview['max_length']}자")
    
    # 샘플 청크 표시
    for sample in preview['samples']:
        with st.expander(f"청크 {sample['index']} ({sample['length']}자)"):
            st.text(sample['content'])
            st.caption(f"메타데이터: {sample['metadata']}")
    
    st.info("🗄️ 사이드바에서 벡터스토어를 생성해주세요.")
    st.stop()

# ===== 검색 테스트 =====
if not st.session_state.get('rag_chain'):
    st.markdown("### 🔍 검색 테스트")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        test_query = st.text_input(
            "테스트 질문",
            value="연말정산 신고 기한은?",
            placeholder="질문을 입력하세요"
        )
    
    with col2:
        test_k = st.number_input("검색 개수", min_value=1, max_value=10, value=3)
    
    if st.button("🔍 검색 테스트", type="primary"):
        with st.spinner("검색 중..."):
            results = test_vectorstore_search(
                st.session_state.vectorstore,
                test_query,
                k=test_k
            )
            
            if results:
                st.success(f"✅ 검색 완료: {len(results)}개 문서")
                
                for i, doc in enumerate(results):
                    with st.expander(f"📄 문서 {i+1} (페이지: {doc.metadata.get('page', 'N/A')})"):
                        st.text(doc.page_content)
                        st.caption(f"소스: {doc.metadata.get('source', 'Unknown')}")
            else:
                st.warning("검색 결과가 없습니다.")
    
    st.warning("⚠️ 사이드바에서 'RAG 시스템 준비' 버튼을 눌러주세요.")
    st.stop()

# ===== RAG 챗봇 인터페이스 =====
st.divider()
st.markdown("### 💬 연말정산 상담")

# 질문 입력
question = st.text_input(
    "질문을 입력하세요",
    placeholder="예: 연말정산 신고 기한은 언제까지인가요?",
    key="question_input"
)

col1, col2 = st.columns([1, 5])

with col1:
    ask_button = st.button("💬 질문하기", type="primary", use_container_width=True)

with col2:
    if st.button("🗑️ 대화 기록 삭제", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# 질문 처리
if ask_button and question:
    with st.spinner("답변 생성 중..."):
        result = generate_answer_with_sources(
            st.session_state.llm,
            st.session_state.retriever,
            question,
            k=5  # 검색 문서 개수
        )
        
        # 대화 기록에 추가
        st.session_state.chat_history.append({
            'question': question,
            'answer': result['answer'],
            'sources': result['sources']
        })
        
        st.rerun()

# 대화 기록 표시
if st.session_state.chat_history:
    st.divider()
    st.markdown("### 📝 대화 기록")
    
    for i, chat in enumerate(reversed(st.session_state.chat_history)):
        with st.container():
            # 질문
            st.markdown(f"**🙋 질문 {len(st.session_state.chat_history) - i}:**")
            st.info(chat['question'])
            
            # 답변
            st.markdown("**🤖 답변:**")
            st.success(chat['answer'])
            
            # 참조 문서
            with st.expander(f"📚 참조 문서 ({len(chat['sources'])}개)"):
                for j, doc in enumerate(chat['sources']):
                    st.markdown(f"**[문서 {j+1}]** (페이지: {doc.metadata.get('page', 'N/A')})")
                    st.text(doc.page_content[:300] + "...")
                    st.divider()
            
            st.markdown("---")
else:
    st.info("👆 질문을 입력하고 '질문하기' 버튼을 눌러주세요.")
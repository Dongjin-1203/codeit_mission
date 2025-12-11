# utils/retriever.py

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import streamlit as st


# ==================== 문서 포맷팅 ====================

def format_docs(docs):
    """
    검색된 문서를 문자열로 포맷팅
    
    Args:
        docs: 검색된 Document 리스트
    
    Returns:
        str: 포맷팅된 문서 문자열
    """
    formatted = []
    for i, doc in enumerate(docs):
        formatted.append(f"[문서 {i+1}]\n{doc.page_content}\n")
    return "\n".join(formatted)


# ==================== Retriever 생성 ====================

def create_retriever(vectorstore, search_type="mmr", k=5, lambda_mult=0.5):
    """
    Retriever 생성
    
    Args:
        vectorstore: 벡터스토어 객체
        search_type: 검색 타입 ("similarity", "mmr")
        k: 검색할 문서 개수
        lambda_mult: MMR의 다양성 파라미터 (0~1, 높을수록 유사도 우선)
    
    Returns:
        VectorStoreRetriever: Retriever 객체
    """
    if not vectorstore:
        st.error("벡터스토어가 없습니다")
        return None
    
    try:
        if search_type == "mmr":
            retriever = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": k,
                    "lambda_mult": lambda_mult
                }
            )
        else:  # similarity
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": k}
            )
        
        print(f"✅ Retriever 생성: {search_type}, k={k}")
        return retriever
        
    except Exception as e:
        st.error(f"Retriever 생성 실패: {e}")
        return None


# ==================== 프롬프트 템플릿 ====================

def create_prompt_template():
    """
    RAG 프롬프트 템플릿 생성 (페르소나 방식)
    
    Returns:
        ChatPromptTemplate: 프롬프트 템플릿
    """
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

아래 문서를 참고하여 질문에 답변해주세요:

{context}

질문: {question}

답변:"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "{question}")
    ])
    
    return prompt


# ==================== RAG 체인 생성 ====================

def create_rag_chain(llm, retriever):
    """
    RAG 체인 생성
    
    Args:
        llm: 언어 모델
        retriever: Retriever 객체
    
    Returns:
        Chain: RAG 체인
    """
    if not llm or not retriever:
        st.error("LLM 또는 Retriever가 없습니다")
        return None
    
    try:
        # 프롬프트 템플릿
        prompt = create_prompt_template()
        
        # RAG 체인 구성
        rag_chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        
        print("✅ RAG 체인 생성 완료")
        return rag_chain
        
    except Exception as e:
        st.error(f"RAG 체인 생성 실패: {e}")
        return None


# ==================== 답변 생성 ====================

def generate_answer(rag_chain, question):
    """
    질문에 대한 답변 생성
    
    Args:
        rag_chain: RAG 체인
        question: 질문
    
    Returns:
        str: 답변
    """
    if not rag_chain:
        return "RAG 체인이 준비되지 않았습니다."
    
    try:
        print(f"🔍 질문: {question}")
        
        # 답변 생성
        answer = rag_chain.invoke(question)
        
        print(f"✅ 답변 생성 완료 ({len(answer)}자)")
        
        return answer
        
    except Exception as e:
        error_message = f"답변 생성 실패: {e}"
        st.error(error_message)
        return error_message


# ==================== 검색된 문서와 함께 답변 생성 ====================

def generate_answer_with_sources(llm, retriever, question, k=5):
    """
    질문에 대한 답변과 참조 문서를 함께 반환
    
    Args:
        llm: 언어 모델
        retriever: Retriever 객체
        question: 질문
        k: 검색할 문서 개수
    
    Returns:
        dict: {
            'answer': 답변,
            'sources': 참조 문서 리스트,
            'question': 질문
        }
    """
    if not llm or not retriever:
        return {
            'answer': "LLM 또는 Retriever가 준비되지 않았습니다.",
            'sources': [],
            'question': question
        }
    
    try:
        print(f"🔍 질문: {question}")
        
        # 1. 관련 문서 검색
        docs = retriever.invoke(question)
        print(f"📄 검색된 문서: {len(docs)}개")
        
        # 2. 프롬프트 생성
        prompt = create_prompt_template()
        context = format_docs(docs)
        
        # 3. LLM 호출
        messages = prompt.format_messages(context=context, question=question)
        response = llm.invoke(messages)
        answer = response.content
        
        print(f"✅ 답변 생성 완료 ({len(answer)}자)")
        
        return {
            'answer': answer,
            'sources': docs,
            'question': question
        }
        
    except Exception as e:
        error_message = f"답변 생성 실패: {e}"
        st.error(error_message)
        return {
            'answer': error_message,
            'sources': [],
            'question': question
        }


# ==================== 스트리밍 답변 생성 ====================

def generate_answer_stream(rag_chain, question):
    """
    질문에 대한 답변을 스트리밍으로 생성
    
    Args:
        rag_chain: RAG 체인
        question: 질문
    
    Yields:
        str: 답변 청크
    """
    if not rag_chain:
        yield "RAG 체인이 준비되지 않았습니다."
        return
    
    try:
        print(f"🔍 질문: {question}")
        
        # 스트리밍 답변 생성
        for chunk in rag_chain.stream(question):
            yield chunk
        
        print("✅ 스트리밍 답변 완료")
        
    except Exception as e:
        error_message = f"답변 생성 실패: {e}"
        st.error(error_message)
        yield error_message


# ==================== 배치 질문 처리 ====================

def generate_batch_answers(rag_chain, questions):
    """
    여러 질문에 대한 답변 일괄 생성
    
    Args:
        rag_chain: RAG 체인
        questions: 질문 리스트
    
    Returns:
        list: 답변 리스트
    """
    if not rag_chain:
        return ["RAG 체인이 준비되지 않았습니다."] * len(questions)
    
    try:
        print(f"🔍 배치 처리: {len(questions)}개 질문")
        
        # 배치 처리
        answers = rag_chain.batch(questions)
        
        print(f"✅ 배치 답변 완료")
        
        return answers
        
    except Exception as e:
        error_message = f"배치 답변 생성 실패: {e}"
        st.error(error_message)
        return [error_message] * len(questions)
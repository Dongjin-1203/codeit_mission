# utils/document_loader.py

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st
import tempfile
import os


# ==================== PDF 로드 ====================

def load_pdf_from_upload(uploaded_file):
    """
    업로드된 PDF 파일 로드
    
    Args:
        uploaded_file: Streamlit UploadedFile 객체
    
    Returns:
        list: Document 객체 리스트
    """
    try:
        # 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # PyPDFLoader로 로드
        loader = PyPDFLoader(tmp_file_path)
        
        # 동기 로드
        pages = loader.load()
        
        print(f"✅ PDF 로드 완료: {len(pages)}페이지")
        
        return pages
        
    except Exception as e:
        st.error(f"PDF 로드 실패: {e}")
        return []
    
    finally:
        # 임시 파일 삭제
        try:
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
        except:
            pass


def load_pdf_from_path(file_path):
    """
    파일 경로에서 PDF 로드
    
    Args:
        file_path: PDF 파일 경로
    
    Returns:
        list: Document 객체 리스트
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
        
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        
        print(f"✅ PDF 로드 완료: {len(pages)}페이지")
        
        return pages
        
    except Exception as e:
        st.error(f"PDF 로드 실패: {e}")
        return []


# ==================== 문서 청킹 ====================

def create_text_splitter(chunk_size=1000, chunk_overlap=200):
    """
    텍스트 분할기 생성
    
    Args:
        chunk_size: 청크 최대 길이 (문자 수)
        chunk_overlap: 인접 청크 간 겹치는 문자 수
    
    Returns:
        RecursiveCharacterTextSplitter: 분할기 객체
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],  # 우선순위대로 시도
        length_function=len  # 문자 수 기준
    )
    
    return splitter


def split_documents(pages, chunk_size=1000, chunk_overlap=200):
    """
    문서를 청크로 분할
    
    Args:
        pages: Document 객체 리스트
        chunk_size: 청크 크기
        chunk_overlap: 중첩 크기
    
    Returns:
        list: 분할된 Document 청크 리스트
    """
    if not pages:
        st.warning("분할할 문서가 없습니다")
        return []
    
    try:
        # Splitter 생성
        splitter = create_text_splitter(chunk_size, chunk_overlap)
        
        # 문서 분할
        splits = splitter.split_documents(pages)
        
        print(f"✅ 문서 분할 완료: {len(splits)}개 청크")
        
        return splits
        
    except Exception as e:
        st.error(f"문서 분할 실패: {e}")
        return []


# ==================== 청킹 설정 검증 ====================

def validate_chunk_settings(chunk_size, chunk_overlap):
    """
    청킹 설정 유효성 검증
    
    Args:
        chunk_size: 청크 크기
        chunk_overlap: 중첩 크기
    
    Returns:
        tuple: (is_valid: bool, message: str)
    """
    # 1. chunk_size 범위 확인
    if chunk_size < 100:
        return (False, "chunk_size는 100 이상이어야 합니다")
    if chunk_size > 5000:
        return (False, "chunk_size는 5000 이하를 권장합니다")
    
    # 2. chunk_overlap 범위 확인
    if chunk_overlap < 0:
        return (False, "chunk_overlap은 0 이상이어야 합니다")
    if chunk_overlap >= chunk_size:
        return (False, "chunk_overlap은 chunk_size보다 작아야 합니다")
    
    # 3. 권장 비율 확인
    overlap_ratio = chunk_overlap / chunk_size
    if overlap_ratio > 0.3:
        return (True, f"⚠️ 중첩 비율이 높습니다 ({overlap_ratio:.1%}). 20% 이하 권장")
    
    return (True, "✅ 설정이 유효합니다")


# ==================== 청크 미리보기 ====================

def preview_chunks(splits, num_preview=3):
    """
    분할된 청크 미리보기
    
    Args:
        splits: 청크 리스트
        num_preview: 미리보기할 청크 수
    
    Returns:
        dict: 미리보기 정보
    """
    if not splits:
        return None
    
    # 통계 정보
    total_chunks = len(splits)
    chunk_lengths = [len(chunk.page_content) for chunk in splits]
    avg_length = sum(chunk_lengths) / total_chunks
    min_length = min(chunk_lengths)
    max_length = max(chunk_lengths)
    
    # 미리보기 청크
    preview_samples = []
    for i in range(min(num_preview, total_chunks)):
        chunk = splits[i]
        preview_samples.append({
            'index': i,
            'length': len(chunk.page_content),
            'content': chunk.page_content[:200] + "..." if len(chunk.page_content) > 200 else chunk.page_content,
            'metadata': chunk.metadata
        })
    
    return {
        'total_chunks': total_chunks,
        'avg_length': avg_length,
        'min_length': min_length,
        'max_length': max_length,
        'samples': preview_samples
    }


# ==================== 문서 정보 추출 ====================

def get_document_info(pages):
    """
    로드된 문서 정보 추출
    
    Args:
        pages: Document 객체 리스트
    
    Returns:
        dict: 문서 정보
    """
    if not pages:
        return None
    
    total_pages = len(pages)
    total_chars = sum(len(page.page_content) for page in pages)
    
    # 첫 페이지 메타데이터
    first_page_metadata = pages[0].metadata if pages else {}
    
    # 첫 페이지 미리보기
    first_page_content = pages[0].page_content if pages else ""
    first_page_preview = first_page_content[:300] + "..." if len(first_page_content) > 300 else first_page_content
    
    return {
        'total_pages': total_pages,
        'total_chars': total_chars,
        'avg_chars_per_page': total_chars / total_pages if total_pages > 0 else 0,
        'source': first_page_metadata.get('source', 'Unknown'),
        'first_page_preview': first_page_preview
    }


# ==================== 청크 통계 ====================

def get_chunk_statistics(splits):
    """
    청크 통계 정보
    
    Args:
        splits: 청크 리스트
    
    Returns:
        dict: 통계 정보
    """
    if not splits:
        return None
    
    chunk_lengths = [len(chunk.page_content) for chunk in splits]
    
    return {
        'total': len(splits),
        'avg_length': sum(chunk_lengths) / len(splits),
        'min_length': min(chunk_lengths),
        'max_length': max(chunk_lengths),
        'median_length': sorted(chunk_lengths)[len(chunk_lengths) // 2]
    }
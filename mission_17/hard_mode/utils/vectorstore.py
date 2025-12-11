from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import streamlit as st
import os
import shutil


# ==================== 임베딩 모델 ====================

@st.cache_resource
def load_embedding_model(model_name="jhgan/ko-sroberta-multitask"):
    """임베딩 모델 로드 (캐싱 적용)"""
    try:
        print(f"🔄 임베딩 모델 로딩: {model_name}")
        
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},  # 'cuda' for GPU
            encode_kwargs={'normalize_embeddings': True}  # 정규화
        )
        
        # 테스트
        test_embedding = embeddings.embed_query("테스트")
        print(f"✅ 임베딩 모델 로드 완료 (차원: {len(test_embedding)})")
        
        return embeddings
        
    except Exception as e:
        st.error(f"임베딩 모델 로드 실패: {e}")
        raise


# ==================== 벡터스토어 생성 ====================

def create_vectorstore(splits, embeddings, persist_directory="./chroma_db", collection_name="rag_documents"):
    """ChromaDB 벡터스토어 생성"""
    if not splits:
        st.error("벡터스토어를 생성할 문서가 없습니다")
        return None
    
    try:
        print(f"🔄 벡터스토어 생성 중... ({len(splits)}개 청크)")
        
        # 기존 디렉토리 삭제 (Windows 안전 버전)
        if os.path.exists(persist_directory):
            try:
                # 가비지 컬렉션 실행
                gc.collect()
                time.sleep(0.5)  # 잠시 대기
                
                shutil.rmtree(persist_directory)
                print(f"기존 벡터스토어 삭제: {persist_directory}")
                time.sleep(0.5)  # 삭제 후 대기
                
            except PermissionError:
                # 삭제 실패 시 새로운 컬렉션 이름 사용
                import random
                collection_name = f"{collection_name}_{random.randint(1000, 9999)}"
                print(f"⚠️ 기존 파일 삭제 불가. 새 컬렉션 생성: {collection_name}")
            except Exception as e:
                print(f"⚠️ 삭제 실패: {e}. 덮어쓰기 시도")
        
        # ChromaDB 생성
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=persist_directory,
            collection_name=collection_name
        )
        
        # 저장된 문서 개수 확인
        try:
            count = vectorstore._collection.count()
            print(f"✅ 벡터스토어 생성 완료: {count}개 문서")
        except:
            print(f"✅ 벡터스토어 생성 완료: {len(splits)}개 청크")
        
        return vectorstore
        
    except Exception as e:
        st.error(f"벡터스토어 생성 실패: {e}")
        return None


# ==================== 벡터스토어 로드 ====================

def load_vectorstore(embeddings, persist_directory="./chroma_db", collection_name="rag_documents"):
    """기존 벡터스토어 로드"""
    try:
        # 디렉토리 존재 확인
        if not os.path.exists(persist_directory):
            print("벡터스토어가 존재하지 않습니다")
            return None
        
        print(f"🔄 벡터스토어 로딩: {persist_directory}")
        
        # ChromaDB 로드
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
            collection_name=collection_name
        )
        
        # 문서 개수 확인
        try:
            count = vectorstore._collection.count()
            
            if count == 0:
                print("벡터스토어가 비어있습니다")
                return None
            
            print(f"✅ 벡터스토어 로드 완료: {count}개 문서")
        except:
            print(f"✅ 벡터스토어 로드 완료")
        
        return vectorstore
        
    except Exception as e:
        st.error(f"벡터스토어 로드 실패: {e}")
        return None


# ==================== 유사도 검색 테스트 ====================

def test_vectorstore_search(vectorstore, test_query="연말정산 신고 기한은?", k=3):
    """벡터스토어 검색 테스트"""
    if not vectorstore:
        return []
    
    try:
        print(f"🔍 검색 테스트: '{test_query}'")
        
        # 유사도 검색
        results = vectorstore.similarity_search(test_query, k=k)
        
        print(f"✅ 검색 완료: {len(results)}개 문서")
        
        # 결과 미리보기
        for i, doc in enumerate(results):
            print(f"[문서 {i+1}] {doc.page_content[:100]}...")
        
        return results
        
    except Exception as e:
        st.error(f"검색 테스트 실패: {e}")
        return []


# ==================== 벡터스토어 정보 ====================

def get_vectorstore_info(vectorstore):
    """벡터스토어 정보 조회"""
    if not vectorstore:
        return None
    
    try:
        # 컬렉션 정보 가져오기
        collection = vectorstore._collection
        
        # 문서 개수
        try:
            count = collection.count()
        except:
            count = 0
        
        # 컬렉션 이름
        try:
            name = collection.name
        except:
            name = 'rag_documents'
        
        return {
            'total_documents': count,
            'collection_name': name
        }
        
    except Exception as e:
        # 에러 발생 시 기본값 반환
        print(f"정보 조회 실패: {e}")
        return {
            'total_documents': 0,
            'collection_name': 'rag_documents'
        }


# ==================== 벡터스토어 삭제 ====================

def delete_vectorstore(persist_directory="./chroma_db"):
    """벡터스토어 삭제"""
    try:
        if os.path.exists(persist_directory):
            shutil.rmtree(persist_directory)
            print(f"✅ 벡터스토어 삭제 완료: {persist_directory}")
            return True
        else:
            print("삭제할 벡터스토어가 없습니다")
            return False
            
    except Exception as e:
        st.error(f"벡터스토어 삭제 실패: {e}")
        return False


# ==================== 벡터스토어 존재 확인 ====================

def vectorstore_exists(persist_directory="./chroma_db"):
    """벡터스토어 존재 여부 확인"""
    return os.path.exists(persist_directory) and os.path.isdir(persist_directory)
import google.generativeai as genai
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI


# ==================== API 키 검증 ====================

def validate_api_key(api_key):
    """
    API 키가 유효한지 검증
    
    Args:
        api_key: Gemini API 키
    
    Returns:
        tuple: (is_valid: bool, message: str)
    """
    if api_key is None or api_key == "":
        return (False, "API 키를 입력해주세요")
    
    try:
        # API 키 설정
        genai.configure(api_key=api_key)
        
        # 모델 리스트를 가져와서 검증
        models = list(genai.list_models())
        
        if not models:
            return (False, "❌ API 키는 유효하지만 사용 가능한 모델이 없습니다")
        
        return (True, "✅ API 키가 유효합니다")
        
    except Exception as e:
        error_message = str(e)
        if "API_KEY_INVALID" in error_message or "invalid" in error_message.lower():
            return (False, "❌ API 키가 유효하지 않습니다")
        elif "403" in error_message:
            return (False, "❌ API 키 접근 권한이 없습니다")
        elif "429" in error_message:
            return (False, "❌ API 호출 한도를 초과했습니다")
        return (False, f"❌ API 키 검증 실패: {error_message}")


# ==================== 사용 가능한 모델 목록 ====================

def get_available_models(api_key):
    """
    사용 가능한 Gemini 모델 목록 가져오기
    
    Args:
        api_key: Gemini API 키
    
    Returns:
        list: 모델 이름 리스트 또는 빈 리스트
    """
    try:
        # API 키 설정
        genai.configure(api_key=api_key)
        
        # 모델 목록 가져오기
        models = genai.list_models()
        
        # generateContent를 지원하는 모델만 필터링
        available_models = []
        for model in models:
            if 'generateContent' in model.supported_generation_methods:
                available_models.append(model.name)
        
        # "models/" 접두사 제거
        clean_models = [m.replace("models/", "") for m in available_models]
        
        # Gemini 모델만 필터링
        gemini_models = [m for m in clean_models if 'gemini' in m.lower()]
        
        print(f"✅ 사용 가능한 모델: {gemini_models}")
        
        return gemini_models if gemini_models else clean_models
        
    except Exception as e:
        st.error(f"모델 목록 가져오기 실패: {e}")
        return []


# ==================== 모델 정보 ====================

def get_model_info(model_name):
    """
    특정 모델의 정보 반환
    
    Args:
        model_name: 모델 이름
    
    Returns:
        dict: 모델 설명 정보
    """
    MODEL_INFO = {
        'gemini-2.0-flash-exp': {
            'description': '🚀 실험적 최신 모델 (2.0), 빠른 응답',
            'recommended': True
        },
        'gemini-1.5-flash': {
            'description': '⚡ 빠른 응답 속도, 효율적 (권장)',
            'recommended': True
        },
        'gemini-1.5-flash-8b': {
            'description': '💨 초고속 응답, 경량 작업',
            'recommended': False
        },
        'gemini-1.5-pro': {
            'description': '🎯 높은 성능, 복잡한 작업에 적합',
            'recommended': False
        },
        'gemini-1.5-flash-latest': {
            'description': '⚡ 최신 Flash 모델',
            'recommended': False
        },
        'gemini-1.5-pro-latest': {
            'description': '🎯 최신 Pro 모델',
            'recommended': False
        }
    }
    
    return MODEL_INFO.get(model_name, {
        'description': '🤖 범용 모델',
        'recommended': False
    })


# ==================== 모델 로딩 ====================

@st.cache_resource
def load_gemini_model(_api_key, model_name, temperature=0):
    """
    Gemini 모델 로드 (캐싱 적용)
    
    Args:
        _api_key: Gemini API 키 (언더스코어로 시작하면 캐싱에서 제외)
        model_name: 사용할 모델 이름
        temperature: 생성 온도 (0~1)
    
    Returns:
        ChatGoogleGenerativeAI: 로드된 모델
    """
    try:
        print(f"🔄 {model_name} 모델 로딩 중...")
        
        # LangChain Gemini 모델 생성
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=_api_key,
            convert_system_message_to_human=True
        )
        
        # 간단한 테스트
        test_response = llm.invoke("안녕")
        
        print(f"✅ {model_name} 모델 로드 완료")
        
        return llm
        
    except Exception as e:
        st.error(f"모델 로드 실패: {e}")
        raise


# ==================== 모델 테스트 ====================

def test_model_connection(llm, test_query="안녕하세요"):
    """
    모델 연결 테스트
    
    Args:
        llm: 로드된 LLM 모델
        test_query: 테스트 질문
    
    Returns:
        tuple: (success: bool, response: str)
    """
    try:
        response = llm.invoke(test_query)
        return (True, response.content)
        
    except Exception as e:
        return (False, str(e))


# ==================== 모델 상태 확인 ====================

def is_model_loaded():
    """
    세션 상태에서 모델 로드 여부 확인
    
    Returns:
        bool: 모델 로드 여부
    """
    return (
        'llm' in st.session_state and 
        st.session_state.llm is not None and
        'model_loaded' in st.session_state and
        st.session_state.model_loaded
    )
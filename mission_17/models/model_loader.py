import streamlit as st
import onnxruntime
import requests
from pathlib import Path

# 모델 설정
MODEL_URL = "https://github.com/onnx/models/raw/main/validated/vision/classification/mnist/model/mnist-8.onnx"
MODEL_DIR = "models"
MODEL_PATH = "models/mnist-8.onnx"


def download_model(url: str, save_path: str):
    """
    GitHub에서 ONNX 모델 다운로드
    
    Args:
        url: 모델 다운로드 URL
        save_path: 저장할 파일 경로
    """
    # 디렉토리 생성
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    try:
        print(f"Downloading model from {url}...")
        response = requests.get(url, stream=True)
        
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"✅ Model downloaded successfully: {save_path}")
        else:
            raise Exception(f"Failed to download model. Status code: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error during download: {e}")
        raise


@st.cache_resource
def load_onnx_model():
    """
    ONNX 모델 로드 (캐싱 적용)
    
    Returns:
        dict: {
            'session': onnxruntime.InferenceSession,
            'input_name': str,
            'output_name': str
        }
    """
    # 모델 파일이 없으면 다운로드
    if not Path(MODEL_PATH).exists():
        print(f"Model not found locally. Downloading...")
        download_model(MODEL_URL, MODEL_PATH)
    else:
        print(f"Using cached model: {MODEL_PATH}")
    
    try:
        # ONNX Runtime 세션 생성
        session = onnxruntime.InferenceSession(MODEL_PATH)
        
        # 입력/출력 이름 추출
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        print(f"✅ Model loaded successfully")
        print(f"   - Input name: {input_name}")
        print(f"   - Output name: {output_name}")
        
        return {
            'session': session,
            'input_name': input_name,
            'output_name': output_name
        }
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise
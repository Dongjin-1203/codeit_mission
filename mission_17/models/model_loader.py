import streamlit as st
import onnxruntime
import requests
from pathlib import Path
import os


# ==================== 모델 설정 ====================
MODEL_CONFIGS = {
    'MNIST-7': {
        'url': 'https://github.com/onnx/models/raw/main/validated/vision/classification/mnist/model/mnist-7.onnx',
        'path': 'models/mnist-7.onnx',
        'description': 'MNIST ONNX 모델 버전 7',
        'opset_version': 7
    },
    'MNIST-8': {
        'url': 'https://github.com/onnx/models/raw/main/validated/vision/classification/mnist/model/mnist-8.onnx',
        'path': 'models/mnist-8.onnx',
        'description': 'MNIST ONNX 모델 버전 8 (기본)',
        'opset_version': 8
    },
    'MNIST-12': {
        'url': 'https://github.com/onnx/models/raw/main/validated/vision/classification/mnist/model/mnist-12.onnx',
        'path': 'models/mnist-12.onnx',
        'description': 'MNIST ONNX 모델 버전 12',
        'opset_version': 12
    }
}


# ==================== 함수 정의 ====================
def get_available_models():
    """사용 가능한 모델 리스트 반환"""
    return list(MODEL_CONFIGS.keys())


def get_model_info(model_name: str):
    """특정 모델의 정보 반환"""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Available models: {get_available_models()}")
    
    return MODEL_CONFIGS[model_name]


def download_model(url: str, save_path: str):
    """GitHub에서 ONNX 모델 다운로드"""
    # 디렉토리 생성
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    try:
        print(f"📥 Downloading model from {url}...")
        response = requests.get(url, stream=True, timeout=60)
        
        if response.status_code == 200:
            total_size = int(response.headers.get('content-length', 0))
            
            with open(save_path, 'wb') as f:
                if total_size == 0:
                    f.write(response.content)
                else:
                    downloaded = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        downloaded += len(chunk)
                        f.write(chunk)
                        # 진행률 표시 (선택사항)
                        progress = (downloaded / total_size) * 100
                        print(f"Progress: {progress:.1f}%", end='\r')
            
            print(f"\n✅ Model downloaded successfully: {save_path}")
        else:
            raise Exception(f"Failed to download model. Status code: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error during download: {e}")
        # 실패 시 부분 다운로드 파일 삭제
        if os.path.exists(save_path):
            os.remove(save_path)
        raise


@st.cache_resource
def load_onnx_model(model_name: str = 'MNIST-8'):
    """ONNX 모델 로드 (캐싱 적용)"""
    # 모델 설정 가져오기
    if model_name not in MODEL_CONFIGS:
        st.error(f"❌ 알 수 없는 모델: {model_name}")
        st.info(f"사용 가능한 모델: {', '.join(get_available_models())}")
        raise ValueError(f"Unknown model: {model_name}")
    
    config = MODEL_CONFIGS[model_name]
    model_url = config['url']
    model_path = config['path']
    
    # 모델 파일이 없으면 다운로드
    if not Path(model_path).exists():
        print(f"📦 Model not found locally. Downloading {model_name}...")
        download_model(model_url, model_path)
    else:
        print(f"✅ Using cached model: {model_path}")
    
    try:
        # ONNX Runtime 세션 생성
        print(f"🔄 Loading {model_name}...")
        session = onnxruntime.InferenceSession(model_path)
        
        # 입력/출력 이름 추출
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        # 입력/출력 shape 정보
        input_shape = session.get_inputs()[0].shape
        output_shape = session.get_outputs()[0].shape
        
        print(f"✅ Model loaded successfully: {model_name}")
        print(f"   - Input: {input_name} {input_shape}")
        print(f"   - Output: {output_name} {output_shape}")
        print(f"   - Opset version: {config['opset_version']}")
        
        # 모델 파일 크기 확인
        model_size = Path(model_path).stat().st_size / (1024 * 1024)  # MB
        print(f"   - Model size: {model_size:.2f} MB")
        
        return {
            'session': session,
            'input_name': input_name,
            'output_name': output_name,
            'model_name': model_name,
            'description': config['description'],
            'opset_version': config['opset_version'],
            'input_shape': input_shape,
            'output_shape': output_shape,
            'model_size_mb': round(model_size, 2)
        }
    
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise

def get_model_file_size(model_name: str):
    """모델 파일 크기 반환 (MB)"""
    config = MODEL_CONFIGS.get(model_name)
    if not config:
        return 0
    
    model_path = Path(config['path'])
    if model_path.exists():
        return model_path.stat().st_size / (1024 * 1024)
    return 0


def is_model_downloaded(model_name: str):
    """모델이 로컬에 다운로드되어 있는지 확인"""
    config = MODEL_CONFIGS.get(model_name)
    if not config:
        return False
    
    return Path(config['path']).exists()
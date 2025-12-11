import numpy as np
from PIL import Image
import streamlit as st

def preprocess_canvas_image(canvas_data):
    """Canvas 이미지를 ONNX 모델 입력 형식으로 전처리"""
    try:
        # Step 1: numpy array로 변환 (canvas_data가 PIL Image일 수도 있음)
        if isinstance(canvas_data, Image.Image):
            image_array = np.array(canvas_data)
        else:
            image_array = canvas_data
        
        # Step 2: RGBA → Grayscale 변환 (알파 채널 고려)
        if image_array.shape[-1] == 4:  # RGBA
            # RGB 채널과 알파 채널 분리
            rgb = image_array[:, :, :3]
            alpha = image_array[:, :, 3]
            
            # 알파 채널을 이용해 배경 제거 (흰 배경 가정)
            # MNIST는 검은 배경에 흰 글씨이므로 반전 필요
            grayscale = np.dot(rgb[..., :3], [0.299, 0.587, 0.114])  # RGB → Grayscale
            
            # 알파가 0인 영역은 배경(흰색=255)으로, 알파가 있는 영역은 검은색(0)으로
            # Canvas에서 그린 부분은 알파가 높음
            grayscale = 255 - (alpha / 255.0 * (255 - grayscale))
            
        elif image_array.shape[-1] == 3:  # RGB
            grayscale = np.dot(image_array[..., :3], [0.299, 0.587, 0.114])
        else:  # Already grayscale
            grayscale = image_array
        
        # Step 3: PIL Image로 변환하여 리사이즈 (안티앨리어싱 적용)
        grayscale_image = Image.fromarray(grayscale.astype(np.uint8))
        resized_image = grayscale_image.resize((28, 28), Image.LANCZOS)
        
        # Step 4: numpy array로 다시 변환
        resized_array = np.array(resized_image)
        
        # Step 5: 정규화 [0, 255] → [0, 1]
        normalized_array = resized_array.astype(np.float32) / 255.0
        
        # Step 6: 차원 변환 (H, W) → (1, 1, H, W)
        # ONNX 모델 입력: (batch_size, channels, height, width)
        preprocessed_array = normalized_array.reshape(1, 1, 28, 28)
        
        return preprocessed_array
    
    except Exception as e:
        st.error(f"이미지 전처리 중 오류 발생: {e}")
        raise


def create_blank_canvas(size=(280, 280)):
    """빈 캔버스 이미지 생성 (디버깅/테스트용)"""
    # 흰색 배경의 RGBA 이미지 생성
    canvas = np.ones((size[1], size[0], 4), dtype=np.uint8) * 255
    canvas[:, :, 3] = 0  # 알파 채널을 0으로 (투명)
    
    return canvas

def visualize_preprocessed_image(preprocessed_array):
    """전처리된 이미지를 Streamlit에 표시"""
    try:
        # (1, 1, 28, 28) → (28, 28) 추출
        image_2d = preprocessed_array[0, 0, :, :]
        
        # [0, 1] → [0, 255] 변환
        image_uint8 = (image_2d * 255).astype(np.uint8)
        
        # PIL Image로 변환
        pil_image = Image.fromarray(image_uint8, mode='L')
        
        return pil_image
    
    except Exception as e:
        st.error(f"이미지 시각화 중 오류 발생: {e}")
        raise

def validate_preprocessed_shape(preprocessed_array):
    """전처리된 이미지의 shape, dtype, range 검증"""
    try:
        # Shape 검증
        expected_shape = (1, 1, 28, 28)
        if preprocessed_array.shape != expected_shape:
            st.warning(f"❌ Shape 불일치: {preprocessed_array.shape} (예상: {expected_shape})")
            return False
        
        # Dtype 검증
        if preprocessed_array.dtype != np.float32:
            st.warning(f"❌ Dtype 불일치: {preprocessed_array.dtype} (예상: float32)")
            return False
        
        # Range 검증
        min_val, max_val = preprocessed_array.min(), preprocessed_array.max()
        if not (0 <= min_val and max_val <= 1):
            st.warning(f"❌ Range 초과: [{min_val:.3f}, {max_val:.3f}] (예상: [0, 1])")
            return False
        
        st.success("✅ 전처리 검증 통과")
        return True
    
    except Exception as e:
        st.error(f"검증 중 오류 발생: {e}")
        return False
import numpy as np
import streamlit as st
import time
from typing import Dict, Tuple, List


def softmax(logits: np.ndarray) -> np.ndarray:
    """Softmax 함수 (수치 안정성 고려)"""
    # 수치 안정성을 위해 최댓값을 빼줌 (overflow 방지)
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / np.sum(exp_logits)


def predict_digit(model_info: Dict, preprocessed_image: np.ndarray) -> Dict:
    """ONNX 모델로 숫자 예측"""
    try:
        # Step 1: 모델 세션 및 입출력 이름 추출
        session = model_info['session']
        input_name = model_info['input_name']
        output_name = model_info['output_name']
        
        # Step 2: 추론 시간 측정 시작
        start_time = time.time()
        
        # Step 3: 모델 추론 실행
        input_dict = {input_name: preprocessed_image}
        outputs = session.run([output_name], input_dict)
        
        # Step 4: Raw output (logits) 추출
        logits = outputs[0][0]  # (1, 10) → (10,)
        
        # Step 5: Softmax 적용하여 확률로 변환
        probabilities = softmax(logits)
        
        # Step 6: 예측 레이블 및 신뢰도 계산
        predicted_label = int(np.argmax(probabilities))
        confidence = float(np.max(probabilities))
        
        # Step 7: 추론 시간 계산
        inference_time = time.time() - start_time
        
        # Step 8: 확률을 dictionary로 변환
        probabilities_dict = {
            str(i): float(probabilities[i]) for i in range(10)
        }
        
        # Step 9: 결과 반환
        result = {
            'predicted_label': predicted_label,
            'confidence': confidence,
            'probabilities': probabilities_dict,
            'inference_time': inference_time
        }
        
        return result
    
    except Exception as e:
        st.error(f"모델 추론 중 오류 발생: {e}")
        raise


def get_top_k_predictions(probabilities: Dict[str, float], k: int = 3) -> List[Tuple[str, float]]:
    """상위 K개 예측 결과 반환"""
    # 확률 기준으로 정렬
    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    return sorted_probs[:k]


def format_prediction_result(result: Dict) -> str:
    """예측 결과를 사람이 읽기 쉬운 문자열로 포맷팅"""
    predicted_label = result['predicted_label']
    confidence = result['confidence']
    inference_time = result['inference_time']
    
    # 신뢰도에 따른 이모지 선택
    if confidence >= 0.9:
        emoji = "🎯"
    elif confidence >= 0.7:
        emoji = "✅"
    elif confidence >= 0.5:
        emoji = "⚠️"
    else:
        emoji = "❓"
    
    # 결과 문자열 생성
    result_str = f"""
    {emoji} **예측 결과: {predicted_label}**
    - 신뢰도: {confidence * 100:.2f}%
    - 추론 시간: {inference_time * 1000:.2f}ms
    """
    
    return result_str.strip()


def display_confidence_warning(confidence: float, threshold: float = 0.5):
    """신뢰도가 낮을 때 경고 메시지 표시"""
    if confidence < threshold:
        st.warning(f"""
        ⚠️ 신뢰도가 낮습니다 ({confidence * 100:.1f}%)
        
        **개선 방법:**
        - 숫자를 더 크고 선명하게 그려주세요
        - 캔버스 중앙에 그려주세요
        - 선이 끊기지 않도록 그려주세요
        """)


def validate_model_output(logits: np.ndarray) -> bool:
    """모델 출력 유효성 검증"""
    try:
        # Shape 검증
        if logits.shape != (10,):
            st.warning(f"❌ Output shape 불일치: {logits.shape} (예상: (10,))")
            return False
        
        # NaN/Inf 체크
        if np.isnan(logits).any() or np.isinf(logits).any():
            st.warning("❌ 출력값에 NaN 또는 Inf가 포함되어 있습니다")
            return False
        
        return True
    
    except Exception as e:
        st.error(f"출력 검증 중 오류 발생: {e}")
        return False


def create_probability_dataframe(probabilities: Dict[str, float]):
    """확률을 Streamlit 차트에 적합한 형식으로 변환"""
    import pandas as pd
    
    # DataFrame 생성
    df = pd.DataFrame({
        '숫자': list(probabilities.keys()),
        '확률': [probabilities[str(i)] * 100 for i in range(10)]  # 백분율로 변환
    })
    
    return df
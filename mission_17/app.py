# app.py

import streamlit as st
import numpy as np
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import pandas as pd

# 커스텀 모듈 import
from models.model_loader import load_onnx_model
from utils.image_processor import preprocess_canvas_image, visualize_preprocessed_image, validate_preprocessed_shape
from utils.inferencer import predict_digit, format_prediction_result, display_confidence_warning, get_top_k_predictions


# ==================== 페이지 설정 ====================
st.set_page_config(
    page_title="손글씨 숫자 인식 서비스",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================== 커스텀 CSS ====================
st.markdown("""
<style>
    .main-title {
        text-align: center;
        color: #1E88E5;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .section-header {
        background: linear-gradient(90deg, #1E88E5, #42A5F5);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background-color: #E3F2FD;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1E88E5;
        margin: 1rem 0;
    }
    .history-item {
        background-color: #F5F5F5;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ==================== 세션 상태 초기화 ====================
if 'history' not in st.session_state:
    st.session_state.history = []

if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

if 'model_info' not in st.session_state:
    st.session_state.model_info = None


# ==================== 사이드바 ====================
with st.sidebar:
    st.markdown("### ⚙️ 설정")
    
    # 캔버스 설정
    st.markdown("#### 캔버스 설정")
    canvas_size = st.slider("캔버스 크기", 200, 400, 280, 20)
    stroke_width = st.slider("펜 굵기", 5, 30, 15, 1)
    
    st.divider()
    
    # 모델 정보
    st.markdown("#### 📊 모델 정보")
    st.info("""
    **MNIST ONNX 모델**
    - 입력: 28x28 흑백 이미지
    - 출력: 0-9 숫자 확률
    - 정확도: ~99%
    """)
    
    st.divider()
    
    # 히스토리 관리
    st.markdown("#### 📝 히스토리 관리")
    if st.button("🗑️ 전체 삭제", use_container_width=True):
        st.session_state.history = []
        st.success("히스토리가 삭제되었습니다!")
        st.rerun()
    
    st.markdown(f"**저장된 이미지: {len(st.session_state.history)}개**")


# ==================== 메인 헤더 ====================
st.markdown('<h1 class="main-title">🔢 손글자 숫자 인식 서비스</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">손으로 숫자를 그려보세요! AI가 인식합니다.</p>', unsafe_allow_html=True)


# ==================== 모델 로드 ====================
if not st.session_state.model_loaded:
    with st.spinner("🔄 ONNX 모델을 로딩중입니다..."):
        try:
            st.session_state.model_info = load_onnx_model()
            st.session_state.model_loaded = True
            st.success("✅ 모델 로드 완료!")
        except Exception as e:
            st.error(f"❌ 모델 로드 실패: {e}")
            st.stop()


# ==================== 메인 영역 ====================
# 상단: 입력 & 전처리 & 결과
col1, col2, col3 = st.columns([2, 1.5, 2])

# -------------------- 컬럼 1: 입력 캔버스 --------------------
with col1:
    st.markdown('<h3 class="section-header">✏️ 1. 숫자 그리기</h3>', unsafe_allow_html=True)
    
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",
        stroke_width=stroke_width,
        stroke_color="#000000",
        background_color="#FFFFFF",
        height=canvas_size,
        width=canvas_size,
        drawing_mode="freedraw",
        key="canvas",
    )
    
    col1_1, col1_2 = st.columns(2)
    with col1_1:
        predict_button = st.button("🎯 예측하기", use_container_width=True, type="primary")
    with col1_2:
        clear_button = st.button("🔄 초기화", use_container_width=True)


# -------------------- 컬럼 2: 전처리 이미지 --------------------
with col2:
    st.markdown('<h3 class="section-header">🔧 2. 전처리 결과</h3>', unsafe_allow_html=True)
    
    preprocessed_placeholder = st.empty()
    with preprocessed_placeholder.container():
        st.info("👈 왼쪽에서 숫자를 그리고 '예측하기'를 눌러주세요")


# -------------------- 컬럼 3: 예측 결과 --------------------
with col3:
    st.markdown('<h3 class="section-header">📊 3. 예측 결과</h3>', unsafe_allow_html=True)
    
    result_placeholder = st.empty()
    with result_placeholder.container():
        st.info("예측 결과가 여기에 표시됩니다")


# ==================== 예측 로직 ====================
if predict_button and canvas_result.image_data is not None:
    try:
        # 캔버스가 비어있는지 확인
        if np.all(canvas_result.image_data[:, :, 3] == 0):
            st.warning("⚠️ 캔버스가 비어있습니다. 숫자를 그려주세요!")
        else:
            # 1. 이미지 전처리
            with st.spinner("🔄 이미지 전처리 중..."):
                canvas_image = canvas_result.image_data
                preprocessed = preprocess_canvas_image(canvas_image)
                
                # 전처리 검증
                validate_preprocessed_shape(preprocessed)
            
            # 2. 전처리 이미지 표시
            with preprocessed_placeholder.container():
                st.image(
                    visualize_preprocessed_image(preprocessed),
                    caption="전처리된 28x28 이미지",
                    use_container_width=True
                )
            
            # 3. 모델 추론
            with st.spinner("🤖 AI가 숫자를 인식중입니다..."):
                result = predict_digit(st.session_state.model_info, preprocessed)
            
            # 4. 결과 표시
            with result_placeholder.container():
                # 예측 결과 박스
                st.markdown(
                    f'<div class="prediction-box">{format_prediction_result(result)}</div>',
                    unsafe_allow_html=True
                )
                
                # 신뢰도 경고
                display_confidence_warning(result['confidence'], threshold=0.6)
                
                # 확률 막대 차트
                st.markdown("#### 📈 각 숫자별 확률")
                prob_df = pd.DataFrame({
                    '숫자': [f"{i}" for i in range(10)],
                    '확률(%)': [result['probabilities'][str(i)] * 100 for i in range(10)]
                })
                st.bar_chart(prob_df.set_index('숫자'))
                
                # Top-3 예측
                st.markdown("#### 🏆 Top-3 예측")
                top3 = get_top_k_predictions(result['probabilities'], k=3)
                for idx, (digit, prob) in enumerate(top3, 1):
                    st.write(f"{idx}. **{digit}** - {prob * 100:.2f}%")
            
            # 5. 히스토리에 저장
            history_item = {
                'image': visualize_preprocessed_image(preprocessed),
                'predicted_label': result['predicted_label'],
                'confidence': result['confidence'],
                'timestamp': pd.Timestamp.now().strftime('%H:%M:%S')
            }
            st.session_state.history.insert(0, history_item)  # 최신 항목을 맨 앞에
            
            # 히스토리 최대 20개로 제한
            if len(st.session_state.history) > 20:
                st.session_state.history = st.session_state.history[:20]
    
    except Exception as e:
        st.error(f"❌ 오류가 발생했습니다: {e}")


# ==================== 이미지 저장소 ====================
st.divider()
st.markdown('<h3 class="section-header">💾 4. 이미지 저장소</h3>', unsafe_allow_html=True)

if len(st.session_state.history) == 0:
    st.info("아직 저장된 이미지가 없습니다. 숫자를 그려서 예측해보세요!")
else:
    # 그리드 레이아웃 (5개씩)
    cols_per_row = 5
    for i in range(0, len(st.session_state.history), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            idx = i + j
            if idx < len(st.session_state.history):
                with col:
                    item = st.session_state.history[idx]
                    
                    # 이미지 표시
                    st.image(item['image'], use_container_width=True)
                    
                    # 예측 정보
                    confidence_emoji = "🎯" if item['confidence'] > 0.9 else "✅" if item['confidence'] > 0.7 else "⚠️"
                    st.markdown(
                        f"""<div class="history-item">
                        {confidence_emoji} <b>{item['predicted_label']}</b><br>
                        {item['confidence']*100:.1f}%<br>
                        <small>{item['timestamp']}</small>
                        </div>""",
                        unsafe_allow_html=True
                    )


# ==================== 푸터 ====================
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>🤖 MNIST ONNX 모델 기반 숫자 인식 서비스</p>
    <p><small>Powered by Streamlit & ONNX Runtime</small></p>
</div>
""", unsafe_allow_html=True)
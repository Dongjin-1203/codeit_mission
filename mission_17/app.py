# app.py

import streamlit as st
import numpy as np
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import pandas as pd
import time

# 커스텀 모듈 import
from models.model_loader import load_onnx_model, get_available_models, get_model_info, is_model_downloaded
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
    .model-info-box {
        background-color: #F0F8FF;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #1E88E5;
        margin: 1rem 0;
    }
    .comparison-box {
        background-color: #FFF3E0;
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #FF9800;
        margin: 1rem 0;
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

if 'current_model' not in st.session_state:
    st.session_state.current_model = 'MNIST-8'

if 'compare_mode' not in st.session_state:
    st.session_state.compare_mode = False

if 'loaded_models' not in st.session_state:
    st.session_state.loaded_models = {}


# ==================== 사이드바 ====================
with st.sidebar:
    st.markdown("### 🤖 모델 설정")
    
    # 비교 모드 토글
    compare_mode = st.checkbox(
        "📊 모델 비교 모드",
        value=st.session_state.compare_mode,
        help="여러 모델을 동시에 실행하고 결과를 비교합니다"
    )
    st.session_state.compare_mode = compare_mode
    
    st.divider()
    
    # 사용 가능한 모델 목록
    available_models = get_available_models()
    
    if not compare_mode:
        # ===== 단일 모델 모드 =====
        st.markdown("#### 모델 선택")
        model_choice = st.selectbox(
            "사용할 모델",
            options=available_models,
            index=available_models.index(st.session_state.current_model),
            help="ONNX 모델 버전을 선택하세요"
        )
        
        # 모델 변경 감지
        if model_choice != st.session_state.current_model:
            st.session_state.current_model = model_choice
            st.session_state.model_loaded = False
            st.rerun()
        
        # 선택된 모델 정보 표시
        selected_model_info = get_model_info(model_choice)
        is_downloaded = is_model_downloaded(model_choice)
        download_status = "✅ 다운로드됨" if is_downloaded else "⬇️ 다운로드 필요"
        
        st.markdown(f"""
        <div class="model-info-box">
            <b>📦 {model_choice}</b><br>
            {selected_model_info['description']}<br>
            <small>Opset: {selected_model_info['opset_version']}</small><br>
            <small>{download_status}</small>
        </div>
        """, unsafe_allow_html=True)
        
    else:
        # ===== 비교 모드 =====
        st.markdown("#### 비교할 모델 선택")
        selected_models = st.multiselect(
            "모델 선택 (최대 3개)",
            options=available_models,
            default=[available_models[0], available_models[1]],
            max_selections=3,
            help="비교할 모델들을 선택하세요"
        )
        
        if len(selected_models) == 0:
            st.warning("⚠️ 최소 1개 이상의 모델을 선택하세요")
        
        # 선택된 모델들 정보 표시
        for model_name in selected_models:
            model_info = get_model_info(model_name)
            is_downloaded = is_model_downloaded(model_name)
            status = "✅" if is_downloaded else "⬇️"
            st.markdown(f"""
            <div class="model-info-box">
                {status} <b>{model_name}</b><br>
                <small>{model_info['description']}</small>
            </div>
            """, unsafe_allow_html=True)
    
    st.divider()
    
    # 캔버스 설정
    st.markdown("### ⚙️ 캔버스 설정")
    canvas_size = st.slider("캔버스 크기", 200, 400, 280, 20)
    stroke_width = st.slider("펜 굵기", 5, 30, 15, 1)
    
    st.divider()
    
    # 로드된 모델 정보
    if st.session_state.model_loaded and not compare_mode:
        st.markdown("### 📊 로드된 모델 정보")
        model_data = st.session_state.model_info
        st.info(f"""
        **모델**: {model_data['model_name']}  
        **크기**: {model_data['model_size_mb']} MB  
        **입력**: {model_data['input_shape']}  
        **출력**: {model_data['output_shape']}
        """)
    
    st.divider()
    
    # 히스토리 관리
    st.markdown("### 📝 히스토리 관리")
    if st.button("🗑️ 전체 삭제", use_container_width=True):
        st.session_state.history = []
        st.success("히스토리가 삭제되었습니다!")
        st.rerun()
    
    st.markdown(f"**저장된 이미지: {len(st.session_state.history)}개**")


# ==================== 메인 헤더 ====================
st.markdown('<h1 class="main-title">🔢 손글씨 숫자 인식 서비스</h1>', unsafe_allow_html=True)

if compare_mode:
    st.markdown('<p class="sub-title">📊 모델 비교 모드 - 여러 모델의 성능을 비교해보세요!</p>', unsafe_allow_html=True)
else:
    st.markdown('<p class="sub-title">손으로 숫자를 그려보세요! AI가 인식합니다.</p>', unsafe_allow_html=True)


# ==================== 모델 로드 ====================
if not compare_mode:
    # 단일 모델 모드
    if not st.session_state.model_loaded:
        with st.spinner(f"🔄 {st.session_state.current_model} 모델을 로딩중입니다..."):
            try:
                st.session_state.model_info = load_onnx_model(st.session_state.current_model)
                st.session_state.model_loaded = True
                st.success(f"✅ {st.session_state.current_model} 모델 로드 완료!")
            except Exception as e:
                st.error(f"❌ 모델 로드 실패: {e}")
                st.stop()
else:
    # 비교 모드 - 선택된 모델들 로드
    if len(selected_models) > 0:
        for model_name in selected_models:
            if model_name not in st.session_state.loaded_models:
                with st.spinner(f"🔄 {model_name} 로딩중..."):
                    try:
                        st.session_state.loaded_models[model_name] = load_onnx_model(model_name)
                    except Exception as e:
                        st.error(f"❌ {model_name} 로드 실패: {e}")


# ==================== 메인 영역 ====================
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
                validate_preprocessed_shape(preprocessed)
            
            # 2. 전처리 이미지 표시
            with preprocessed_placeholder.container():
                st.image(
                    visualize_preprocessed_image(preprocessed),
                    caption="전처리된 28x28 이미지",
                    use_container_width=True
                )
            
            if not compare_mode:
                # ===== 단일 모델 모드 =====
                with st.spinner(f"🤖 {st.session_state.current_model}이(가) 숫자를 인식중입니다..."):
                    result = predict_digit(st.session_state.model_info, preprocessed)
                
                with result_placeholder.container():
                    st.markdown(
                        f'<div class="prediction-box">{format_prediction_result(result)}</div>',
                        unsafe_allow_html=True
                    )
                    st.caption(f"🤖 사용된 모델: **{st.session_state.current_model}**")
                    display_confidence_warning(result['confidence'], threshold=0.6)
                    
                    st.markdown("#### 📈 각 숫자별 확률")
                    prob_df = pd.DataFrame({
                        '숫자': [f"{i}" for i in range(10)],
                        '확률(%)': [result['probabilities'][str(i)] * 100 for i in range(10)]
                    })
                    st.bar_chart(prob_df.set_index('숫자'))
                    
                    st.markdown("#### 🏆 Top-3 예측")
                    top3 = get_top_k_predictions(result['probabilities'], k=3)
                    for idx, (digit, prob) in enumerate(top3, 1):
                        st.write(f"{idx}. **{digit}** - {prob * 100:.2f}%")
                
                # 히스토리 저장
                history_item = {
                    'image': visualize_preprocessed_image(preprocessed),
                    'predicted_label': result['predicted_label'],
                    'confidence': result['confidence'],
                    'model_name': st.session_state.current_model,
                    'timestamp': pd.Timestamp.now().strftime('%H:%M:%S')
                }
                st.session_state.history.insert(0, history_item)
                
            else:
                # ===== 비교 모드 =====
                if len(selected_models) == 0:
                    st.warning("⚠️ 비교할 모델을 선택하세요")
                else:
                    with st.spinner(f"🤖 {len(selected_models)}개 모델 비교 중..."):
                        comparison_results = []
                        
                        for model_name in selected_models:
                            model_info = st.session_state.loaded_models[model_name]
                            result = predict_digit(model_info, preprocessed)
                            comparison_results.append({
                                'model': model_name,
                                'prediction': result['predicted_label'],
                                'confidence': result['confidence'],
                                'inference_time': result['inference_time'],
                                'probabilities': result['probabilities']
                            })
                    
                    # 결과 표시
                    with result_placeholder.container():
                        st.markdown('<div class="comparison-box">', unsafe_allow_html=True)
                        st.markdown("### 📊 모델 비교 결과")
                        
                        # 비교 테이블
                        comparison_df = pd.DataFrame([{
                            '모델': r['model'],
                            '예측': r['prediction'],
                            '신뢰도(%)': f"{r['confidence']*100:.2f}",
                            '추론시간(ms)': f"{r['inference_time']*1000:.2f}"
                        } for r in comparison_results])
                        
                        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                        
                        # 일치 여부 확인
                        predictions = [r['prediction'] for r in comparison_results]
                        if len(set(predictions)) == 1:
                            st.success(f"✅ 모든 모델이 **{predictions[0]}** 으로 일치합니다!")
                        else:
                            st.warning(f"⚠️ 모델 예측 불일치: {set(predictions)}")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # 각 모델별 상세 결과
                        st.markdown("### 📈 모델별 확률 분포")
                        
                        tabs = st.tabs([r['model'] for r in comparison_results])
                        for idx, tab in enumerate(tabs):
                            with tab:
                                result = comparison_results[idx]
                                prob_df = pd.DataFrame({
                                    '숫자': [f"{i}" for i in range(10)],
                                    '확률(%)': [result['probabilities'][str(i)] * 100 for i in range(10)]
                                })
                                st.bar_chart(prob_df.set_index('숫자'))
                                
                                top3 = get_top_k_predictions(result['probabilities'], k=3)
                                st.markdown("**Top-3:**")
                                for rank, (digit, prob) in enumerate(top3, 1):
                                    st.write(f"{rank}. **{digit}** - {prob * 100:.2f}%")
            
            # 히스토리 제한
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
    cols_per_row = 5
    for i in range(0, len(st.session_state.history), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            idx = i + j
            if idx < len(st.session_state.history):
                with col:
                    item = st.session_state.history[idx]
                    st.image(item['image'], use_container_width=True)
                    
                    confidence_emoji = "🎯" if item['confidence'] > 0.9 else "✅" if item['confidence'] > 0.7 else "⚠️"
                    st.markdown(
                        f"""<div class="history-item">
                        {confidence_emoji} <b>{item['predicted_label']}</b><br>
                        {item['confidence']*100:.1f}%<br>
                        <small>{item['model_name']}</small><br>
                        <small>{item['timestamp']}</small>
                        </div>""",
                        unsafe_allow_html=True
                    )


# ==================== 푸터 ====================
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>🤖 MNIST ONNX 모델 기반 숫자 인식 서비스</p>
    <p><small>Powered by Streamlit & ONNX Runtime | Multi-Model Comparison Support</small></p>
</div>
""", unsafe_allow_html=True)
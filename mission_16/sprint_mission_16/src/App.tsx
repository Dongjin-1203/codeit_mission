// src/App.tsx

import { useState, useEffect } from 'react';
import * as ort from 'onnxruntime-web';
import { ImageUploader } from './components/ImageUploader';
import { ModelCard } from './components/ModelCard';
import { preprocessImage } from './utils/ImagePreprocess';
import { loadModel, runInference } from './utils/onnxInference';
import { InferenceResult } from './utils/types';

function App() {
  // ========== State 선언 ==========
  // 1. 모델 세션
  const [cnnSession, setCnnSession] = useState<ort.InferenceSession | null>(null);
  const [vitSession, setVitSession] = useState<ort.InferenceSession | null>(null);
  
  // 2. 로딩 상태
  const [isModelLoading, setIsModelLoading] = useState(true);
  
  // 3. 추론 결과
  const [cnnResult, setCnnResult] = useState<InferenceResult | null>(null);
  const [vitResult, setVitResult] = useState<InferenceResult | null>(null);
  
  // 4. 추론 중 상태
  const [isCnnInferring, setIsCnnInferring] = useState(false);
  const [isVitInferring, setIsVitInferring] = useState(false);
  
  // 5. 에러
  const [error, setError] = useState<string | null>(null);

  // ========== 모델 로드 (컴포넌트 마운트 시) ==========
  useEffect(() => {
    async function loadModels() {
      // 구현
      try{
        setIsModelLoading(true);

        // 두 모델 동시 로드
        const [cnn, vit] = await Promise.all([
            loadModel('/models/mnist_cnn.onnx'),
            loadModel('/models/mission_16_ViT_v1_epoch_10__merged.onnx')
        ]);

        setCnnSession(cnn);
        setVitSession(vit);
        setIsModelLoading(false);
      } catch (err) {
        setError(`모델 로드 실패: ${err}`);
        setIsModelLoading(false);
      }
    }
    loadModels();
  }, []);   // 컴포넌트 마운트 시 1번만 실행

  // ========== 이미지 업로드 핸들러 ==========
  const handleImageUpload = async (file: File) => {
    // 구현
    try {
        // 1. 이미지 전처리
        const inputData = await preprocessImage(file);

        // 2. CNN 추론
        if (cnnSession) {
            setIsCnnInferring(true);
            const result = await runInference(cnnSession, inputData);
            setCnnResult(result);
            setIsCnnInferring(false);
        }

        // 3. CNN 추론
        if (vitSession) {
            setIsVitInferring(true);
            const result = await runInference(vitSession, inputData);
            setVitResult(result);
            setIsVitInferring(false);
        }
    } catch (err) {
        setError(`추론 실패: ${err}`);
        setIsCnnInferring(false);
        setIsVitInferring(false);
    }
  };

  // ========== UI 렌더링 ==========
  return (
    <div style={{ padding: '20px', maxWidth: '1200px', margin: '0 auto' }}>
      {/* 헤더 */}
      <h1 style={{ textAlign: 'center', marginBottom: '30px' }}>
        MNIST 모델 비교 - CNN vs ViT
      </h1>

      {/* 모델 로딩 중 */}
      {isModelLoading && (
        <div style={{ textAlign: 'center', padding: '20px' }}>
          <p>⏳ 모델 로드 중...</p>
        </div>
      )}

      {/* 에러 표시 */}
      {error && (
        <div style={{ 
          backgroundColor: '#fee', 
          color: '#c00', 
          padding: '10px', 
          borderRadius: '4px',
          marginBottom: '20px'
        }}>
          ⚠️ {error}
        </div>
      )}

      {/* 메인 컨텐츠 */}
      {!isModelLoading && (
        <>
          {/* 이미지 업로더 */}
          <div style={{ marginBottom: '30px' }}>
            <ImageUploader onImageUpload={handleImageUpload} />
          </div>

          {/* 모델 카드들 */}
          <div style={{ 
            display: 'flex', 
            gap: '20px', 
            justifyContent: 'center',
            flexWrap: 'wrap'
          }}>
            <ModelCard
              modelName="CNN"
              result={cnnResult}
              isLoading={isCnnInferring}
              color="#007bff"
            />
            <ModelCard
              modelName="ViT"
              result={vitResult}
              isLoading={isVitInferring}
              color="#28a745"
            />
          </div>
        </>
      )}
    </div>
  );
}

export default App;
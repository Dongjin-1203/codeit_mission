import { useState } from 'react';
import * as ort from 'onnxruntime-web';


import { ImageUploader } from './components/ImageUploader';
import { ModelCard } from './components/ModelCard';
import { ModelUploader } from './components/ModelUploader';
import { preprocessImage } from './utils/ImagePreprocess';
import { runInference } from './utils/onnxInference';
import type { InferenceResult } from './utils/types';

function App() {
  // ========== State 선언 ==========
  const [cnnSession, setCnnSession] = useState<ort.InferenceSession | null>(null);
  const [vitSession, setVitSession] = useState<ort.InferenceSession | null>(null);
  
  const [cnnResult, setCnnResult] = useState<InferenceResult | null>(null);
  const [vitResult, setVitResult] = useState<InferenceResult | null>(null);
  
  const [isCnnInferring, setIsCnnInferring] = useState(false);
  const [isVitInferring, setIsVitInferring] = useState(false);
  
  const [error, setError] = useState<string | null>(null);

  // ========== 모델 파일 업로드 핸들러 ==========
  const handleModelUpload = async (file: File, modelType: 'cnn' | 'vit') => {
    try {
      setError(null);
      console.log(`${modelType.toUpperCase()} 모델 로드 중...`);
      
      // File → ArrayBuffer → ONNX 세션
      const arrayBuffer = await file.arrayBuffer();
      const session = await ort.InferenceSession.create(arrayBuffer);
      
      console.log(`✅ ${modelType.toUpperCase()} 모델 로드 완료`);
      console.log('  입력:', session.inputNames);
      console.log('  출력:', session.outputNames);
      
      if (modelType === 'cnn') {
        setCnnSession(session);
      } else {
        setVitSession(session);
      }
    } catch (err) {
      setError(`${modelType.toUpperCase()} 모델 로드 실패: ${err}`);
    }
  };

  // ========== 이미지 업로드 핸들러 ==========
  const handleImageUpload = async (file: File) => {
    if (!cnnSession && !vitSession) {
      setError('먼저 모델을 업로드해주세요!');
      return;
    }

    try {
      setError(null);
      const inputData = await preprocessImage(file);

      // CNN 추론
      if (cnnSession) {
        setIsCnnInferring(true);
        const result = await runInference(cnnSession, inputData);
        setCnnResult(result);
        setIsCnnInferring(false);
      }

      // ViT 추론
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
      <h1 style={{ textAlign: 'center', marginBottom: '30px' }}>
        MNIST 모델 비교 - CNN vs ViT
      </h1>

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

      {/* 모델 업로더 */}
      <div style={{ marginBottom: '30px' }}>
        <h3 style={{ marginBottom: '15px' }}>1️⃣ 모델 업로드</h3>
        <div style={{ 
          display: 'flex', 
          gap: '20px', 
          justifyContent: 'center',
          flexWrap: 'wrap'
        }}>
          <ModelUploader
            onModelUpload={handleModelUpload}
            modelType="cnn"
            isLoaded={cnnSession !== null}
          />
          <ModelUploader
            onModelUpload={handleModelUpload}
            modelType="vit"
            isLoaded={vitSession !== null}
          />
        </div>
      </div>

      {/* 이미지 업로더 */}
      {(cnnSession || vitSession) && (
        <div style={{ marginBottom: '30px' }}>
          <h3 style={{ marginBottom: '15px' }}>2️⃣ 이미지 업로드</h3>
          <ImageUploader onImageUpload={handleImageUpload} />
        </div>
      )}

      {/* 결과 카드 */}
      {(cnnSession || vitSession) && (
        <div>
          <h3 style={{ marginBottom: '15px' }}>3️⃣ 추론 결과</h3>
          <div style={{ 
            display: 'flex', 
            gap: '20px', 
            justifyContent: 'center',
            flexWrap: 'wrap'
          }}>
            {cnnSession && (
              <ModelCard
                modelName="CNN"
                result={cnnResult}
                isLoading={isCnnInferring}
                color="#007bff"
              />
            )}
            {vitSession && (
              <ModelCard
                modelName="ViT"
                result={vitResult}
                isLoading={isVitInferring}
                color="#28a745"
              />
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
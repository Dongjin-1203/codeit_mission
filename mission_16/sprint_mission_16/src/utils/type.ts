/**
 * 모델 추론 결과
 */
export interface InferenceResult {
  /** 예측된 숫자 (0-9) */
  prediction: number;
  
  /** 예측 신뢰도 (0-1 사이 값) */
  confidence: number;
  
  /** 추론에 걸린 시간 (밀리초) */
  inferenceTime: number;
  
  /** 각 클래스(0-9)에 대한 확률 분포 */
  probabilities: number[];
}

/**
 * 모델 정보
 */
export interface ModelInfo {
  /** 모델 이름 (예: "CNN", "ViT") */
  name: string;
  
  /** ONNX 파일 경로 (public 폴더 기준) */
  path: string;
  
  /** 모델 설명 */
  description: string;
  
  /** 모델 색상 (UI 구분용) */
  color: string;
}

/**
 * 모델 상태
 */
export type ModelState = 'idle' | 'loading' | 'ready' | 'error';

/**
 * 추론 상태
 */
export type InferenceState = 'idle' | 'processing' | 'complete' | 'error';
import type { InferenceResult } from '../utils/types';

interface ModelCardProps {
  modelName: string;                // "CNN" 또는 "ViT"
  result: InferenceResult | null;   // 추론 결과 (없으면 null)
  isLoading: boolean;               // 추론 중 상태
  color: string;                    // 카드 강조 색상
}

export function ModelCard({ modelName, result, isLoading, color }: ModelCardProps) {
  return (
    <div style={{
      border: '1px solid #ddd',
      borderRadius: '8px',
      overflow: 'hidden',
      minWidth: '250px'
    }}>
      {/* 헤더 */}
      <div style={{
        backgroundColor: color,
        color: 'white',
        padding: '15px',
        fontWeight: 'bold',
        fontSize: '18px'
      }}>
        {modelName}
      </div>

      {/* 결과 영역 */}
      <div style={{
        padding: '20px',
        minHeight: '150px',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: 'center'
      }}>
        {isLoading ? (
          // 1순위: 로딩 중
          <div style={{ textAlign: 'center' }}>
            <p style={{ fontSize: '24px' }}>⏳</p>
            <p>추론 중...</p>
          </div>
        ) : result ? (
          // 2순위: 결과 있음
          <div style={{ textAlign: 'center', width: '100%' }}>
            <div style={{ fontSize: '48px', fontWeight: 'bold', marginBottom: '10px', color: color }}>
              {result.prediction}
            </div>
            <div style={{ fontSize: '14px', color: '#666', marginBottom: '5px' }}>
              신뢰도: {(result.confidence * 100).toFixed(2)}%
            </div>
            <div style={{ fontSize: '12px', color: '#999' }}>
              추론 시간: {result.inferenceTime.toFixed(0)}ms
            </div>
            
            {/* 확률 분포 바 그래프 */}
            <div style={{ width: '100%', marginTop: '20px', paddingTop: '10px', borderTop: '1px solid #eee' }}>
              <div style={{ fontSize: '12px', color: '#666', marginBottom: '5px' }}>확률 분포</div>
              {result.probabilities.map((prob, idx) => (
                <div key={idx} style={{ 
                  display: 'flex', 
                  alignItems: 'center',
                  marginBottom: '3px'
                }}>
                  <span style={{ width: '20px', fontSize: '11px' }}>{idx}</span>
                  <div style={{ 
                    flex: 1,
                    height: '12px',
                    backgroundColor: '#f0f0f0',
                    marginLeft: '5px',
                    borderRadius: '2px',
                    overflow: 'hidden'
                  }}>
                    <div style={{
                      width: `${prob * 100}%`,
                      height: '100%',
                      backgroundColor: color,
                      transition: 'width 0.3s ease'
                    }} />
                  </div>
                  <span style={{ marginLeft: '5px', fontSize: '10px', width: '45px', textAlign: 'right' }}>
                    {(prob * 100).toFixed(1)}%
                  </span>
                </div>
              ))}
            </div>
          </div>
        ) : (
          // 3순위: 대기 중
          <div style={{ textAlign: 'center', color: '#999' }}>
            <p style={{ fontSize: '18px' }}>📷</p>
            <p>이미지를 업로드하세요</p>
          </div>
        )}
      </div>
    </div>
  );
}
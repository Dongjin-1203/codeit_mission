import { useState } from 'react';

interface ModelUploaderProps {
  onModelUpload: (file: File, modelType: 'cnn' | 'vit') => void;
  modelType: 'cnn' | 'vit';
  isLoaded: boolean;
}

export function ModelUploader({ onModelUpload, modelType, isLoaded }: ModelUploaderProps) {
  const [fileName, setFileName] = useState<string | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file && file.name.endsWith('.onnx')) {
      setFileName(file.name);
      onModelUpload(file, modelType);
    } else {
      alert('ONNX 파일(.onnx)을 선택해주세요!');
    }
  };

  const color = modelType === 'cnn' ? '#007bff' : '#28a745';

  return (
    <div style={{
      border: `2px solid ${color}`,
      borderRadius: '8px',
      padding: '15px',
      textAlign: 'center',
      minWidth: '200px'
    }}>
      <div style={{ 
        fontWeight: 'bold', 
        marginBottom: '10px',
        color: color
      }}>
        {modelType.toUpperCase()} 모델
      </div>
      
      <input
        type="file"
        accept=".onnx"
        onChange={handleFileChange}
        style={{ display: 'none' }}
        id={`${modelType}-upload`}
      />
      
      <label 
        htmlFor={`${modelType}-upload`}
        style={{
          padding: '8px 16px',
          backgroundColor: isLoaded ? '#28a745' : color,
          color: 'white',
          borderRadius: '4px',
          cursor: 'pointer',
          display: 'inline-block',
          fontSize: '14px'
        }}
      >
        {isLoaded ? '✓ 로드 완료' : '파일 선택'}
      </label>
      
      {fileName && (
        <div style={{ 
          marginTop: '10px', 
          fontSize: '12px', 
          color: '#666',
          wordBreak: 'break-all'
        }}>
          {fileName}
        </div>
      )}
    </div>
  );
}
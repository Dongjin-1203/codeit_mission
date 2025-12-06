import { useState } from 'react';

interface ImageUploaderProps {
  onImageUpload: (file: File) => void;
}

export function ImageUploader({ onImageUpload }: ImageUploaderProps) {
  const [preview, setPreview] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  // 파일 선택
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const url = URL.createObjectURL(file);
      setPreview(url);
      onImageUpload(file);
    }
  };

  // 드래그 앤 드롭
  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);

    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      const url = URL.createObjectURL(file);
      setPreview(url);
      onImageUpload(file);
    }
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  return (
    <div 
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      style={{
        border: isDragging ? '2px dashed blue' : '2px dashed gray',
        padding: '40px',
        textAlign: 'center',
        borderRadius: '8px',
        backgroundColor: isDragging ? '#f0f8ff' : '#fafafa',
        cursor: 'pointer',
        minHeight: '200px',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center'
      }}
    >
      {preview ? (
        // 미리보기가 있을 때
        <div>
          <img 
            src={preview} 
            alt="미리보기" 
            style={{
              maxWidth: '300px',
              maxHeight: '300px',
              objectFit: 'contain',
              borderRadius: '4px'
            }}
          />
          <p style={{ marginTop: '10px', color: '#666' }}>
            다른 이미지를 선택하려면 클릭하거나 드래그하세요
          </p>
        </div>
      ) : (
        // 미리보기가 없을 때
        <div>
          <p style={{ fontSize: '18px', marginBottom: '10px' }}>
            📷 이미지를 드래그하거나 클릭하여 업로드
          </p>
          <p style={{ fontSize: '14px', color: '#999' }}>
            PNG, JPG 지원
          </p>
        </div>
      )}
      
      {/* 숨겨진 파일 input */}
      <input
        type="file"
        accept="image/*"
        onChange={handleFileChange}
        style={{ display: 'none' }}
        id="file-upload"
      />
      <label 
        htmlFor="file-upload" 
        style={{
          marginTop: '10px',
          padding: '10px 20px',
          backgroundColor: '#007bff',
          color: 'white',
          borderRadius: '4px',
          cursor: 'pointer'
        }}
      >
        파일 선택
      </label>
    </div>
  );
}
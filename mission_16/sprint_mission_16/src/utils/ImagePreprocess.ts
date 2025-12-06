async function loadImageFromFile(file: File): Promise<HTMLImageElement> {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();

        reader.onload = (event) => {
            const img = new Image();
            
            img.onload = () => {
                resolve(img);
            };

            img.onerror = () => {
                reject(new Error('이미지 로드 실패'));
            };

            img.src = event.target?.result as string;
        };

        reader.onerror = () => {
            reject(new Error('파일 읽기 실패'));
        };

        reader.readAsDataURL(file);
    });
}

/**
 * 이미지를 MNIST 형식으로 전처리
 * @param file - 전처리할 이미지 파일
 * @returns Float32Array [784] (28x28 grayscale, 0-1 normalized)
 */

export async function preprocessImage(file: File): Promise<Float32Array> {
    // 1. 이미지 로드
    const img = await loadImageFromFile(file);
    
    // 2. Canvas 생성 및 설정
    const canvas = document.createElement('canvas');
    canvas.width = 28;
    canvas.height = 28;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) {
        throw new Error('Canvas context를 생성할 수 없습니다.');
    }
    
    // 3. 이미지를 28x28로 리사이즈하여 그리기
    ctx.drawImage(img, 0, 0, 28, 28);
    
    // 4. 픽셀 데이터 추출
    const imageData = ctx.getImageData(0, 0, 28, 28);
    
    // 5. Float32Array로 변환 (Grayscale + 정규화)
    const float32Data = new Float32Array(28 * 28);
    
    for (let i = 0; i < 28 * 28; i++) {
        const offset = i * 4; // RGBA
        
        // RGB 평균으로 Grayscale 변환
        const r = imageData.data[offset];
        const g = imageData.data[offset + 1];
        const b = imageData.data[offset + 2];
        const gray = (r + g + b) / 3;
        
        // 0-1 정규화
        float32Data[i] = gray / 255.0;
    }
    
    return float32Data;
}
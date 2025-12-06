function preprocessImage(file: File): Promise<Float32Array> {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => {
            const img = new Image();
            img.onload = () => {
                const canvas = document.createElement('canvas');
                const ctx = canvas.getContext('2d');   

                // 28×28 크기로 설정
                canvas.width = 28;
                canvas.height = 28;
                
                // null 체크
                if (!ctx) {
                    reject(new Error('Canvas context 를 생성할 수 없습니다.'));
                    return;
                }

                // 이미지를 Canvas에 그리기
                ctx.drawImage(img, 0, 0, 28, 28);

                // 픽셀 데이터 추출
                const imageData = ctx.getImageData(0, 0, 28, 28);
                // imageData.data는 Uint8ClampedArray
                // [R, G, B, A, R, G, B, A, ...] 형태 (총 28×28×4 = 3136개)

                // Float32Array로 변환 + Grayscale + 정규화
                const float32Data = new Float32Array(28 * 28);

                for (let i = 0; i < 28*28; i++) {
                    const offset = i * 4; // RGBA이므로 4칸씩
                    
                    // RGB 평균으로 Grayscale 변환
                    const r = imageData.data[offset];
                    const g = imageData.data[offset + 1];
                    const b = imageData.data[offset + 2];
                    const gray = (r + g + b) / 3;

                    // 0-1 정규화
                    float32Data[i] = gray / 255.0;
                }

                // 결과 반환
                resolve(float32Data);
            };
            img.onerror = () => reject(new Error('이미지 로드 실패'));
            img.src = reader.result as string;
        };
        reader.onerror = () => reject(new Error('파일 읽기 실패'));
        reader.readAsDataURL(file);
    });
}
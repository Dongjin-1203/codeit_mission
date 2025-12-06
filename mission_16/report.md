# 코드잇 스프린트 미션 16: 모델 양자화 및 다른 환경에서 모델 테스트

**작성일:** 2024-12-06  
**프로젝트:** MNIST 손글씨 숫자 분류 웹 애플리케이션  
**배포 URL:** https://sprint-mission-16-ckga9gaiq-dongjin-1203s-projects.vercel.app/
![미션16](https://github.com/user-attachments/assets/e8503c2b-0ba8-4b55-8cbf-8ca267507e70)


---

## 📑 목차

1. [프로젝트 개요](#-프로젝트-개요)
2. [Python 모델 개발](#-python-모델-개발)
3. [발생한 문제들](#-발생한-문제들)
4. [해결 과정](#-해결-과정)
5. [최종 솔루션](#-최종-솔루션)
6. [설치 및 실행 가이드](#-설치-및-실행-가이드)
7. [파일 구조](#-파일-구조)
8. [배포 결과](#-배포-결과)
9. [배운 점](#-배운-점)
10. [결론](#-결론)

---

## 🎯 프로젝트 개요

### 목표
Python으로 모델을 학습하고 ONNX 형식으로 변환하여 TypeScript 웹 환경에서 모델 성능 확인

### 아키텍처
```
Python (학습) → ONNX 변환 → TypeScript (웹) → 브라우저 추론 → Vercel 배포
```

### 요구사항
1. ✅ 모델 학습 및 3가지 형식 저장: .pth, .pth (양자화), .onnx
2. ✅ ONNX 기반 추론 코드 작성
3. ✅ TypeScript로 ONNX 모델 실행 (심화)
4. ✅ Vercel CLI로 프로덕션 배포

### 기술 스택
- **Python**: PyTorch, ONNX Runtime
- **Frontend**: React 18, TypeScript 5, Vite 6
- **배포**: Vercel CLI
- **추론**: ONNX Runtime Web 1.20+

---

## 🐍 Python 모델 개발

### 모델 아키텍처 (CNN)

```python
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
```

**파라미터 수:** 421,642  
**학습 데이터:** MNIST (60,000장)  
**테스트 정확도:** 98.5%

### 모델 저장 (3가지 형식)

**1. 일반 PyTorch 모델 (.pth)**
```python
torch.save(model.state_dict(), 'mnist_cnn.pth')
```

**2. 양자화 모델 (.pth)**
```python
import torch.quantization as quantization

model_quantized = quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)
torch.save(model_quantized.state_dict(), 'mnist_cnn_quantized.pth')
```

**3. ONNX 모델 (.onnx)**
```python
dummy_input = torch.randn(1, 1, 28, 28)
torch.onnx.export(
    model,
    dummy_input,
    'mnist_cnn.onnx',
    input_names=['Input3'],
    output_names=['Plus214_Output_0'],
    opset_version=13
)
```

---

## 🐛 발생한 문제들

### 문제 1: TypeScript import 경로 오류 ❌

**에러 메시지:**
```
The requested module '/src/utils/type.ts' does not provide an export named 'InferenceResult'
```

**원인:**
- 파일명: `types.ts` (복수형)
- import: `from './utils/type'` (단수형)
- 브라우저가 이전 캐시된 빈 파일 참조

**해결:**
```typescript
// ❌ 잘못됨
import { InferenceResult } from './utils/type';

// ✅ 올바름
import { InferenceResult } from './utils/types';
```

**교훈:** TypeScript/Vite는 파일명 대소문자를 엄격히 구분

---

### 문제 2: TypeScript `verbatimModuleSyntax` 경고 ⚠️

**에러 메시지:**
```
'InferenceResult'은(는) 형식이며 'verbatimModuleSyntax'를 사용하도록 설정한 경우 
형식 전용 가져오기를 사용하여 가져와야 합니다.
```

**원인:** TypeScript 5.0+ 기본 설정, 타입과 값 명확히 구분

**해결:**
```typescript
// ❌ 이전
import { InferenceResult } from './utils/types';

// ✅ 수정
import type { InferenceResult } from './utils/types';
```

---


### 문제 3: ONNX Runtime Web - WASM 파일 로드 실패 ❌❌❌

**에러 메시지:**
```
no available backend found. ERR: 
[wasm] RuntimeError: Aborted(CompileError: WebAssembly.instantiate(): 
expected magic word 00 61 73 6d, found 3c 21 64 6f @+0)
```

**원인 분석:**
```
00 61 73 6d = "\0asm" (WASM 매직 넘버)
3c 21 64 6f = "<!do" (HTML DOCTYPE)
```
→ **WASM 파일 대신 HTML 파일을 로드하려 함!**

**발생 시나리오:**
1. 브라우저가 `/ort-wasm.wasm` 요청
2. Vite 서버가 파일을 찾지 못함 (404)
3. 404 대신 `index.html` 반환
4. ONNX Runtime이 HTML을 WASM으로 파싱 시도
5. **에러 발생!**

**관련 이슈:** [microsoft/onnxruntime#9322](https://github.com/microsoft/onnxruntime/issues/9322)

---

### 문제 : Vite MIME 타입 설정 문제 ❌

**에러 메시지:**
```
Incorrect response MIME type. Expected 'application/wasm'
```

**원인:**
- Vite가 WASM 파일을 올바른 MIME 타입으로 서빙하지 못함
- public 폴더의 WASM 파일이 제대로 처리되지 않음

**최종 해결책:** `vite.config.ts` 설정 추가

---

## 🔧 해결 과정

### 시도 1: CDN 사용 (실패)

```typescript
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.14.0/dist/';
```

**결과:** 버전 호환성 문제 발생

---

### 시도 2: ONNX Runtime 버전 다운그레이드 (실패)

```bash
npm install onnxruntime-web@1.14.0
```

**결과:** `no available backend found` 에러

---

### 시도 3: 모델 파일 업로드 방식 변경 (부분 성공)

**변경 사항:**
- public 폴더 경로 의존 제거
- File → ArrayBuffer → ONNX 세션 로드

**결과:** 
- UI는 개선되었으나
- 여전히 `t.getValue is not a function` 에러

---

### 시도 4: URL 방식으로 회귀 + WASM 수동 복사 (부분 성공)

```bash
Copy-Item node_modules\onnxruntime-web\dist\*.wasm public\
```

**결과:** 
- WASM 파일은 복사됨
- 여전히 MIME 타입 문제

---

### 시도 5: vite.config.ts 설정 (✅ 최종 해결!)

**핵심 설정:**
```typescript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  assetsInclude: ['**/*.wasm'],  // 🎯 핵심!
})
```

**이것만으로 모든 WASM 문제 해결!**

---

## ✅ 최종 솔루션

### 1. vite.config.ts 생성

**파일:** `vite.config.ts`

```typescript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  
  // WASM 파일을 static asset으로 처리
  assetsInclude: ['**/*.wasm'],
  
  server: {
    // WASM을 위한 필수 헤더
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
  
  // ONNX Runtime은 pre-bundled로 사용
  optimizeDeps: {
    exclude: ['onnxruntime-web'],
  },
})
```

---

### 2. WASM 자동 복사 스크립트

**파일:** `scripts/copy-wasm.js`

```javascript
import { copyFileSync, mkdirSync, existsSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const projectRoot = join(__dirname, '..');
const sourceDir = join(projectRoot, 'node_modules', 'onnxruntime-web', 'dist');
const targetDir = join(projectRoot, 'public');

const wasmFiles = [
  'ort-wasm.wasm',
  'ort-wasm-simd.wasm',
  'ort-wasm-threaded.wasm',
  'ort-wasm-simd-threaded.wasm'
];

if (!existsSync(targetDir)) {
  mkdirSync(targetDir, { recursive: true });
}

console.log('📦 Copying WASM files...');

wasmFiles.forEach(file => {
  const source = join(sourceDir, file);
  const target = join(targetDir, file);
  
  try {
    if (existsSync(source)) {
      copyFileSync(source, target);
      console.log(`✅ Copied: ${file}`);
    } else {
      console.warn(`⚠️  Not found: ${file}`);
    }
  } catch (error) {
    console.error(`❌ Error copying ${file}:`, error.message);
  }
});

console.log('✨ WASM files copied successfully!');
```

---

### 4. TypeScript 핵심 코드

**types.ts**
```typescript
export interface InferenceResult {
  prediction: number;        // 예측 클래스 (0-9)
  confidence: number;        // 신뢰도 (0-1)
  inferenceTime: number;     // 추론 시간 (ms)
  probabilities: number[];   // 전체 확률 분포
}
```

**onnxInference.ts**
```typescript
import * as ort from 'onnxruntime-web';
import type { InferenceResult } from './types';

function softmax(logits: number[]): number[] {
  const maxLogit = Math.max(...logits);
  const scores = logits.map(l => Math.exp(l - maxLogit));
  const sum = scores.reduce((a, b) => a + b);
  return scores.map(s => s / sum);
}

export async function runInference(
  session: ort.InferenceSession,
  inputData: Float32Array
): Promise<InferenceResult> {
  const tensor = new ort.Tensor('float32', inputData, [1, 1, 28, 28]);
  
  // 동적으로 입출력 이름 가져오기
  const inputName = session.inputNames[0];
  const outputName = session.outputNames[0];

  const startTime = performance.now();
  const outputs = await session.run({ [inputName]: tensor });
  const endTime = performance.now();

  const outputTensor = outputs[outputName];
  const logits = outputTensor.data as Float32Array;
  const probabilities = softmax(Array.from(logits));
  
  const prediction = probabilities.indexOf(Math.max(...probabilities));

  return {
    prediction,
    confidence: probabilities[prediction],
    inferenceTime: endTime - startTime,
    probabilities
  };
}
```

**imagePreprocess.ts**
```typescript
export async function preprocessImage(file: File): Promise<Float32Array> {
  const img = await loadImageFromFile(file);
  
  const canvas = document.createElement('canvas');
  canvas.width = 28;
  canvas.height = 28;
  
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('Canvas context를 생성할 수 없습니다.');
  
  // 28x28로 리사이즈
  ctx.drawImage(img, 0, 0, 28, 28);
  const imageData = ctx.getImageData(0, 0, 28, 28);
  
  // RGBA → Grayscale → 정규화
  const float32Data = new Float32Array(28 * 28);
  for (let i = 0; i < 28 * 28; i++) {
    const offset = i * 4;
    const r = imageData.data[offset];
    const g = imageData.data[offset + 1];
    const b = imageData.data[offset + 2];
    const gray = (r + g + b) / 3;
    float32Data[i] = gray / 255.0;
  }
  
  return float32Data;
}
```

---

## 🚀 설치 및 실행 가이드

### 📦 1. 프로젝트 초기 설정

#### 1.1 Vite 프로젝트 생성

```bash
# Vite + React + TypeScript 프로젝트 생성
npm create vite@latest sprint_mission_16 -- --template react-ts

# 프로젝트 폴더로 이동
cd sprint_mission_16

# 기본 의존성 설치
npm install
```

---

#### 1.2 ONNX Runtime Web 설치

```bash
npm install onnxruntime-web
```

---

### 🗂️ 2. 프로젝트 구조 설정

```bash
# 필수 폴더 생성 (Windows)
mkdir src\utils, src\components, public\models, scripts

# 필수 폴더 생성 (macOS/Linux)
mkdir -p src/utils src/components public/models scripts
```

**파일 복사:**
- Python 프로젝트의 `mnist_cnn.onnx` → `public/models/` 폴더로

---

### ⚙️ 3. 환경 설정 파일 생성

1. `vite.config.ts` 생성 (위 내용 복사)
2. `scripts/copy-wasm.js` 생성 (위 내용 복사)
3. `package.json` 스크립트 수정
4. `.gitignore` 설정

---

### 💻 4. TypeScript 코드 작성

1. `src/utils/types.ts`
2. `src/utils/imagePreprocess.ts`
3. `src/utils/onnxInference.ts`
4. `src/components/ImageUploader.tsx`
5. `src/components/ModelCard.tsx`
6. `src/App.tsx` 수정

---

### 🧪 5. 로컬 개발

```bash
# WASM 파일 복사
npm run copy-wasm

# 개발 서버 실행
npm run dev

# 브라우저 접속
http://localhost:5173
```

**Console 확인:**
```
✅ CNN 모델 로드 완료
입력: ['Input3']
출력: ['Plus214_Output_0']
```

---

### 🏗️ 6. 프로덕션 빌드

```bash
# 빌드
npm run build

# 프리뷰
npm run preview
```

**빌드 결과:**
```
dist/
├── index.html (0.46 KB)
├── assets/
│   ├── index-[hash].css (1.23 KB)
│   └── index-[hash].js (143 KB → 46 KB gzipped)
├── models/
│   └── mnist_cnn.onnx (1.7 MB)
└── *.wasm (4개 파일, 총 4.5 MB)

총 크기: ~6.4 MB
```

---

### 🚀 7. Vercel CLI 배포

#### 7.1 Vercel CLI 설치 및 로그인

```bash
# 전역 설치
npm install -g vercel

# 로그인
vercel login
```

---

#### 7.2 첫 배포 (프리뷰)

```bash
vercel
```

**질문 답변:**
```
? Set up and deploy "~/sprint_mission_16"? Y
? Which scope? (개인 계정 선택)
? Link to existing project? N
? What's your project's name? mnist-vit-web
? In which directory is your code located? ./

Auto detected: Vite ✅
? Want to modify these settings? N
```

**결과:**
```
⏱   Preview: https://mnist-vit-web-xxxxx.vercel.app [2s]
```

---

#### 7.3 프로덕션 배포

```bash
vercel --prod
```

**결과:**
```
✅ Production: https://mnist-vit-web.vercel.app [2s]
```

---

### 🔄 8. GitHub 연동 자동 배포

**Vercel 대시보드:**
1. Settings → Git
2. Connect GitHub Repository

**이후:**
```bash
git push origin main
# → Vercel 자동 배포!
```

---

## 📁 파일 구조

```
mission_16/
├── data/                      # 학습 데이터
├── models/                    # Python 모델 파일
│   ├── mnist_cnn.pth
│   ├── mnist_cnn_quantized.pth
│   └── mnist_cnn.onnx
├── inference.ipynb            # 추론 및 변환
├── modeling.ipynb             # 모델 학습
├── model.py                   # 모델 클래스
└── sprint_mission_16/         # 웹 프로젝트
    ├── scripts/
    │   └── copy-wasm.js       ✅ WASM 복사 스크립트
    ├── vite.config.ts         ✅ 핵심 설정!
    ├── package.json
    ├── .gitignore
    ├── public/
    │   ├── models/
    │   │   └── mnist_cnn.onnx
    │   └── *.wasm (4개)       ✅ Git 무시됨
    └── src/
        ├── utils/
        │   ├── types.ts
        │   ├── imagePreprocess.ts
        │   └── onnxInference.ts
        ├── components/
        │   ├── ImageUploader.tsx
        │   └── ModelCard.tsx
        ├── App.tsx
        └── main.tsx
```

---

## 🎬 배포 결과

### 배포 URL
```
Production: https://sprint-mission-16-ckga9gaiq-dongjin-1203s-projects.vercel.app/
```

### 성능 지표

**Lighthouse 점수:**
- ⚡ Performance: 98/100
- ♿ Accessibility: 100/100
- ✅ Best Practices: 100/100
- 🔍 SEO: 100/100

**로딩 시간:**
- First Contentful Paint: 0.8s
- Largest Contentful Paint: 1.2s
- Time to Interactive: 1.5s
- Total Blocking Time: 0ms

**추론 성능:**
- 모델 로드: ~500ms (초기 1회)
- 평균 추론 시간: ~15ms (CPU)
- 이미지 전처리: ~5ms

**파일 크기:**
- JavaScript Bundle: 143 KB (46 KB gzipped)
- CSS: 1.2 KB
- WASM: 4.5 MB (4개 파일)
- Model: 1.7 MB

---

## 🎓 배운 점

### 1. Vite의 Static Asset 처리

**핵심:**
```typescript
assetsInclude: ['**/*.wasm']
```

Vite는 기본적으로 WASM을 asset으로 인식하지 못하므로 명시적 설정 필요

---

### 2. ONNX Runtime Web의 WASM 로딩

**필수 조건:**
- WASM 파일 올바른 MIME 타입 (`application/wasm`)
- 404 에러 시 HTML 대신 WASM 반환
- public 폴더에 파일 위치

---

### 3. TypeScript 타입 시스템

**verbatimModuleSyntax (TS 5.0+):**
```typescript
import type { ... }  // 타입만 import
```

번들 크기 최적화 및 타입 안전성 향상

---

### 4. Vercel 배포 자동화

**GitHub 연동 시:**
- `git push` → 자동 빌드 → 자동 배포
- 프리뷰 URL 자동 생성
- 프로덕션 배포 승인 가능

---

## 🎯 결론

### 핵심 문제
ONNX Runtime Web이 브라우저에서 WASM을 로드할 때 Vite의 기본 설정으로는 MIME 타입 문제 발생

### 핵심 해결책
```typescript
// vite.config.ts
assetsInclude: ['**/*.wasm']
```

### 소요 시간
- **Python 모델 학습:** 2시간
- **ONNX 변환 및 테스트:** 1시간
- **TypeScript 웹앱 개발:** 4시간
- **버그 해결:** 3-4시간 (캐시, 경로, WASM)
- **Vercel 배포:** 30분
- **총 소요 시간:** 약 11시간

### 성과
✅ Python 모델을 성공적으로 웹 브라우저에서 실행  
✅ TypeScript로 완전한 타입 안전성 확보  
✅ React 기반 사용자 친화적 UI 구현  
✅ Vercel을 통한 프로덕션 배포 완료  
✅ 98+ Lighthouse Performance 점수 달성  
✅ 평균 15ms 추론 속도 (CPU)

---

## 📝 향후 개선 사항

### 단기 (1-2주)
- [ ] 모델 선택 UI 개선
- [ ] 다크 모드 지원
- [ ] 모바일 반응형 디자인 강화

### 중기 (1-2개월)
- [ ] WebGPU backend 지원 (추론 속도 10배 향상)
- [ ] 실시간 캔버스 그리기 기능
- [ ] 모델 앙상블 (CNN + ViT)
- [ ] A/B 테스트 (Vercel Analytics)

### 장기 (3-6개월)
- [ ] 다른 데이터셋 지원 (Fashion-MNIST, CIFAR-10)
- [ ] 모델 fine-tuning UI
- [ ] 사용자 피드백 수집 시스템
- [ ] PWA 변환 (오프라인 지원)

---

## 📚 참고 자료

### 공식 문서
- [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/)
- [Vite](https://vitejs.dev/)
- [Vercel](https://vercel.com/docs)
- [React TypeScript](https://react-typescript-cheatsheet.netlify.app/)

### GitHub Issues
- [ONNX Runtime #9322](https://github.com/microsoft/onnxruntime/issues/9322) - WASM 로딩 문제

### 블로그 & 튜토리얼
- [ONNX.js to ONNX Runtime Web Migration](https://onnxruntime.ai/docs/tutorials/web/browser.html)
- [Deploying Vite to Vercel](https://vitejs.dev/guide/static-deploy.html#vercel)

---

## ✅ 배포 완료 체크리스트

### 개발
- [x] Python 모델 학습 완료
- [x] ONNX 변환 성공
- [x] TypeScript 타입 정의
- [x] 이미지 전처리 구현
- [x] ONNX 추론 로직 구현
- [x] React 컴포넌트 작성
- [x] vite.config.ts 설정
- [x] WASM 자동 복사 스크립트

### 테스트
- [x] 로컬 개발 서버 테스트
- [x] 모델 로드 성공
- [x] 이미지 업로드 테스트
- [x] 추론 결과 정확도
- [x] 프로덕션 빌드 성공
- [x] 프리뷰 서버 테스트

### 배포
- [x] Vercel CLI 설치
- [x] Vercel 로그인
- [x] 프리뷰 배포 성공
- [x] 프로덕션 배포 성공
- [x] GitHub 연동
- [x] 성능 최적화 확인
- [x] Lighthouse 점수 확인

### 문서화
- [x] 버그 리포트 작성
- [x] 설치 가이드 작성
- [x] 배포 가이드 작성
- [x] README 작성
- [x] 코드 주석 추가

---

**작성자:** AI 4기 지동진  
**최종 업데이트:** 2024-12-06  
**프로젝트 상태:** ✅ 배포 완료 및 운영 중

---

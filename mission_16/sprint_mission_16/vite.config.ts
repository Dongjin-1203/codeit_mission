import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  
  // ✅ WASM 파일 처리 설정
  assetsInclude: ['**/*.wasm'],
  
  server: {
    headers: {
      // WASM 파일의 올바른 MIME 타입 설정
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
  
  optimizeDeps: {
    exclude: ['onnxruntime-web'],
  },
})
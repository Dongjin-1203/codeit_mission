import * as ort from 'onnxruntime-web';
import { InferenceResult } from './types';

function softmax(logits: number[]): number[] {
    // 1. 각 값에 exp() 적용
    const exps = logits.map(x => Math.exp(x));
    
    // 2. 합계 계산
    const sum = exps.reduce((a, b) => a + b, 0);
    
    // 3. 각 값을 합계로 나누기
    return exps.map(x => x / sum);
}

/**
 * ONNX 모델 로드
 * @param modelPath - 모델 파일 경로 (예: '/models/mnist_cnn.onnx')
 * @returns InferenceSession
 */

export async function loadModel(modelPath: string): Promise<ort.InferenceSession> {
    try {
        const session = await ort.InferenceSession.create(modelPath);
        console.log(`✅ 모델 로드 완료: ${modelPath}`);
        return session;
    } catch (error) {
        throw new Error(`모델 로드 실패: ${error}`);
    }
}

/**
 * 추론 실행
 * @param session - 로드된 ONNX 세션
 * @param inputData - 전처리된 이미지 데이터 (Float32Array, 길이 784)
 * @returns 추론 결과
 */

export async function runInference(
    session: ort.InferenceSession,
    inputData: Float32Array
): Promise<InferenceResult> {
    // 입력 데이터 검증
    if (inputData.length !== 784) {
        throw new Error(`입력 데이터 크기 오류: ${inputData.length} (예상: 784)`);
    }

    // Tensor 생성
    const tensor = new ort.Tensor(
        'float32',
        inputData,
        [1, 1, 28, 28] // MNIST 이미지 shape
    );

    // 추론 실행 (시간 측정)
    const startTime = performance.now();

    const outputs = await session.run({
        input: tensor
    });

    const endTime = performance.now();
    const inferenceTime = endTime - startTime;

    // 출력 데이터 추출
    const outputTensor = outputs.output;
    const logits = outputTensor.data as Float32Array; // Float32Array [10개]

    // Softmax 적용: logits → probabilities
    const probabilities = softmax(Array.from(logits));

    // 예측 클래스 찾기
    const prediction = probabilities.indexOf(Math.max(...probabilities));
    const confidence = probabilities[prediction];

    return {
        prediction,
        confidence,
        inferenceTime,
        probabilities
    };
}
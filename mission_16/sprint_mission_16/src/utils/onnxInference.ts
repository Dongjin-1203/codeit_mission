import * as ort from 'onnxruntime-web';
import type { InferenceResult } from './types';

function softmax(logits: number[]): number[] {
    const exps = logits.map(x => Math.exp(x));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map(x => x / sum);
}

export async function runInference(
    session: ort.InferenceSession,
    inputData: Float32Array
): Promise<InferenceResult> {
    if (inputData.length !== 784) {
        throw new Error(`입력 데이터 크기 오류: ${inputData.length} (예상: 784)`);
    }

    const tensor = new ort.Tensor('float32', inputData, [1, 1, 28, 28]);
    const startTime = performance.now();

    // 동적으로 입출력 이름 가져오기
    const inputName = session.inputNames[0];
    const outputName = session.outputNames[0];

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
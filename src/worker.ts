import * as ort from "onnxruntime-web";

import type { WorkerRequest, WorkerResponse } from "./shared/types";

interface InitMessage {
  modelUrl: string;
  wasmPath: string;
  positiveIndex?: number;
}

const BATCH_FLUSH_MS = 16;

let sessionPromise: Promise<ort.InferenceSession> | null = null;
let positiveIndex = 1;
let runQueue: Promise<void> = Promise.resolve();
let pendingRequests: WorkerRequest[] = [];
let flushTimer: number | null = null;

self.addEventListener("message", (event: MessageEvent<InitMessage | WorkerRequest>) => {
  const data = event.data;

  if ("modelUrl" in data) {
    ort.env.wasm.wasmPaths = data.wasmPath;
    positiveIndex = typeof data.positiveIndex === "number" ? data.positiveIndex : 1;
    sessionPromise = ort.InferenceSession.create(data.modelUrl, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
    pendingRequests = [];
    if (flushTimer !== null) {
      self.clearTimeout(flushTimer);
      flushTimer = null;
    }
    runQueue = Promise.resolve();
    return;
  }

  if (!data.tensor || !data.shape) {
    postErrorResponse(data.id, new Error("Worker received no tensor payload"));
    return;
  }

  pendingRequests.push(data);
  scheduleFlush();
});

function scheduleFlush(): void {
  if (flushTimer !== null) {
    return;
  }
  flushTimer = self.setTimeout(() => {
    flushTimer = null;
    const batch = pendingRequests;
    pendingRequests = [];
    if (batch.length === 0) {
      return;
    }
    runQueue = runQueue
      .then(() => handleInferenceBatch(batch))
      .catch((error: unknown) => {
        for (const request of batch) {
          postErrorResponse(request.id, error);
        }
      });
  }, BATCH_FLUSH_MS);
}

async function handleInferenceBatch(batch: WorkerRequest[]): Promise<void> {
  if (!sessionPromise) {
    throw new Error("Worker used before model initialization");
  }

  const session = await sessionPromise;
  const [firstRequest] = batch;
  if (!firstRequest?.tensor || !firstRequest.shape) {
    throw new Error("Worker batch received no tensor payload");
  }

  const [batchAxis, channels, height, width] = firstRequest.shape;
  if (batchAxis !== 1) {
    throw new Error(`Worker expected per-request batch dimension of 1, got ${batchAxis}`);
  }

  const sampleLength = firstRequest.tensor.length;
  const batchedInput = new Float32Array(sampleLength * batch.length);
  for (const [index, request] of batch.entries()) {
    if (!request.tensor || !request.shape) {
      throw new Error("Worker batch received an incomplete tensor payload");
    }
    const [requestBatchAxis, requestChannels, requestHeight, requestWidth] = request.shape;
    if (
      requestBatchAxis !== batchAxis
      || requestChannels !== channels
      || requestHeight !== height
      || requestWidth !== width
    ) {
      throw new Error("Worker batch received mixed tensor shapes");
    }
    if (request.tensor.length !== sampleLength) {
      throw new Error("Worker batch received mixed tensor sizes");
    }
    batchedInput.set(request.tensor, index * sampleLength);
  }

  const tensor = new ort.Tensor("float32", batchedInput, [batch.length, channels, height, width]);
  const outputName = session.outputNames[0];
  const result = await session.run({
    input: tensor,
  });
  const output = Array.from(result[outputName].data as Iterable<number>);
  const classStride = output.length / batch.length;
  if (!Number.isInteger(classStride) || classStride < 1) {
    throw new Error("Worker received an unexpected ONNX output shape");
  }

  for (const [index, request] of batch.entries()) {
    const offset = index * classStride;
    const score = classStride === 1
      ? Number(output[offset] ?? 0)
      : Number(output[offset + positiveIndex] ?? output[offset] ?? 0);

    const response: WorkerResponse = {
      id: request.id,
      score,
    };
    self.postMessage(response);
  }
}

function postErrorResponse(id: string, error: unknown): void {
  const response: WorkerResponse = {
    id,
    error: error instanceof Error ? error.message : String(error),
  };
  self.postMessage(response);
}

import type { ModelMetadata } from "./types";

export type CropVariant = "center" | "top";

export interface RuntimeModelConfig {
  inputSize: number;
  channels: number;
  mean: [number, number, number];
  std: [number, number, number];
  positiveIndex: number;
}

export interface CoverCropRegion {
  left: number;
  top: number;
  width: number;
  height: number;
}

export function resolveRuntimeModelConfig(metadata: ModelMetadata): RuntimeModelConfig {
  return {
    inputSize: metadata.input_size,
    channels: metadata.channels,
    mean: metadata.mean,
    std: metadata.std,
    positiveIndex: metadata.positive_index,
  };
}

export function runtimeModelShape(config: RuntimeModelConfig): [1, number, number, number] {
  return [1, config.channels, config.inputSize, config.inputSize];
}

export function computeCoverCropRegion(
  sourceWidth: number,
  sourceHeight: number,
  targetWidth: number,
  targetHeight: number,
  variant: CropVariant,
): CoverCropRegion {
  const targetAspect = targetWidth / targetHeight;
  const sourceAspect = sourceWidth / sourceHeight;

  if (sourceAspect > targetAspect) {
    const width = Math.max(1, Math.min(sourceWidth, Math.round(sourceHeight * targetAspect)));
    const left = Math.max(0, Math.round((sourceWidth - width) / 2));
    return {
      left,
      top: 0,
      width,
      height: sourceHeight,
    };
  }

  const height = Math.max(1, Math.min(sourceHeight, Math.round(sourceWidth / targetAspect)));
  const top = variant === "top"
    ? 0
    : Math.max(0, Math.round((sourceHeight - height) / 2));
  return {
    left: 0,
    top,
    width: sourceWidth,
    height,
  };
}

export function computeNormalizedTensorFromRgbBuffer(
  buffer: ArrayLike<number>,
  config: RuntimeModelConfig,
): Float32Array {
  const pixelCount = config.inputSize * config.inputSize;
  const tensor = new Float32Array(config.channels * pixelCount);

  for (let pixelIndex = 0; pixelIndex < pixelCount; pixelIndex += 1) {
    const offset = pixelIndex * 3;
    tensor[pixelIndex] = (buffer[offset] / 255 - config.mean[0]) / config.std[0];
    tensor[pixelCount + pixelIndex] = (buffer[offset + 1] / 255 - config.mean[1]) / config.std[1];
    tensor[pixelCount * 2 + pixelIndex] = (buffer[offset + 2] / 255 - config.mean[2]) / config.std[2];
  }

  return tensor;
}

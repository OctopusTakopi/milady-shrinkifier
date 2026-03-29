import sharp from "sharp";

import {
  computeCoverCropRegion,
  computeNormalizedTensorFromRgbBuffer,
  runtimeModelShape,
  type CropVariant,
  type RuntimeModelConfig,
} from "./model-config";
import type { RuntimeImageFeatures } from "./runtime-image-types";

export async function computeNodeImageFeatures(
  buffer: Buffer,
  config: RuntimeModelConfig,
  variant: CropVariant = "center",
): Promise<RuntimeImageFeatures> {
  const metadata = await sharp(buffer).metadata();
  if (!metadata.width || !metadata.height) {
    throw new Error("Unable to read image dimensions for classifier preprocessing");
  }
  const region = computeCoverCropRegion(
    metadata.width,
    metadata.height,
    config.inputSize,
    config.inputSize,
    variant,
  );
  const classifierRaw = await sharp(buffer)
    .extract({
      left: region.left,
      top: region.top,
      width: region.width,
      height: region.height,
    })
    .resize(config.inputSize, config.inputSize, {
      fit: "fill",
      kernel: "lanczos3",
    })
    .raw()
    .toBuffer();

  return {
    modelTensor: computeNormalizedTensorFromRgbBuffer(classifierRaw, config),
    modelShape: runtimeModelShape(config),
  };
}

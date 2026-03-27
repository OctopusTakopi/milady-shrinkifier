import { mkdir, readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";

import { computeNodeImageFeatures } from "../src/shared/node-image";
import type { HashDatabase } from "../src/shared/types";

const TOTAL_TOKENS = 10_000;
const CONCURRENCY = 16;
const IMAGE_DIR = resolve("cache/milady-maker");
const OUTPUT_DIR = resolve("public/generated");
const OUTPUT_PATH = resolve(OUTPUT_DIR, "milady-maker.hashes.json");

async function main(): Promise<void> {
  await mkdir(OUTPUT_DIR, { recursive: true });

  const tokens = Array.from({ length: TOTAL_TOKENS }, (_, index) => index);
  const hashes: HashDatabase["hashes"] = [];
  const skippedTokenIds: number[] = [];

  for (let offset = 0; offset < tokens.length; offset += CONCURRENCY) {
    const slice = tokens.slice(offset, offset + CONCURRENCY);
    const results = await Promise.all(slice.map((tokenId) => processToken(tokenId)));
    for (const result of results) {
      hashes.push(...result.entries);
      skippedTokenIds.push(...result.skippedTokenIds);
    }
    if ((offset + slice.length) % 256 === 0 || offset + slice.length === tokens.length) {
      console.log(`processed ${offset + slice.length}/${tokens.length}`);
    }
  }

  const database: HashDatabase = {
    collection: "milady-maker",
    algorithm: "dhash64-rgbavg-32x32-center-and-top-crop",
    generatedAt: new Date().toISOString(),
    hashes,
    skippedTokenIds,
  };

  await writeFile(OUTPUT_PATH, JSON.stringify(database));
  console.log(`wrote ${OUTPUT_PATH}`);
}

async function processToken(tokenId: number) {
  const imagePath = resolve(IMAGE_DIR, `${tokenId}.png`);
  let buffer: Buffer;
  try {
    buffer = await readFile(imagePath);
  } catch {
    return {
      entries: [],
      skippedTokenIds: [tokenId],
    };
  }

  if (buffer.length === 0) {
    return {
      entries: [],
      skippedTokenIds: [tokenId],
    };
  }

  const variants = await Promise.all([
    computeNodeImageFeatures(buffer, "center"),
    computeNodeImageFeatures(buffer, "top"),
  ]);

  return {
    entries: variants.map((features, index) => ({
      tokenId,
      variant: index === 0 ? "center" : "top",
      hash: features.hash,
      averageColor: features.averageColor,
    })),
    skippedTokenIds: [],
  };
}

void main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});

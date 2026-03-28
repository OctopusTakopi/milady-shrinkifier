import { access, mkdir, stat, writeFile } from "node:fs/promises";
import { constants } from "node:fs";
import { resolve } from "node:path";

const TOTAL_TOKENS = 10_000;
const CONCURRENCY = 16;
const OUTPUT_DIR = resolve("cache/collections/milady-maker");
const HOSTS = ["https://www.miladymaker.net", "https://miladymaker.net"];
const MAX_RETRIES = 4;

async function main(): Promise<void> {
  await mkdir(OUTPUT_DIR, { recursive: true });

  const tokens = Array.from({ length: TOTAL_TOKENS }, (_, index) => index);
  const failures: number[] = [];
  for (let offset = 0; offset < tokens.length; offset += CONCURRENCY) {
    const slice = tokens.slice(offset, offset + CONCURRENCY);
    const results = await Promise.all(slice.map((tokenId) => downloadToken(tokenId)));
    failures.push(...results.filter((tokenId): tokenId is number => tokenId !== null));
    const completed = offset + slice.length;
    if (completed % 240 === 0 || completed === TOTAL_TOKENS) {
      console.log(`processed ${completed}/${TOTAL_TOKENS}`);
    }
  }

  if (failures.length > 0) {
    console.warn(`failed to download ${failures.length} token(s): ${failures.join(", ")}`);
  }
}

async function downloadToken(tokenId: number): Promise<number | null> {
  const destination = resolve(OUTPUT_DIR, `${tokenId}.png`);
  if (await fileExists(destination)) {
    return null;
  }

  for (let attempt = 0; attempt < MAX_RETRIES; attempt += 1) {
    for (const host of HOSTS) {
      const imageUrl = `${host}/milady/${tokenId}.png`;
      try {
        const response = await fetch(imageUrl);
        if (!response.ok) {
          continue;
        }

        const buffer = Buffer.from(await response.arrayBuffer());
        if (buffer.length === 0) {
          continue;
        }
        await writeFile(destination, buffer);
        return null;
      } catch (error) {
        if (attempt === MAX_RETRIES - 1 && host === HOSTS[HOSTS.length - 1]) {
          console.warn(`download failed for token ${tokenId}: ${String(error)}`);
        }
      }
    }

    await sleep(250 * (attempt + 1));
  }

  return tokenId;
}

async function fileExists(path: string): Promise<boolean> {
  try {
    await access(path, constants.F_OK);
    const info = await stat(path);
    return info.size > 0;
  } catch {
    return false;
  }
}

function sleep(durationMs: number): Promise<void> {
  return new Promise((resolve) => {
    setTimeout(resolve, durationMs);
  });
}

void main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});

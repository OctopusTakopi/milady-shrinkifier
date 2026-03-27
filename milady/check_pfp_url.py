from __future__ import annotations

import argparse
import json
import re
from io import BytesIO
from pathlib import Path

import httpx
import torch
from PIL import Image
from torchvision import transforms

from .mobilenet_common import MODEL_IMAGE_SIZE, MODEL_MEAN, MODEL_STD, create_model
from .pipeline_common import MODEL_RUN_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score a single profile image URL with the current Milady classifier.")
    parser.add_argument("url", help="Profile image URL to fetch and score.")
    parser.add_argument("--run-id", help="Training run id under cache/models/mobilenet_v3_small/. Defaults to the newest run with a summary.")
    parser.add_argument("--threshold", type=float, default=None, help="Override the decision threshold.")
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout in seconds.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_id = args.run_id or resolve_latest_run_id()
    if not run_id:
        raise SystemExit("No classifier runs found. Train a classifier first.")

    run_dir = MODEL_RUN_ROOT / run_id
    checkpoint_path = run_dir / "best.pt"
    summary_path = run_dir / "summary.json"
    if not checkpoint_path.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")
    if not summary_path.exists():
        raise SystemExit(f"Training summary not found: {summary_path}")

    summary = json.loads(summary_path.read_text())
    threshold = float(args.threshold if args.threshold is not None else summary["threshold"])
    normalized_url = normalize_profile_image_url(args.url)

    model = create_model(pretrained=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()

    response = httpx.get(normalized_url, timeout=args.timeout, follow_redirects=True)
    response.raise_for_status()
    probability = infer_probability(model, response.content)

    print(
        json.dumps(
            {
                "url": args.url,
                "normalizedUrl": normalized_url,
                "runId": run_id,
                "threshold": threshold,
                "probability": probability,
                "matched": probability >= threshold,
            },
            indent=2,
            sort_keys=True,
        )
    )


def resolve_latest_run_id() -> str | None:
    if not MODEL_RUN_ROOT.exists():
        return None
    candidates = [
        path.name
        for path in sorted(MODEL_RUN_ROOT.iterdir(), key=lambda path: path.stat().st_mtime, reverse=True)
        if path.is_dir() and (path / "summary.json").exists() and (path / "best.pt").exists()
    ]
    return candidates[0] if candidates else None


def normalize_profile_image_url(url: str) -> str:
    return re.sub(
        r"_(normal|bigger|mini|reasonably_small|(?:\d+x\d+)|(?:x\d+))(\.[a-z0-9]+)$",
        r"_400x400\2",
        url,
        flags=re.IGNORECASE,
    )


def infer_probability(model: torch.nn.Module, image_bytes: bytes) -> float:
    with Image.open(BytesIO(image_bytes)) as image:
        prepared = image.convert("RGB").resize((MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE), Image.Resampling.BICUBIC)
    tensor = transforms.ToTensor()(prepared)
    normalized = transforms.Normalize(mean=MODEL_MEAN, std=MODEL_STD)(tensor).unsqueeze(0)
    with torch.no_grad():
        probability = torch.softmax(model(normalized), dim=1)[0, 1]
    return float(probability.item())


if __name__ == "__main__":
    main()

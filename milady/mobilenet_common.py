from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

MODEL_IMAGE_SIZE = 128
MODEL_MEAN = [0.485, 0.456, 0.406]
MODEL_STD = [0.229, 0.224, 0.225]
POSITIVE_LABEL = "milady"
NEGATIVE_LABEL = "not_milady"
CLASS_NAMES = [NEGATIVE_LABEL, POSITIVE_LABEL]
POSITIVE_INDEX = 1
SPLIT_SEED = 1337


@dataclass(slots=True)
class DatasetEntry:
    sample_id: str
    path: Path
    label: str
    source: str
    split: str


class AvatarDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, entries: list[DatasetEntry], training: bool) -> None:
        self.entries = entries
        self.training = training
        self.to_tensor = transforms.Compose(
            [
                transforms.Resize((MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=MODEL_MEAN, std=MODEL_STD),
            ]
        )

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        entry = self.entries[index]
        with Image.open(entry.path) as image:
            prepared = image.convert("RGB")
            if self.training:
                prepared = apply_training_augment(prepared)
            tensor = self.to_tensor(prepared)
        label_index = POSITIVE_INDEX if entry.label == POSITIVE_LABEL else 0
        return tensor, label_index


def create_model(pretrained: bool = True) -> nn.Module:
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, len(CLASS_NAMES))
    return model


class ExportWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = self.model(inputs)
        return self.softmax(logits)


def apply_training_augment(image: Image.Image) -> Image.Image:
    square = ImageOps.fit(image, (160, 160), method=Image.Resampling.BICUBIC, centering=(0.5, 0.4))
    crop_size = random.randint(116, 154)
    max_offset = 160 - crop_size
    offset_x = random.randint(0, max_offset)
    max_top_offset = max(1, math.floor(max_offset * 0.55))
    offset_y = random.randint(0, max_top_offset)
    cropped = square.crop((offset_x, offset_y, offset_x + crop_size, offset_y + crop_size))
    augmented = cropped.resize((MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE), Image.Resampling.BICUBIC)

    if random.random() < 0.35:
        enhanced = ImageEnhance.Brightness(augmented).enhance(random.uniform(0.9, 1.12))
        augmented = ImageEnhance.Contrast(enhanced).enhance(random.uniform(0.9, 1.12))
    if random.random() < 0.25:
        augmented = ImageEnhance.Color(augmented).enhance(random.uniform(0.92, 1.08))
    if random.random() < 0.2:
        augmented = augmented.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.8)))
    if random.random() < 0.25:
        buffer = BytesIO()
        augmented.save(buffer, format="JPEG", quality=random.randint(52, 86))
        buffer.seek(0)
        augmented = Image.open(buffer).convert("RGB")
    if random.random() < 0.18:
        augmented = apply_circle_edge_artifact(augmented)

    return augmented


def apply_circle_edge_artifact(image: Image.Image) -> Image.Image:
    size = image.size[0]
    mask = Image.new("L", (size, size), 0)
    mask_draw = Image.new("L", (size, size), 0)
    inner = int(size * 0.94)
    offset = (size - inner) // 2
    circle = Image.new("L", (inner, inner), 255)
    mask_draw.paste(circle, (offset, offset))
    blurred = mask_draw.filter(ImageFilter.GaussianBlur(radius=size * 0.03))
    base = Image.new("RGB", image.size, (255, 255, 255))
    return Image.composite(image, base, blurred)


def load_dataset_entries(path: Path) -> list[DatasetEntry]:
    entries: list[DatasetEntry] = []
    if not path.exists():
        return entries
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        entries.append(
            DatasetEntry(
                sample_id=str(payload["id"]),
                path=Path(str(payload["path"])),
                label=str(payload["label"]),
                source=str(payload["source"]),
                split=str(payload["split"]),
            )
        )
    return entries


def deterministic_split_ids(ids: Iterable[str], ratios: tuple[float, float, float]) -> dict[str, str]:
    train_ratio, val_ratio, _ = ratios
    shuffled = list(ids)
    rng = random.Random(SPLIT_SEED)
    rng.shuffle(shuffled)
    total = len(shuffled)
    train_cutoff = int(total * train_ratio)
    val_cutoff = train_cutoff + int(total * val_ratio)
    assignments: dict[str, str] = {}
    for index, sample_id in enumerate(shuffled):
        if index < train_cutoff:
            assignments[sample_id] = "train"
        elif index < val_cutoff:
            assignments[sample_id] = "val"
        else:
            assignments[sample_id] = "test"
    return assignments


def compute_metrics(probabilities: list[float], labels: list[int], threshold: float) -> dict[str, float]:
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0

    for probability, label in zip(probabilities, labels, strict=True):
        predicted = 1 if probability >= threshold else 0
        if predicted == 1 and label == 1:
            true_positive += 1
        elif predicted == 1 and label == 0:
            false_positive += 1
        elif predicted == 0 and label == 0:
            true_negative += 1
        else:
            false_negative += 1

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
    accuracy = (true_positive + true_negative) / max(1, len(labels))
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "truePositive": float(true_positive),
        "falsePositive": float(false_positive),
        "trueNegative": float(true_negative),
        "falseNegative": float(false_negative),
    }


def choose_threshold(probabilities: list[float], labels: list[int], precision_floor: float) -> tuple[float, dict[str, float]]:
    if not probabilities:
        return 0.995, compute_metrics(probabilities, labels, 0.995)

    candidates = sorted({0.0, 1.0, *probabilities}, reverse=True)
    best_threshold = candidates[0]
    best_metrics = compute_metrics(probabilities, labels, best_threshold)
    best_recall = -1.0

    for threshold in candidates:
        metrics = compute_metrics(probabilities, labels, threshold)
        if metrics["precision"] >= precision_floor and metrics["recall"] >= best_recall:
            best_threshold = threshold
            best_metrics = metrics
            best_recall = metrics["recall"]

    if best_recall < 0:
        best_threshold = max(candidates)
        best_metrics = max((compute_metrics(probabilities, labels, threshold) for threshold in candidates), key=lambda metrics: metrics["precision"])

    return float(best_threshold), best_metrics


def score_logits_to_probabilities(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=1)[:, POSITIVE_INDEX]


def dataset_entries_to_jsonl(entries: list[DatasetEntry], path: Path) -> None:
    lines = [
        json.dumps(
            {
                "id": entry.sample_id,
                "path": str(entry.path),
                "label": entry.label,
                "source": entry.source,
                "split": entry.split,
            }
        )
        for entry in entries
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def load_image_for_inference(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        prepared = image.convert("RGB").resize((MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE), Image.Resampling.BICUBIC)
    tensor = transforms.ToTensor()(prepared)
    normalized = transforms.Normalize(mean=MODEL_MEAN, std=MODEL_STD)(tensor)
    return normalized.unsqueeze(0)


def probabilities_from_model(model: nn.Module, paths: list[Path], device: torch.device) -> np.ndarray:
    model.eval()
    tensors = torch.cat([load_image_for_inference(path) for path in paths], dim=0).to(device)
    with torch.no_grad():
        logits = model(tensors)
        probabilities = score_logits_to_probabilities(logits)
    return probabilities.detach().cpu().numpy()

import argparse
import csv
import json
import os
import random
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models import get_model
from validate import (
    MEAN,
    STD,
    check_all_paths_exist,
    gaussian_blur,
    png2jpg,
    read_images_in_dir,
)


SEED = 42
GENIMAGE_STYLE_DATASETS = {
    "GenImage",
    "AIGCDetectionBenchmark",
    "ForenSynth",
    "AIGIBench",
}


def collate_images_filter_none(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return torch.empty(0)
    return torch.stack(batch)


@dataclass
class ModelSpec:
    name: str
    arch: str
    ckpt: str
    crop_size: int
    lora_rank: int
    lora_alpha: float
    threshold: float
    use_resize: bool


class EvalImageDataset(Dataset):
    def __init__(
        self,
        file_paths: List[str],
        arch: str,
        crop_size: int,
        jpeg: Optional[int] = None,
        resize: Optional[float] = None,
        blur: Optional[float] = None,
        is_genimage_fake: bool = False,
        is_crop: bool = True,
    ):
        self.file_paths = file_paths
        self.jpeg = jpeg
        self.resize = resize
        self.blur = blur
        self.is_genimage_fake = is_genimage_fake
        self.is_crop = is_crop
        stat_from = "imagenet" if arch.lower().startswith("imagenet") else "clip"
        crop_func = (
            transforms.CenterCrop(crop_size)
            if is_crop
            else transforms.Resize((crop_size, crop_size))
        )
        self.transform = transforms.Compose(
            [
                crop_func,
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN[stat_from], std=STD[stat_from]),
            ]
        )

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        try:
            img = Image.open(img_path).convert("RGB")

            if self.jpeg is None and self.is_genimage_fake:
                img = png2jpg(img, 96)

            if self.jpeg is not None and self.jpeg != 100:
                img = png2jpg(img, self.jpeg)

            if self.resize is not None:
                w, h = img.size
                new_w = max(1, int(w * self.resize))
                new_h = max(1, int(h * self.resize))
                img = img.resize((new_w, new_h))

            if self.blur is not None:
                img = gaussian_blur(img, self.blur)

            img = self.transform(img)
            return img
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            return None


def set_seed(seed: int = SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def read_and_filter_paths(data_path: str, max_sample: int) -> List[str]:
    if not data_path or not os.path.exists(data_path):
        return []

    must_contain = (
        "_0.jpg" if "EvalGEN" in data_path and "GPT-4o" not in data_path else ""
    )
    paths = sorted(read_images_in_dir(data_path, must_contain=must_contain))

    if max_sample is not None and max_sample > 0 and len(paths) > max_sample:
        rng = random.Random(SEED)
        rng.shuffle(paths)
        paths = paths[:max_sample]

    return paths


def compute_avg_acc(
    real_acc: Optional[float], fake_acc: Optional[float]
) -> Optional[float]:
    values = [v for v in [real_acc, fake_acc] if v is not None]
    if not values:
        return None
    return float(sum(values) / len(values))


def format_metric(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def build_model_specs(args) -> List[ModelSpec]:
    return [
        ModelSpec(
            name="sd",
            arch=args.sd_arch,
            ckpt=args.sd_ckpt,
            crop_size=args.sd_crop_size,
            lora_rank=args.sd_lora_rank,
            lora_alpha=args.sd_lora_alpha,
            threshold=args.sd_threshold,
            use_resize=args.sd_is_resize,
        ),
        ModelSpec(
            name="flux",
            arch=args.flux_arch,
            ckpt=args.flux_ckpt,
            crop_size=args.flux_crop_size,
            lora_rank=args.flux_lora_rank,
            lora_alpha=args.flux_lora_alpha,
            threshold=args.flux_threshold,
            use_resize=args.flux_is_resize,
        ),
        ModelSpec(
            name="clip",
            arch=args.clip_arch,
            ckpt=args.clip_ckpt,
            crop_size=args.clip_crop_size,
            lora_rank=args.clip_lora_rank,
            lora_alpha=args.clip_lora_alpha,
            threshold=args.clip_threshold,
            use_resize=args.clip_is_resize,
        ),
    ]


def load_model(spec: ModelSpec, gpu_id: int):
    model = get_model(spec.arch, lora_rank=spec.lora_rank, lora_alpha=spec.lora_alpha)
    state_dict = torch.load(spec.ckpt, map_location="cpu")["model"]
    model.load_state_dict(state_dict)
    model.eval()
    model.cuda(gpu_id)
    return model


def load_models(model_specs: List[ModelSpec], gpu_id: int):
    loaded_models = OrderedDict()
    for spec in model_specs:
        print(f"\nLoading model once: {spec.name} ({spec.arch}) from {spec.ckpt}")
        loaded_models[spec.name] = load_model(spec, gpu_id)
    return loaded_models


def predict_scores(
    model,
    spec: ModelSpec,
    file_paths: List[str],
    gpu_id: int,
    batch_size: int,
    jpeg: Optional[int],
    resize: Optional[float],
    blur: Optional[float],
    is_genimage_fake: bool,
) -> List[float]:
    if not file_paths:
        return []

    dataset = EvalImageDataset(
        file_paths=file_paths,
        arch=spec.arch,
        crop_size=spec.crop_size,
        jpeg=jpeg,
        resize=resize,
        blur=blur,
        is_genimage_fake=is_genimage_fake,
        is_crop=not spec.use_resize,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate_images_filter_none,
    )

    scores: List[float] = []
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, tuple):
                images = batch[0]
            else:
                images = batch
            if images.numel() == 0:
                continue
            images = images.cuda(gpu_id, non_blocking=True)
            batch_scores = model(images).sigmoid().flatten().tolist()
            scores.extend(float(x) for x in batch_scores)

    return scores


def fuse_and_score(label: int, score_lists: List[List[float]], thresholds: List[float]):
    if not score_lists or any(len(scores) == 0 for scores in score_lists):
        return None, 0

    total = min(len(scores) for scores in score_lists)
    if total == 0:
        return None, 0

    correct = 0
    for idx in range(total):
        is_fake = any(
            score_lists[m][idx] > thresholds[m] for m in range(len(score_lists))
        )
        if label == 1 and is_fake:
            correct += 1
        if label == 0 and not is_fake:
            correct += 1
    return correct / total, total


def extract_eval_entries(dataset_configs: Dict) -> List[Dict[str, Optional[str]]]:
    entries: List[Dict[str, Optional[str]]] = []

    def add_entry(
        dataset_name: str,
        subset_name: str,
        real_path: Optional[str],
        fake_path: Optional[str],
    ):
        entries.append(
            {
                "dataset": dataset_name,
                "subset": subset_name,
                "real_path": real_path,
                "fake_path": fake_path,
            }
        )

    shared_real_datasets = ["DRCT", "synbuster", "synthwildx"]
    fake_only_datasets = ["GenEval", "FullAligned", "Any-Single-Fake"]
    standard_datasets = [
        "GenImage",
        "AIGCDetectionBenchmark",
        "ForenSynth",
        "Chameleon",
        "WildRF",
        "AIGIBench",
        "BFree-Online",
        "AIGI-X",
        "Any",
    ]

    for dataset_name in shared_real_datasets:
        if dataset_name not in dataset_configs:
            continue
        dataset_cfg = dataset_configs[dataset_name]
        real_path = dataset_cfg.get("real")
        for subset_name, subset_cfg in dataset_cfg.items():
            if subset_name == "real":
                continue
            if isinstance(subset_cfg, dict) and "fake" in subset_cfg:
                add_entry(dataset_name, subset_name, real_path, subset_cfg["fake"])

    for dataset_name in standard_datasets:
        if dataset_name not in dataset_configs:
            continue
        dataset_cfg = dataset_configs[dataset_name]
        for subset_name, subset_cfg in dataset_cfg.items():
            if isinstance(subset_cfg, dict):
                add_entry(
                    dataset_name,
                    subset_name,
                    subset_cfg.get("real"),
                    subset_cfg.get("fake"),
                )

    for dataset_name in fake_only_datasets:
        if dataset_name not in dataset_configs:
            continue
        dataset_cfg = dataset_configs[dataset_name]
        for subset_name, subset_cfg in dataset_cfg.items():
            if isinstance(subset_cfg, dict) and "fake" in subset_cfg:
                add_entry(dataset_name, subset_name, None, subset_cfg["fake"])
            elif isinstance(subset_cfg, str):
                add_entry(dataset_name, subset_name, None, subset_cfg)

    return entries


def summarize_dataset_rows(
    subset_rows: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    grouped: "OrderedDict[str, List[Dict[str, object]]]" = OrderedDict()
    for row in subset_rows:
        grouped.setdefault(row["dataset"], []).append(row)

    dataset_rows: List[Dict[str, object]] = []
    for dataset_name, rows in grouped.items():
        real_values = [row["real_acc"] for row in rows if row["real_acc"] is not None]
        fake_values = [row["fake_acc"] for row in rows if row["fake_acc"] is not None]
        real_acc = float(sum(real_values) / len(real_values)) if real_values else None
        fake_acc = float(sum(fake_values) / len(fake_values)) if fake_values else None
        dataset_rows.append(
            {
                "level": "dataset",
                "dataset": dataset_name,
                "subset": "ALL",
                "real_acc": real_acc,
                "fake_acc": fake_acc,
                "avg_acc": compute_avg_acc(real_acc, fake_acc),
                "n_real": sum(int(row["n_real"]) for row in rows),
                "n_fake": sum(int(row["n_fake"]) for row in rows),
            }
        )
    return dataset_rows


def write_results_csv(
    result_csv: str,
    dataset_rows: List[Dict[str, object]],
    subset_rows: List[Dict[str, object]],
):
    with open(result_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "level",
                "dataset",
                "subset",
                "real_acc",
                "fake_acc",
                "avg_acc",
                "n_real",
                "n_fake",
            ]
        )
        for row in dataset_rows:
            writer.writerow(
                [
                    row["level"],
                    row["dataset"],
                    row["subset"],
                    format_metric(row["real_acc"]),
                    format_metric(row["fake_acc"]),
                    format_metric(row["avg_acc"]),
                    row["n_real"],
                    row["n_fake"],
                ]
            )
        for row in subset_rows:
            writer.writerow(
                [
                    row["level"],
                    row["dataset"],
                    row["subset"],
                    format_metric(row["real_acc"]),
                    format_metric(row["fake_acc"]),
                    format_metric(row["avg_acc"]),
                    row["n_real"],
                    row["n_fake"],
                ]
            )


def write_config_summary(output_path: str, args, model_specs: List[ModelSpec]):
    summary = {
        "config": args.config,
        "gpu_id": args.gpu_id,
        "batch_size": args.batch_size,
        "max_sample": args.max_sample,
        "jpeg_quality": args.jpeg_quality,
        "jpeg_override": args.jpeg,
        "resize_override": args.resize,
        "blur_override": args.blur,
        "is_resize": args.is_resize,
        "models": [spec.__dict__ for spec in model_specs],
    }
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config file"
    )
    parser.add_argument(
        "--result_folder",
        type=str,
        required=True,
        help="Output folder for merged evaluation",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--max_sample", type=int, default=-1)
    parser.add_argument("--jpeg_quality", type=int, default=100)
    parser.add_argument("--jpeg", type=int, default=None)
    parser.add_argument("--resize", type=float, default=None)
    parser.add_argument("--blur", type=float, default=None)
    parser.add_argument(
        "--is_resize",
        action="store_true",
        help="Use resize instead of center crop during preprocessing",
    )
    parser.add_argument(
        "--skip_path_check", action="store_true", help="Skip checking if paths exist"
    )

    parser.add_argument("--sd_arch", type=str, default="DINOv3-LoRA:dinov3_vith16plus")
    parser.add_argument("--sd_ckpt", type=str, required=True)
    parser.add_argument("--sd_crop_size", type=int, default=224)
    parser.add_argument("--sd_lora_rank", type=int, default=16)
    parser.add_argument("--sd_lora_alpha", type=float, default=32)
    parser.add_argument("--sd_threshold", type=float, default=0.5)
    parser.add_argument("--sd_is_resize", action="store_true")

    parser.add_argument(
        "--flux_arch", type=str, default="DINOv3-LoRA:dinov3_vith16plus"
    )
    parser.add_argument("--flux_ckpt", type=str, required=True)
    parser.add_argument("--flux_crop_size", type=int, default=224)
    parser.add_argument("--flux_lora_rank", type=int, default=16)
    parser.add_argument("--flux_lora_alpha", type=float, default=32)
    parser.add_argument("--flux_threshold", type=float, default=0.98)
    parser.add_argument("--flux_is_resize", action="store_true")

    parser.add_argument("--clip_arch", type=str, default="CLIP-LoRA:ViT-L/14")
    parser.add_argument("--clip_ckpt", type=str, required=True)
    parser.add_argument("--clip_crop_size", type=int, default=224)
    parser.add_argument("--clip_lora_rank", type=int, default=16)
    parser.add_argument("--clip_lora_alpha", type=float, default=32)
    parser.add_argument("--clip_threshold", type=float, default=0.5)
    parser.add_argument("--clip_is_resize", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed()

    if os.path.exists(args.result_folder):
        import shutil

        shutil.rmtree(args.result_folder)
    os.makedirs(args.result_folder, exist_ok=True)

    with open(args.config, "r") as f:
        dataset_configs = yaml.safe_load(f)

    if not args.skip_path_check:
        print("Checking if all paths in the configuration exist...")
        all_paths_exist, missing_paths = check_all_paths_exist(dataset_configs)
        if not all_paths_exist:
            missing_paths_file = os.path.join(args.result_folder, "missing_paths.txt")
            with open(missing_paths_file, "w") as f:
                f.write("The following paths do not exist:\n")
                for path in missing_paths:
                    f.write(f"{path}\n")
            raise FileNotFoundError(
                f"Missing paths found. Details written to {missing_paths_file}"
            )

    model_specs = build_model_specs(args)
    write_config_summary(
        os.path.join(args.result_folder, "config_summary.json"), args, model_specs
    )
    loaded_models = load_models(model_specs, args.gpu_id)

    subset_rows: List[Dict[str, object]] = []
    entries = extract_eval_entries(dataset_configs)
    thresholds = [spec.threshold for spec in model_specs]

    for entry in tqdm(entries, desc="Evaluating subsets"):
        dataset_name = entry["dataset"]
        subset_name = entry["subset"]
        print(
            "\n" + "-" * 60 + f"\nEvaluating {dataset_name}/{subset_name}\n" + "-" * 60
        )

        real_paths = (
            read_and_filter_paths(entry["real_path"], args.max_sample)
            if entry["real_path"]
            else []
        )
        fake_paths = (
            read_and_filter_paths(entry["fake_path"], args.max_sample)
            if entry["fake_path"]
            else []
        )
        is_genimage_fake = dataset_name in GENIMAGE_STYLE_DATASETS

        real_score_lists: List[List[float]] = []
        fake_score_lists: List[List[float]] = []

        for spec in model_specs:
            model = loaded_models[spec.name]
            real_scores = predict_scores(
                model=model,
                spec=spec,
                file_paths=real_paths,
                gpu_id=args.gpu_id,
                batch_size=args.batch_size,
                jpeg=args.jpeg,
                resize=args.resize,
                blur=args.blur,
                is_genimage_fake=False,
            )
            fake_scores = predict_scores(
                model=model,
                spec=spec,
                file_paths=fake_paths,
                gpu_id=args.gpu_id,
                batch_size=args.batch_size,
                jpeg=args.jpeg,
                resize=args.resize,
                blur=args.blur,
                is_genimage_fake=is_genimage_fake,
            )
            real_score_lists.append(real_scores)
            fake_score_lists.append(fake_scores)

        real_acc, n_real = fuse_and_score(0, real_score_lists, thresholds)
        fake_acc, n_fake = fuse_and_score(1, fake_score_lists, thresholds)
        avg_acc = compute_avg_acc(real_acc, fake_acc)

        print(
            f"{dataset_name}/{subset_name} - real_acc={real_acc}, fake_acc={fake_acc}, avg_acc={avg_acc}, n_real={n_real}, n_fake={n_fake}"
        )

        subset_rows.append(
            {
                "level": "subset",
                "dataset": dataset_name,
                "subset": subset_name,
                "real_acc": real_acc,
                "fake_acc": fake_acc,
                "avg_acc": avg_acc,
                "n_real": n_real,
                "n_fake": n_fake,
            }
        )

    dataset_rows = summarize_dataset_rows(subset_rows)
    result_csv = os.path.join(args.result_folder, "merged_results.csv")
    write_results_csv(result_csv, dataset_rows, subset_rows)
    print(f"Merged evaluation results saved to {result_csv}")


if __name__ == "__main__":
    main()

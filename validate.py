import argparse
import os
import csv
import torch
from torch.utils import data
import torchvision.transforms as transforms
import torch.utils.data
import numpy as np
from sklearn.metrics import average_precision_score, accuracy_score, roc_auc_score
from torch.utils.data import Dataset
from PIL import Image
from copy import deepcopy
import random
import shutil
from scipy.ndimage.filters import gaussian_filter
import yaml
import time
from collections import defaultdict
import pprint

from data.custom_transforms import RandomScaleCropOrDirect224

JPEG = None
RESIZE = None
BLUR = None


def gaussian_blur(img, sigma):
    """Apply Gaussian blur to an image"""
    img = np.array(img)
    gaussian_filter(img[:, :, 0], output=img[:, :, 0], sigma=sigma)
    gaussian_filter(img[:, :, 1], output=img[:, :, 1], sigma=sigma)
    gaussian_filter(img[:, :, 2], output=img[:, :, 2], sigma=sigma)
    return Image.fromarray(img)


SEED = 42


def set_seed():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


# Constants for normalization
MEAN = {
    "imagenet": [0.485, 0.456, 0.406],
    "clip": [0.48145466, 0.4578275, 0.40821073],
    "pe": [0.5, 0.5, 0.5],
}

STD = {
    "imagenet": [0.229, 0.224, 0.225],
    "clip": [0.26862954, 0.26130258, 0.27577711],
    "pe": [0.5, 0.5, 0.5],
}


def get_stat_from_arch(arch):
    if arch.startswith("PE:"):
        return "pe"
    if arch.lower().startswith("imagenet"):
        return "imagenet"
    return "clip"


def find_best_threshold(y_true, y_pred):
    """Find the best threshold to separate real and fake predictions"""
    N = y_true.shape[0]

    if y_pred[0 : N // 2].max() <= y_pred[N // 2 : N].min():  # perfectly separable case
        return (y_pred[0 : N // 2].max() + y_pred[N // 2 : N].min()) / 2

    best_acc = 0
    best_thres = 0
    for thres in y_pred:
        temp = deepcopy(y_pred)
        temp[temp >= thres] = 1
        temp[temp < thres] = 0
        acc = (temp == y_true).sum() / N
        if acc >= best_acc:
            best_thres = thres
            best_acc = acc
    return best_thres


def png2jpg(img, quality):
    """Convert PNG to JPG with specified quality"""
    from io import BytesIO

    out = BytesIO()
    img.save(out, format="jpeg", quality=quality)
    img = Image.open(out)
    img = np.array(img)
    out.close()
    return Image.fromarray(img)


def calculate_acc(y_true, y_pred, thres):
    """Calculate accuracy metrics"""
    r_acc = accuracy_score(y_true[y_true == 0], y_pred[y_true == 0] > thres)
    f_acc = accuracy_score(y_true[y_true == 1], y_pred[y_true == 1] > thres)
    acc = accuracy_score(y_true, y_pred > thres)
    return r_acc, f_acc, acc


def read_images_in_dir(
    rootdir, must_contain="", exts=("png", "jpg", "jpeg", "bmp", "webp", "jfif")
):
    """Recursively read images under rootdir matching extensions and substring"""
    exts = tuple(e.lower().lstrip(".") for e in exts)
    out = []
    if not os.path.exists(rootdir):
        return out
    for dirpath, _, filenames in os.walk(rootdir, followlinks=False):
        for fname in filenames:
            f = os.path.join(dirpath, fname)
            if fname.lower().endswith(exts) and (must_contain in f):
                out.append(f)
    return out


def get_list(path, must_contain=""):
    """Get list of image paths"""
    if not os.path.exists(path):
        print(f"Warning: Path {path} does not exist")
        return []
    return read_images_in_dir(path, must_contain)


def collate_fn_filter_none(batch):
    """Custom collate function that filters out None values from batch"""
    batch = [b for b in batch if b is not None]

    if len(batch) == 0:
        return torch.empty(0, 3, 224, 224), torch.empty(0, dtype=torch.int64)

    # Image dataset: (image, label)
    images, labels = zip(*batch)
    images = torch.stack(images)
    labels = torch.tensor(labels)
    return images, labels


def print_separator(title=None):
    """Print a separator line with optional title"""
    line = "=" * 80
    if title:
        print(f"\n{line}\n{title}\n{line}")
    else:
        print(f"\n{line}\n")

    """Dataset for real/fake image classification"""


class RealFakeDataset(Dataset):
    def __init__(
        self,
        data_path,
        label,
        max_sample,
        arch,
        jpeg_quality=None,
        gaussian_sigma=None,
        resolution_thres=None,
        is_genimage_fake=False,
        crop_size=336,
        is_crop=True,
    ):
        self.jpeg_quality = jpeg_quality
        self.gaussian_sigma = gaussian_sigma
        self.resolution_thres = resolution_thres
        self.is_genimage_fake = is_genimage_fake

        self.crop_size = crop_size
        self.is_crop = is_crop

        print(f"CROP_SIZE:{crop_size}")

        # Load data paths
        if isinstance(data_path, str):
            self.data_list = self.read_path(data_path, max_sample)
        else:
            print(f"Error loading {data_path}")
            self.data_list = []

        self.label = label

        # Set up image transforms
        stat_from = get_stat_from_arch(arch)
        crop_func = (
            transforms.CenterCrop(crop_size)
            if is_crop
            else transforms.Resize((crop_size, crop_size))
        )
        self.transform = transforms.Compose(
            [
                # RandomScaleCropOrDirect224(
                #     train=False,  # 切到推理
                #     infer_policy="short256_center",
                #     eval_resize_short=512,
                # ),
                crop_func,
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN[stat_from], std=STD[stat_from]),
            ]
        )

    def is_problematic_image(self, img):
        """Check if image is pure black/white or has very low variance"""
        try:
            img_array = np.array(img)
            # Check if image is too small
            if img_array.shape[0] < 10 or img_array.shape[1] < 10:
                return True

            # Check if image has low variance (pure color)
            r_var = np.var(img_array[:, :, 0])
            g_var = np.var(img_array[:, :, 1])
            b_var = np.var(img_array[:, :, 2])

            total_var = r_var + g_var + b_var
            if total_var < 10.0:
                return True

            # Check if image is mostly black or white
            mean_value = np.mean(img_array)
            if (mean_value < 5 or mean_value > 250) and np.var(img_array) < 100:
                return True

            return False
        except Exception as e:
            return True  # If any error occurs, consider it problematic

    def read_path(self, data_path, max_sample):
        """Read and filter image paths"""
        if "EvalGEN" in data_path and "GPT-4o" not in data_path:
            data_list = get_list(data_path, must_contain="_0.jpg")
        else:
            data_list = get_list(data_path, must_contain="")

        # Filter by resolution if needed
        if self.resolution_thres is not None:
            filtered_list = []
            for img_path in data_list:
                try:
                    with Image.open(img_path) as img:
                        w, h = img.size
                        area = w * h
                        if self.resolution_thres[0] <= area <= self.resolution_thres[1]:
                            filtered_list.append(img_path)
                except Exception as e:
                    print(f"Error processing {img_path}: {str(e)}")
            data_list = filtered_list

        if len(data_list) == 0:
            print(f"Warning: After filtering, image count is 0 for path {data_path}")
            return []

        # Sample if needed
        if max_sample is not None and max_sample > 0:
            if max_sample > len(data_list):
                print(f"Not enough images, max_sample set to {len(data_list)}")
            random.shuffle(data_list)
            data_list = data_list[: min(max_sample, len(data_list))]

        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        img_path, label = self.data_list[idx], self.label

        try:
            img = Image.open(img_path).convert("RGB")

            # Check if image is problematic
            if self.is_problematic_image(img):
                print(f"Problematic image detected: {img_path}")
                return None

            if JPEG is None and (self.is_genimage_fake and self.label == 1):
                img = png2jpg(img, 96)

            # Apply JPEG compression for GenImage datasets
            if JPEG is not None:
                if JPEG != 100:
                    img = png2jpg(img, JPEG)

            if RESIZE is not None:
                w, h = img.size
                new_w = int(w * RESIZE)
                new_h = int(h * RESIZE)
                img = img.resize((new_w, new_h))

            if BLUR is not None:
                img = gaussian_blur(img, BLUR)

            img = self.transform(img)

            return img, label

        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            return None


def validate(
    model,
    loader,
    find_thres=False,
    gpu_id=None,
    save_incorrect=False,
    save_dir=None,
    save_path_dir=None,
    save_score_dir=None,
):
    """Validate model on a dataset"""
    with torch.no_grad():
        y_true, y_pred = [], []
        print(f"Length of dataset: {len(loader.dataset)}")

        # Get image paths if saving incorrect predictions
        if save_incorrect or (save_path_dir is not None):
            paths = (
                loader.dataset.data_list if hasattr(loader.dataset, "data_list") else []
            )

        img_idx = 0
        incorrect_fake_indices = []
        fake_prediction_paths = []
        real_prediction_paths = []

        for batch_data in loader:
            img, label = batch_data

            # Process batch
            in_tens = img.cuda(gpu_id) if gpu_id is not None else img.cuda()
            batch_preds = model(in_tens).sigmoid().flatten().tolist()
            batch_size = len(batch_preds)

            # Track incorrect predictions for image datasets
            if save_incorrect:
                for i in range(batch_size):
                    if (
                        label[i] == 1 and batch_preds[i] < 0.5
                    ):  # Fake classified as real
                        incorrect_fake_indices.append(img_idx + i)

            # Save paths of all images classified as fake if requested
            if save_path_dir is not None:
                for i in range(batch_size):
                    if batch_preds[i] >= 0.5:
                        try:
                            img_path = paths[img_idx + i]
                            fake_prediction_paths.append(img_path)
                        except IndexError:
                            pass
                    elif batch_preds[i] < 0.5:
                        try:
                            img_path = paths[img_idx + i]
                            real_prediction_paths.append(img_path)
                        except IndexError:
                            pass

            img_idx += batch_size

            y_pred.extend(batch_preds)
            y_true.extend(label.flatten().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    # Calculate metrics
    ap = average_precision_score(y_true, y_pred)
    r_acc0, f_acc0, acc0 = calculate_acc(y_true, y_pred, 0.5)

    # Save misclassified fake images if requested
    if save_incorrect and len(incorrect_fake_indices) > 0 and len(paths) > 0:
        os.makedirs(save_dir, exist_ok=True)
        if len(incorrect_fake_indices) > 10:
            incorrect_fake_indices = incorrect_fake_indices[:]

        # Create a CSV file to log the misclassifications
        csv_path = os.path.join(save_dir, "misclassified_fake_results.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image_path", "true_label", "prediction_score"])

            for idx in incorrect_fake_indices:
                img_path = paths[idx]
                score = y_pred[idx]
                # Save the image
                try:
                    filename = os.path.basename(img_path)
                    dest_path = os.path.join(save_dir, filename)
                    shutil.copy(img_path, dest_path)
                    writer.writerow([img_path, 1, score])
                except Exception as e:
                    print(f"Error saving misclassified image {img_path}: {str(e)}")

        print(f"Misclassified fake images saved to {save_dir}")
        print(f"Log file saved to {csv_path}")

    # save path of all images which are classified as fake
    if save_path_dir is not None:
        os.makedirs(save_path_dir, exist_ok=True)

        fake_txt_path = os.path.join(save_path_dir, "fake_predictions.txt")
        real_txt_path = os.path.join(save_path_dir, "real_predictions.txt")
        with open(fake_txt_path, "a") as f:
            for path in fake_prediction_paths:
                f.write(f"{path}\n")
        print(
            f"Fake prediction paths saved to {fake_txt_path} ({len(fake_prediction_paths)} images)"
        )
        with open(real_txt_path, "a") as f:
            for path in real_prediction_paths:
                f.write(f"{path}\n")
        print(
            f"Real prediction paths saved to {real_txt_path} ({len(real_prediction_paths)} images)"
        )

    # === Save per-image scores (new feature) ===
    if save_score_dir is not None:
        save_score_root = os.path.dirname(save_score_dir)
        os.makedirs(save_score_root, exist_ok=True)

        json_name = os.path.basename(save_score_dir).replace("/", "_") + ".json"
        json_path = os.path.join(save_score_root, json_name)

        if hasattr(loader.dataset, "data_list"):
            file_names = [os.path.basename(p) for p in loader.dataset.data_list]
            scores = {fn: float(sc) for fn, sc in zip(file_names, y_pred)}

            import json

            with open(json_path, "w") as jf:
                json.dump(scores, jf, indent=2)

            print(f"[Score Saved] {json_path}")

    if not find_thres:
        return ap, r_acc0, f_acc0, acc0, y_true, y_pred

    # Calculate metrics with best threshold
    best_thres = find_best_threshold(y_true, y_pred)
    r_acc1, f_acc1, acc1 = calculate_acc(y_true, y_pred, best_thres)

    return ap, r_acc0, f_acc0, acc0, r_acc1, f_acc1, acc1, best_thres


def check_all_paths_exist(config):
    """Check if all paths in the configuration exist"""
    missing_paths = []

    def check_path(path, path_type, dataset, subdataset=None):
        if not os.path.exists(path):
            location = f"{dataset}/{subdataset}" if subdataset else dataset
            missing_paths.append(f"{path_type} path for {location}: {path}")
            return False
        return True

    # Check paths for each dataset type
    datasets = [
        "DRCT",
        "GenImage",
        "Chameleon",
        "GenEval",
        "synbuster",
        "synthwildx",
        "WildRF",
        "FullAligned",
        "BFree-Online",
        "AIGIBench",
        "AIGI-X",
        "Any-Single-Fake",
        "Any",
    ]

    for dataset in datasets:
        if dataset not in config:
            continue

        dataset_config = config[dataset]

        # Handle DRCT special case with common real path
        if dataset in ["DRCT", "synbuster", "synthwildx"] and "real" in dataset_config:
            check_path(dataset_config["real"], "Real", dataset)

            for sub_dataset, sub_config in dataset_config.items():
                if (
                    sub_dataset != "real"
                    and isinstance(sub_config, dict)
                    and "fake" in sub_config
                ):
                    check_path(sub_config["fake"], "Fake", dataset, sub_dataset)

        # Handle GenEval and FullAligned special case with only fake paths
        elif dataset in ["GenEval", "WildRF", "FullAligned", "Any-Single-Fake"]:
            for sub_dataset, fake_path in dataset_config.items():
                if isinstance(fake_path, str):
                    check_path(fake_path, "Fake", dataset, sub_dataset)
                elif isinstance(fake_path, dict) and "fake" in fake_path:
                    check_path(fake_path["fake"], "Fake", dataset, sub_dataset)

        # Handle standard datasets with real and fake paths per subdataset
        else:
            for sub_dataset, sub_config in dataset_config.items():
                if isinstance(sub_config, dict):
                    if "real" in sub_config:
                        check_path(sub_config["real"], "Real", dataset, sub_dataset)
                    if "fake" in sub_config:
                        check_path(sub_config["fake"], "Fake", dataset, sub_dataset)

    return len(missing_paths) == 0, missing_paths


def validate_and_save_results(
    model,
    config_path,
    result_file,
    iteration=None,
    epoch=None,
    max_sample=500,
    batch_size=128,
    gpu_id=0,
    save_bad_case=False,
    jpeg_quality=None,
    gaussian_sigma=None,
    resolution_thres=None,
    arch="res50",
    cropSize=336,
    is_crop=True,
):
    """
    Validates model on datasets defined in config file and saves results
    """
    # Load config from YAML file
    with open(config_path, "r") as f:
        dataset_configs = yaml.safe_load(f)

    # Dictionary to store results
    results_dict = {}
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    # Create result directory if it doesn't exist
    result_dir = os.path.dirname(result_file)
    os.makedirs(result_dir, exist_ok=True)

    # Process dataset validation
    def process_dataset(
        dataset_name, sub_dataset, real_path, fake_path, cropSize=cropSize, is_crop=True
    ):
        print(f"\n{'-' * 50}")
        print(f"Evaluating {dataset_name}/{sub_dataset} (Image)")
        print(f"{'-' * 50}")

        # Create real dataset if provided
        if real_path:
            real_dataset = RealFakeDataset(
                data_path=real_path,
                label=0,  # Real images have label 0
                max_sample=max_sample,
                arch=arch,
                jpeg_quality=jpeg_quality,
                gaussian_sigma=gaussian_sigma,
                resolution_thres=resolution_thres,
                crop_size=cropSize,
                is_crop=is_crop,
            )

            # Skip if not enough data
            if len(real_dataset) == 0:
                print(f"Warning: No real data found for {dataset_name}/{sub_dataset}")
                r_acc_real = None
            else:
                # Validate real dataset
                real_loader = torch.utils.data.DataLoader(
                    real_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=8,
                    collate_fn=collate_fn_filter_none,
                )

                # === Prepare score folder for this dataset ===
                save_path_dir = os.path.join(
                    result_dir, f"prediction_results/paths/{dataset_name}_{sub_dataset}"
                )
                save_score_dir = os.path.join(
                    result_dir,
                    f"prediction_results/scores/{dataset_name}_{sub_dataset}/real",
                )

                ap_real, r_acc_real, f_acc_real, acc_real, y_true_real, y_pred_real = (
                    validate(
                        model,
                        real_loader,
                        find_thres=False,
                        gpu_id=gpu_id,
                        save_path_dir=save_path_dir,
                        save_score_dir=save_score_dir,
                    )
                )
        else:
            # If no real path, set accuracy to None
            r_acc_real = None
            y_true_real, y_pred_real = None, None

        # Create and validate fake dataset if provided
        if fake_path:
            is_genimage_fake = dataset_name in [
                "GenImage",
                "AIGCDetectionBenchmark",
                "ForenSynth",
                "AIGIBench",
            ]

            fake_dataset = RealFakeDataset(
                data_path=fake_path,
                label=1,  # Fake images have label 1
                max_sample=max_sample,
                arch=arch,
                jpeg_quality=jpeg_quality,
                gaussian_sigma=gaussian_sigma,
                resolution_thres=resolution_thres,
                is_genimage_fake=is_genimage_fake,
                crop_size=cropSize,
                is_crop=is_crop,
            )

            # Skip if not enough data
            if len(fake_dataset) == 0:
                print(f"Warning: No fake data found for {dataset_name}/{sub_dataset}")
                f_acc_fake = None
            else:
                # Set up bad case saving directory if needed
                save_dir = None
                if save_bad_case:
                    save_dir = os.path.join(
                        result_dir, f"bad_case/{dataset_name}_{sub_dataset}"
                    )

                # Validate fake dataset
                fake_loader = torch.utils.data.DataLoader(
                    fake_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=8,
                    collate_fn=collate_fn_filter_none,
                )

                save_path_dir = os.path.join(
                    result_dir, f"prediction_results/paths/{dataset_name}_{sub_dataset}"
                )
                save_score_dir = os.path.join(
                    result_dir,
                    f"prediction_results/scores/{dataset_name}_{sub_dataset}/fake",
                )

                ap_fake, r_acc_fake, f_acc_fake, acc_fake, y_true_fake, y_pred_fake = (
                    validate(
                        model,
                        fake_loader,
                        find_thres=False,
                        gpu_id=gpu_id,
                        save_incorrect=save_bad_case,
                        save_dir=save_dir,
                        save_path_dir=save_path_dir,
                        save_score_dir=save_score_dir,
                    )
                )
        else:
            # If no fake path, set accuracy to None
            f_acc_fake = None
            y_true_fake, y_pred_fake = None, None

        # Calculate average accuracy if both real and fake are available
        if r_acc_real is not None and f_acc_fake is not None:
            avg_acc = (r_acc_real + f_acc_fake) / 2
            print(
                f"{dataset_name}/{sub_dataset} - Real Acc: {r_acc_real:.4f}, Fake Acc: {f_acc_fake:.4f}, Avg: {avg_acc:.4f}"
            )
        elif r_acc_real is not None:
            avg_acc = None  # Don't calculate average if only real is available
            print(f"{dataset_name}/{sub_dataset} - Real Acc: {r_acc_real:.4f}")
        elif f_acc_fake is not None:
            avg_acc = None  # Don't calculate average if only fake is available
            print(f"{dataset_name}/{sub_dataset} - Fake Acc: {f_acc_fake:.4f}")
        else:
            avg_acc = None
            print(f"{dataset_name}/{sub_dataset} - No accuracy data")

        return (r_acc_real, f_acc_fake, avg_acc, None, None)

    # Process datasets according to config

    # Process DRCT dataset
    if "DRCT" in dataset_configs:
        drct_config = dataset_configs["DRCT"]
        real_path = drct_config.get("real")

        for sub_dataset, sub_path in drct_config.items():
            if sub_dataset == "real":
                continue  # Skip the real path entry
            if isinstance(sub_path, dict) and "fake" in sub_path:
                fake_path = sub_path["fake"]
                results = process_dataset(
                    "DRCT", sub_dataset, real_path, fake_path, is_crop=is_crop
                )
                if results:
                    dataset_key = f"DRCT-{sub_dataset}"
                    results_dict[dataset_key] = results

    # Process synbuster dataset
    if "synbuster" in dataset_configs:
        synbuster_config = dataset_configs["synbuster"]
        real_path = synbuster_config.get("real")

        for sub_dataset, sub_path in synbuster_config.items():
            if sub_dataset == "real":
                continue  # Skip the real path entry
            if isinstance(sub_path, dict) and "fake" in sub_path:
                fake_path = sub_path["fake"]
                results = process_dataset(
                    "synbuster", sub_dataset, real_path, fake_path, is_crop=is_crop
                )
                if results:
                    dataset_key = f"synbuster-{sub_dataset}"
                    results_dict[dataset_key] = results

    # Process synthwildx dataset
    if "synthwildx" in dataset_configs:
        synthwildx_config = dataset_configs["synthwildx"]
        real_path = synthwildx_config.get("real")

        for sub_dataset, sub_path in synthwildx_config.items():
            if sub_dataset == "real":
                continue  # Skip the real path entry
            if isinstance(sub_path, dict) and "fake" in sub_path:
                fake_path = sub_path["fake"]
                results = process_dataset(
                    "synthwildx", sub_dataset, real_path, fake_path, is_crop=is_crop
                )
                if results:
                    dataset_key = f"synthwildx-{sub_dataset}"
                    results_dict[dataset_key] = results

    # Process GenImage dataset
    if "GenImage" in dataset_configs:
        genimage_config = dataset_configs["GenImage"]

        for sub_dataset, sub_config in genimage_config.items():
            if isinstance(sub_config, dict):
                real_path = sub_config.get("real")
                fake_path = sub_config.get("fake")
                results = process_dataset(
                    "GenImage", sub_dataset, real_path, fake_path, is_crop=is_crop
                )
                if results:
                    dataset_key = f"GenImage-{sub_dataset}"
                    results_dict[dataset_key] = results

    # Process AIGCDetectionBenchmark dataset
    if "AIGCDetectionBenchmark" in dataset_configs:
        genimage_config = dataset_configs["AIGCDetectionBenchmark"]

        for sub_dataset, sub_config in genimage_config.items():
            if isinstance(sub_config, dict):
                real_path = sub_config.get("real")
                fake_path = sub_config.get("fake")
                results = process_dataset(
                    "AIGCDetectionBenchmark",
                    sub_dataset,
                    real_path,
                    fake_path,
                    is_crop=is_crop,
                )
                if results:
                    dataset_key = f"AIGCDetectionBenchmark-{sub_dataset}"
                    results_dict[dataset_key] = results

    if "ForenSynth" in dataset_configs:
        genimage_config = dataset_configs["ForenSynth"]

        for sub_dataset, sub_config in genimage_config.items():
            if isinstance(sub_config, dict):
                real_path = sub_config.get("real")
                fake_path = sub_config.get("fake")
                results = process_dataset(
                    "ForenSynth", sub_dataset, real_path, fake_path, is_crop=is_crop
                )
                if results:
                    dataset_key = f"ForenSynth-{sub_dataset}"
                    results_dict[dataset_key] = results

    # Process Chameleon dataset
    if "Chameleon" in dataset_configs:
        chameleon_config = dataset_configs["Chameleon"]

        for sub_dataset, sub_config in chameleon_config.items():
            if isinstance(sub_config, dict):
                real_path = sub_config.get("real")
                fake_path = sub_config.get("fake")
                results = process_dataset(
                    "Chameleon", sub_dataset, real_path, fake_path, is_crop=is_crop
                )
                if results:
                    dataset_key = f"Chameleon-{sub_dataset}"
                    results_dict[dataset_key] = results

    # Process GenEval dataset
    if "GenEval" in dataset_configs:
        geneval_config = dataset_configs["GenEval"]
        for sub_dataset, sub_path in geneval_config.items():
            if isinstance(sub_path, dict) and "fake" in sub_path:
                fake_path = sub_path["fake"]
                results = process_dataset(
                    "GenEval", sub_dataset, None, fake_path, is_crop=is_crop
                )
                if results:
                    dataset_key = f"GenEval-{sub_dataset}"
                    results_dict[dataset_key] = results

    # Process WildRF dataset
    if "WildRF" in dataset_configs:
        wildrf_config = dataset_configs["WildRF"]
        for sub_dataset, sub_config in wildrf_config.items():
            if isinstance(sub_config, dict):
                real_path = sub_config.get("real")
                fake_path = sub_config.get("fake")
                results = process_dataset(
                    "WildRF", sub_dataset, real_path, fake_path, is_crop=is_crop
                )
                if results:
                    dataset_key = f"WildRF-{sub_dataset}"
                    results_dict[dataset_key] = results

    # Process FullAligned dataset
    if "FullAligned" in dataset_configs:
        fullaligned_config = dataset_configs["FullAligned"]
        for sub_dataset, sub_path in fullaligned_config.items():
            if isinstance(sub_path, dict) and "fake" in sub_path:
                fake_path = sub_path["fake"]
                results = process_dataset(
                    "FullAligned", sub_dataset, None, fake_path, is_crop=is_crop
                )
                if results:
                    dataset_key = f"FullAligned-{sub_dataset}"
                    results_dict[dataset_key] = results
            elif isinstance(sub_path, str):  # Direct path to fake data
                fake_path = sub_path
                results = process_dataset(
                    "FullAligned", sub_dataset, None, fake_path, is_crop=is_crop
                )
                if results:
                    dataset_key = f"FullAligned-{sub_dataset}"
                    results_dict[dataset_key] = results

    if "AIGIBench" in dataset_configs:
        any_config = dataset_configs["AIGIBench"]

        for sub_dataset, sub_config in any_config.items():
            if isinstance(sub_config, dict):
                real_path = sub_config.get("real")
                fake_path = sub_config.get("fake")
                results = process_dataset(
                    "AIGIBench", sub_dataset, real_path, fake_path, is_crop=is_crop
                )
                if results:
                    dataset_key = f"AIGIBench-{sub_dataset}"
                    results_dict[dataset_key] = results

    if "BFree-Online" in dataset_configs:
        any_config = dataset_configs["BFree-Online"]

        for sub_dataset, sub_config in any_config.items():
            if isinstance(sub_config, dict):
                real_path = sub_config.get("real")
                fake_path = sub_config.get("fake")
                results = process_dataset(
                    "BFree-Online", sub_dataset, real_path, fake_path, is_crop=is_crop
                )
                if results:
                    dataset_key = f"BFree-Online-{sub_dataset}"
                    results_dict[dataset_key] = results

    if "AIGI-X" in dataset_configs:
        any_config = dataset_configs["AIGI-X"]

        for sub_dataset, sub_config in any_config.items():
            if isinstance(sub_config, dict):
                real_path = sub_config.get("real")
                fake_path = sub_config.get("fake")
                results = process_dataset(
                    "AIGI-X", sub_dataset, real_path, fake_path, is_crop=is_crop
                )
                if results:
                    dataset_key = f"AIGI-X-{sub_dataset}"
                    results_dict[dataset_key] = results

    if "Any" in dataset_configs:
        any_config = dataset_configs["Any"]

        for sub_dataset, sub_config in any_config.items():
            if isinstance(sub_config, dict):
                real_path = sub_config.get("real")
                fake_path = sub_config.get("fake")
                results = process_dataset(
                    "Any", sub_dataset, real_path, fake_path, is_crop=is_crop
                )
                if results:
                    dataset_key = f"Any-{sub_dataset}"
                    results_dict[dataset_key] = results

    if "Any-Single-Fake" in dataset_configs:
        any_single_fake_config = dataset_configs["Any-Single-Fake"]
        for sub_dataset, sub_path in any_single_fake_config.items():
            if isinstance(sub_path, dict) and "fake" in sub_path:
                fake_path = sub_path["fake"]
                results = process_dataset(
                    "Any-Single-Fake", sub_dataset, None, fake_path, is_crop=is_crop
                )
                if results:
                    dataset_key = f"Any-Single-Fake-{sub_dataset}"
                    results_dict[dataset_key] = results

    # Get sorted dataset keys for consistent column order
    dataset_keys = list(results_dict.keys())

    # Check if results file exists, create with header if not
    file_exists = os.path.isfile(result_file)

    # Save results to CSV file with transposed format (metrics as rows)
    with open(result_file, "a", newline="") as f:
        writer = csv.writer(f)

        # Write header if file is new
        if not file_exists:
            header = ["Timestamp", "Iteration", "Epoch", "MetricType"] + dataset_keys
            writer.writerow(header)

        # Write real accuracy row
        real_row = [
            timestamp,
            iteration if iteration is not None else "",
            epoch if epoch is not None else "",
            "RealAcc",
        ]
        for key in dataset_keys:
            r_acc, _, _, _, _ = results_dict[key]
            real_row.append(r_acc if r_acc is not None else "")
        writer.writerow(real_row)

        # Write fake accuracy row
        fake_row = [
            timestamp,
            iteration if iteration is not None else "",
            epoch if epoch is not None else "",
            "FakeAcc",
        ]
        for key in dataset_keys:
            _, f_acc, _, _, _ = results_dict[key]
            fake_row.append(f_acc if f_acc is not None else "")
        writer.writerow(fake_row)

        # Write average accuracy row
        avg_row = [
            timestamp,
            iteration if iteration is not None else "",
            epoch if epoch is not None else "",
            "AvgAcc",
        ]
        for key in dataset_keys:
            _, _, avg_acc, _, _ = results_dict[key]
            # Only write average if we have both real and fake accuracies
            if avg_acc is not None:
                avg_row.append(avg_acc)
            else:
                # Calculate average for datasets with both metrics
                r_acc, f_acc, _, _, _ = results_dict[key]
                if r_acc is not None and f_acc is not None:
                    avg_row.append((r_acc + f_acc) / 2)
                else:
                    avg_row.append("")
        writer.writerow(avg_row)

        ap_row = [
            timestamp,
            iteration if iteration is not None else "",
            epoch if epoch is not None else "",
            "AP",
        ]
        for key in dataset_keys:
            _, _, _, ap, _ = results_dict[key]
            # Only write average if we have both real and fake accuracies
            if ap is not None:
                ap_row.append(ap)
            else:
                ap_row.append("")
        writer.writerow(ap_row)

        auroc_row = [
            timestamp,
            iteration if iteration is not None else "",
            epoch if epoch is not None else "",
            "AUROC",
        ]
        for key in dataset_keys:
            _, _, _, _, auroc = results_dict[key]
            # Only write average if we have both real and fake accuracies
            if ap is not None:
                auroc_row.append(auroc)
            else:
                auroc_row.append("")
        writer.writerow(auroc_row)

    print(f"Results saved to {result_file}")

    return results_dict


def main():
    """Main function for model validation"""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config file"
    )
    parser.add_argument(
        "--max_sample",
        type=int,
        default=-1,
        help="Only check this number of images for both fake/real",
    )
    parser.add_argument("--arch", type=str, default="res50")
    parser.add_argument(
        "--ckpt", type=str, default="./pretrained_weights/fc_weights.pth"
    )
    parser.add_argument("--result_folder", type=str, default="./result")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument(
        "--jpeg_quality", type=int, default=None, help="100-30. Used to test robustness"
    )
    parser.add_argument(
        "--gaussian_sigma", type=int, default=None, help="0-4. Used to test robustness"
    )
    parser.add_argument("--resolution_thres", type=int, nargs=2, default=None)
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--save_bad_case", action="store_true")
    parser.add_argument(
        "--skip_path_check", action="store_true", help="Skip checking if paths exist"
    )
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank")
    parser.add_argument(
        "--lora_alpha", type=float, default=1.0, help="LoRA scaling factor"
    )
    parser.add_argument(
        "--lora_targets", type=str, default=None, help="LoRA trainable targets"
    )

    parser.add_argument("--jpeg", type=int, default=None)
    parser.add_argument("--resize", type=float, default=None)
    parser.add_argument("--blur", type=float, default=None)

    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument(
        "--is_resize",
        action="store_true",
        help="Whether to resize images during preprocessing",
    )

    opt = parser.parse_args()

    global JPEG, RESIZE, BLUR
    JPEG = opt.jpeg
    RESIZE = opt.resize
    BLUR = opt.blur

    # Create result directory
    if os.path.exists(opt.result_folder):
        shutil.rmtree(opt.result_folder)
    os.makedirs(opt.result_folder, exist_ok=True)

    # Load config
    with open(opt.config, "r") as f:
        dataset_configs = yaml.safe_load(f)

    # Check paths if required
    if not opt.skip_path_check:
        print("Checking if all paths in the configuration exist...")
        all_paths_exist, missing_paths = check_all_paths_exist(dataset_configs)

        if not all_paths_exist:
            print("\n===== MISSING PATHS =====")
            for path in missing_paths:
                print(path)

            # Write missing paths to file
            missing_paths_file = os.path.join(opt.result_folder, "missing_paths.txt")
            with open(missing_paths_file, "w") as f:
                f.write("The following paths don't exist:\n")
                for path in missing_paths:
                    f.write(f"{path}\n")

            print(f"\nMissing paths written to {missing_paths_file}")
            print("Exiting due to missing paths. Use --skip_path_check to run anyway.")
            import sys

            sys.exit(1)
        else:
            print("All paths in the configuration exist.")

    # Load model
    from models import get_model

    model = get_model(opt.arch, lora_rank=opt.lora_rank, lora_alpha=opt.lora_alpha)
    state_dict = torch.load(opt.ckpt, map_location="cpu")["model"]
    model.load_state_dict(state_dict)
    print("Model loaded..")
    model.eval()
    model.cuda(opt.gpu_id)

    # Set random seed for reproducibility
    set_seed()

    # Use the validate_and_save_results function to process all datasets
    result_file = os.path.join(opt.result_folder, "results.csv")
    results_dict = validate_and_save_results(
        model=model,
        config_path=opt.config,
        result_file=result_file,
        max_sample=opt.max_sample,
        batch_size=opt.batch_size,
        gpu_id=opt.gpu_id,
        save_bad_case=opt.save_bad_case,
        jpeg_quality=opt.jpeg_quality,
        gaussian_sigma=opt.gaussian_sigma,
        resolution_thres=opt.resolution_thres,
        arch=opt.arch,
        cropSize=opt.crop_size,
        is_crop=not opt.is_resize,
    )


if __name__ == "__main__":
    print("Working with validate.py")
    main()

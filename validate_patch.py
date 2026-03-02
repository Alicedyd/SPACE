import argparse
import os
import csv
import torch
import torchvision.transforms as transforms
import torch.utils.data
import numpy as np
from sklearn.metrics import average_precision_score, accuracy_score
from torch.utils.data import Dataset
from PIL import Image 
from copy import deepcopy
import random
import shutil
from scipy.ndimage.filters import gaussian_filter
import yaml
import time
import math
import json
from collections import defaultdict

from scipy.ndimage.filters import gaussian_filter

JPEG = None
RESIZE = None
BLUR = None
CROP_SIZE = 224

def gaussian_blur(img, sigma):
    """Apply Gaussian blur to an image"""
    img = np.array(img)
    gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
    gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
    gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)
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
    "clip": [0.48145466, 0.4578275, 0.40821073]
}

STD = {
    "imagenet": [0.229, 0.224, 0.225],
    "clip": [0.26862954, 0.26130258, 0.27577711]
}

def find_best_threshold(y_true, y_pred):
    """Find the best threshold to separate real and fake predictions"""
    N = y_true.shape[0]

    if y_pred[0:N//2].max() <= y_pred[N//2:N].min():  # perfectly separable case
        return (y_pred[0:N//2].max() + y_pred[N//2:N].min()) / 2 
    
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
    img.save(out, format='jpeg', quality=quality)
    img = Image.open(out)
    img = np.array(img)
    out.close()
    return Image.fromarray(img)

def gaussian_blur(img, sigma):
    """Apply Gaussian blur to an image"""
    img = np.array(img)
    gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
    gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
    gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)
    return Image.fromarray(img)

def calculate_acc(y_true, y_pred, thres):
    """Calculate accuracy metrics"""
    r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > thres)
    f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > thres)
    acc = accuracy_score(y_true, y_pred > thres)
    return r_acc, f_acc, acc

def read_images_in_dir(rootdir, must_contain, exts=["PNG", "png", "jpg", "JPEG", "jpeg", "bmp"]):
    """Read all images in a directory with specified extensions"""
    out = []
    if os.path.exists(rootdir):
        for file in os.listdir(rootdir):
            file_path = os.path.join(rootdir, file)
            if (os.path.isfile(file_path) and 
                any(file.lower().endswith(ext.lower()) for ext in exts) and 
                (must_contain in file_path)):
                out.append(file_path)
    return out

def get_list(path, must_contain=''):
    """Get list of image paths"""
    if not os.path.exists(path):
        print(f"Warning: Path {path} does not exist")
        return []
    return read_images_in_dir(path, must_contain)

# New collate function to handle image patches
def collate_fn_patches(batch):
    """Custom collate function that handles patches and filters out None values"""
    # Filter out None values
    batch = [b for b in batch if b is not None]
    
    if len(batch) == 0:
        return [], [], [], [], []
    
    # Unpack the batch - now each item is (patches, label, image_idx, img_path)
    patches_list, labels, image_indices, img_paths = zip(*batch)
    
    # Flatten patches lists, but keep track of how many patches per image
    flat_patches = []
    patches_per_image = []
    
    for patches in patches_list:
        patches_per_image.append(len(patches))
        flat_patches.extend(patches)
    
    # Stack patches into a single tensor
    if flat_patches:
        flat_patches = torch.stack(flat_patches)
    else:
        flat_patches = torch.empty(0, 3, CROP_SIZE, CROP_SIZE)
    
    # Convert labels to tensor
    labels = torch.tensor(labels)
    image_indices = torch.tensor(image_indices)
    
    return flat_patches, labels, image_indices, patches_per_image, img_paths

def print_separator(title=None):
    """Print a separator line with optional title"""
    line = "=" * 80
    if title:
        print(f"\n{line}\n{title}\n{line}")
    else:
        print(f"\n{line}\n")

class RealFakeDataset(Dataset):
    """Dataset for real/fake image classification with patch-based processing"""
    def __init__(self, data_path, label, max_sample, arch,
                 jpeg_quality=None, gaussian_sigma=None, resolution_thres=None,
                 is_genimage_fake=False, crop_size=CROP_SIZE):
        self.jpeg_quality = jpeg_quality
        self.gaussian_sigma = gaussian_sigma
        self.resolution_thres = resolution_thres
        self.is_genimage_fake = is_genimage_fake  # Flag for GenImage fake data
        self.crop_size = crop_size

        print(f"CROP_SIZE: {crop_size}")
        
        # Load data paths
        if isinstance(data_path, str):
            self.data_list = self.read_path(data_path, max_sample)
        else:
            print(f"Error loading {data_path}")
            self.data_list = []
            
        self.label = label

        # Set up image transforms (without CenterCrop - we'll handle patch extraction separately)
        stat_from = "imagenet" if arch.lower().startswith("imagenet") else "clip"
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN[stat_from], std=STD[stat_from]),
        ])
        
    def is_problematic_image(self, img):
        """Check if image is pure black/white or has very low variance"""
        try:
            img_array = np.array(img)
            # Check if image is too small
            if img_array.shape[0] < 10 or img_array.shape[1] < 10:
                return True
                
            # Check if image has low variance (pure color)
            r_var = np.var(img_array[:,:,0])
            g_var = np.var(img_array[:,:,1])
            b_var = np.var(img_array[:,:,2])
            
            total_var = r_var + g_var + b_var
            if total_var < 10.0:
                return True
            
            # Check if image is mostly black or white
            mean_value = np.mean(img_array)
            if (mean_value < 5 or mean_value > 250) and np.var(img_array) < 100:
                return True
                
            return False
        except:
            return True  # If any error occurs, consider it problematic
        
    def read_path(self, data_path, max_sample):
        """Read and filter image paths"""
        if 'GenEval' in data_path and 'GPT-4o' not in data_path:
            data_list = get_list(data_path, must_contain='_0.png')
        else:
            data_list = get_list(data_path, must_contain='')

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
            data_list = data_list[:min(max_sample, len(data_list))]
            
        return data_list
    
    def create_patches(self, img):
        """Create non-overlapping patches from an image
        If the image dimensions aren't perfect multiples of crop_size,
        the last patches in each row/column will overlap with the previous ones.
        
        Args:
            img: PIL Image object
            
        Returns:
            List of patch tensors
        """
        width, height = img.size
        crop_size = self.crop_size
        
        # Calculate number of complete patches in each dimension
        num_patches_w = math.ceil(width / crop_size)
        num_patches_h = math.ceil(height / crop_size)
        
        patches = []
        
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                # Calculate top-left corner of this patch
                # For the last patch in row/column, we adjust to ensure it's within bounds
                left = min(j * crop_size, max(0, width - crop_size))
                top = min(i * crop_size, max(0, height - crop_size))
                
                # Extract and process the patch
                patch = img.crop((left, top, left + crop_size, top + crop_size))
                patch_tensor = self.transform(patch)
                patches.append(patch_tensor)
                
        return patches

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
                if JPEG != 100 and self.label == 1:
                    img = png2jpg(img, JPEG)
            
            if RESIZE is not None:
                w, h = img.size
                new_w = int(w * RESIZE)
                new_h = int(h * RESIZE)
                img = img.resize((new_w, new_h))
                
            if BLUR is not None:
                img = gaussian_blur(img, BLUR)
            
            # Create multiple patches from the image
            patches = self.create_patches(img)
            
            # Return patches, label, image index, and image path for aggregation in validate function
            return patches, label, idx, img_path
            
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            return None

def validate(model, loader, dataset_name, sub_dataset, result_folder, find_thres=False, gpu_id=None, save_incorrect=False, save_dir=None, patch_batch_size=32):
    """
    Validate model on a dataset using patch-based approach with fixed batch size for patches
    
    Args:
        model: The model to validate
        loader: DataLoader providing batches of images/patches
        dataset_name: Name of the dataset
        sub_dataset: Name of the sub-dataset
        result_folder: Folder to save patch predictions
        find_thres: Whether to find the best threshold
        gpu_id: GPU ID to use (None for default)
        save_incorrect: Whether to save misclassified images
        save_dir: Directory to save misclassified images
        patch_batch_size: Fixed batch size to use for processing patches (default: 32)
    
    Returns:
        Tuple of metrics
    """
    with torch.no_grad():
        # These will store aggregated predictions (one per image)
        image_labels = []
        image_predictions = []
        
        # Store patch predictions for each image and mapping from paths to indices
        patch_predictions = {}
        image_path_to_idx = {}  # Maps image paths to their indices in the dataset
        
        print(f"Length of dataset: {len(loader.dataset)}")
        
        # Get image paths if saving incorrect predictions
        if save_incorrect:
            paths = loader.dataset.data_list if hasattr(loader.dataset, 'data_list') else []
        
        # Process all batches from the loader
        for patches, labels, image_indices, patches_per_image, img_paths in loader:
            if len(patches) == 0:
                continue

            # Map patches to their images
            patch_to_img_mapping = []  # List of (img_path, label, image_idx) for each patch
            start_idx = 0
            for i, num_patches in enumerate(patches_per_image):
                if num_patches > 0:
                    img_path = img_paths[i]
                    label = labels[i].item()
                    image_idx = image_indices[i].item()
                    
                    # Store image index for later reference
                    image_path_to_idx[img_path] = image_idx
                    
                    # Initialize entry for this image if it doesn't exist
                    if img_path not in patch_predictions:
                        patch_predictions[img_path] = {
                            'patches': [],
                            'label': label
                        }
                    
                    # Keep track of which patches belong to which images
                    for p in range(num_patches):
                        patch_to_img_mapping.append((img_path, label, image_idx))
                    
                    start_idx += num_patches
            
            # Process patches in fixed-size batches
            for i in range(0, len(patches), patch_batch_size):
                batch_end = min(i + patch_batch_size, len(patches))
                batch_patches = patches[i:batch_end]
                batch_mappings = patch_to_img_mapping[i:batch_end]
                
                # Process this batch
                batch_patches = batch_patches.cuda(gpu_id) if gpu_id is not None else batch_patches.cuda()
                batch_preds = model(batch_patches).sigmoid().flatten().cpu().numpy()
                
                # Store predictions with their corresponding images
                for j, (img_path, _, _) in enumerate(batch_mappings):
                    patch_pred = batch_preds[j]
                    patch_predictions[img_path]['patches'].append(patch_pred)
        
        # Calculate average prediction for each image and prepare lists for metrics
        incorrect_fake_paths = []
        for img_path, data in patch_predictions.items():
            patches = data['patches']
            label = data['label']
            
            if patches:
                img_pred = np.mean(patches)
            else:
                img_pred = 0.5  # Default if no valid patches
            
            # Store the aggregated prediction
            image_predictions.append(img_pred)
            image_labels.append(label)
            
            # Track incorrect predictions if needed
            if save_incorrect and label == 1 and img_pred < 0.5:
                incorrect_fake_paths.append(img_path)
        
        # Convert to numpy arrays for metrics calculation
        image_labels = np.array(image_labels)
        image_predictions = np.array(image_predictions)
    
    # Calculate metrics
    ap = average_precision_score(image_labels, image_predictions)
    r_acc0, f_acc0, acc0 = calculate_acc(image_labels, image_predictions, 0.5)
    
    # Save patch predictions to a numpy file
    output_dir = os.path.join(result_folder, 'patch_predictions')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a safe filename
    safe_name = f"{dataset_name}_{sub_dataset}"
    safe_name = safe_name.replace('/', '_').replace('\\', '_')
    
    # Save patch predictions
    patch_preds_file = os.path.join(output_dir, f"{safe_name}_patch_preds.npz")
    np.savez(patch_preds_file, patch_predictions=patch_predictions)
    
    print(f"Patch predictions saved to {patch_preds_file}")
    
    # Save misclassified fake images if requested
    if save_incorrect and incorrect_fake_paths:
        os.makedirs(save_dir, exist_ok=True)
        if len(incorrect_fake_paths) > 10:
            incorrect_fake_paths = incorrect_fake_paths[:10]
            
        # Create a CSV file to log the misclassifications
        csv_path = os.path.join(save_dir, "misclassified_fake_results.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['image_path', 'true_label', 'prediction_score'])
            
            for img_path in incorrect_fake_paths:
                score = np.mean(patch_predictions[img_path]['patches'])
                try:
                    filename = os.path.basename(img_path)
                    dest_path = os.path.join(save_dir, filename)
                    shutil.copy(img_path, dest_path)
                    writer.writerow([img_path, 1, score])
                except Exception as e:
                    print(f"Error saving misclassified image {img_path}: {str(e)}")
        
        print(f"Misclassified fake images saved to {save_dir}")
        print(f"Log file saved to {csv_path}")

    if not find_thres:
        return ap, r_acc0, f_acc0, acc0

    # Calculate metrics with best threshold
    best_thres = find_best_threshold(image_labels, image_predictions)
    r_acc1, f_acc1, acc1 = calculate_acc(image_labels, image_predictions, best_thres)

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
    datasets = ['DRCT', 'GenImage', 'Chameleon', 'GenEval', 'synbuster', 'synthwildx',  'WildRF', 'FullAligned']
    
    for dataset in datasets:
        if dataset not in config:
            continue
            
        dataset_config = config[dataset]
        
        # Handle DRCT special case with common real path
        if dataset in ['DRCT','synbuster', 'synthwildx'] and 'real' in dataset_config:
            print(dataset)
            check_path(dataset_config['real'], 'Real', dataset)
            
            for sub_dataset, sub_config in dataset_config.items():
                if sub_dataset != 'real' and isinstance(sub_config, dict) and 'fake' in sub_config:
                    check_path(sub_config['fake'], 'Fake', dataset, sub_dataset)
        
        # Handle GenEval and FullAligned special case with only fake paths
        elif dataset in ['GenEval', 'WildRF', 'FullAligned']:
            for sub_dataset, fake_path in dataset_config.items():
                if isinstance(fake_path, str):
                    check_path(fake_path, 'Fake', dataset, sub_dataset)
                elif isinstance(fake_path, dict) and 'fake' in fake_path:
                    check_path(fake_path['fake'], 'Fake', dataset, sub_dataset)
        
        # Handle standard datasets with real and fake paths per subdataset
        else:
            for sub_dataset, sub_config in dataset_config.items():
                if isinstance(sub_config, dict):
                    if 'real' in sub_config:
                        check_path(sub_config['real'], 'Real', dataset, sub_dataset)
                    if 'fake' in sub_config:
                        check_path(sub_config['fake'], 'Fake', dataset, sub_dataset)
    
    return len(missing_paths) == 0, missing_paths


def process_dataset(model, dataset_name, sub_dataset, real_path, fake_path, max_sample, 
                    batch_size, gpu_id, arch, save_bad_case, save_dir, result_folder,
                    jpeg_quality, gaussian_sigma, resolution_thres, crop_size):
    """Process a single dataset configuration and validate model"""
    print(f"\n{'-' * 50}")
    print(f"Evaluating {dataset_name}/{sub_dataset}")
    print(f"{'-' * 50}")
    
    r_acc_real, f_acc_fake = None, None
    
    # Process real images if path provided
    if real_path:
        real_dataset = RealFakeDataset(
            data_path=real_path,
            label=0,
            max_sample=max_sample,
            arch=arch,
            jpeg_quality=jpeg_quality,
            gaussian_sigma=gaussian_sigma,
            resolution_thres=resolution_thres,
            is_genimage_fake=False,  # Real images are never GenImage fake
            crop_size=crop_size
        )
        
        if len(real_dataset) == 0:
            print(f"Warning: No real images found for {dataset_name}/{sub_dataset}")
        else:
            real_loader = torch.utils.data.DataLoader(
                real_dataset, batch_size=batch_size, shuffle=False,
                num_workers=4, collate_fn=collate_fn_patches
            )
            
            ap_real, r_acc_real, _, _ = validate(
                model, real_loader, dataset_name, f"{sub_dataset}_real", result_folder, 
                find_thres=False, gpu_id=gpu_id, save_incorrect=False
            )
    
    # Process fake images if path provided
    if fake_path:
        # Set the flag for GenImage fake data
        is_genimage_fake = (dataset_name in ['GenImage', 'GenEval', 'AIGCDetectionBenchmark', 'ForenSynth'])
        
        fake_dataset = RealFakeDataset(
            data_path=fake_path,
            label=1,
            max_sample=max_sample,
            arch=arch,
            jpeg_quality=jpeg_quality,
            gaussian_sigma=gaussian_sigma,
            resolution_thres=resolution_thres,
            is_genimage_fake=is_genimage_fake,  # Set the flag for GenImage data
            crop_size=crop_size
        )
        
        if len(fake_dataset) == 0:
            print(f"Warning: No fake images found for {dataset_name}/{sub_dataset}")
        else:
            # Set up save directory for bad cases if needed
            bad_case_save_dir = os.path.join(save_dir, f"bad_case/{dataset_name}_{sub_dataset}") if save_bad_case else None
            
            fake_loader = torch.utils.data.DataLoader(
                fake_dataset, batch_size=batch_size, shuffle=False,
                num_workers=4, collate_fn=collate_fn_patches
            )
            
            ap_fake, _, f_acc_fake, _ = validate(
                model, fake_loader, dataset_name, f"{sub_dataset}_fake", result_folder,
                find_thres=False, gpu_id=gpu_id,
                save_incorrect=save_bad_case, save_dir=bad_case_save_dir
            )
    
    # Calculate average accuracy
    if r_acc_real is not None and f_acc_fake is not None:
        avg_acc = (r_acc_real + f_acc_fake) / 2
        print(f"{dataset_name}/{sub_dataset} - Real Acc: {r_acc_real:.4f}, Fake Acc: {f_acc_fake:.4f}, Avg: {avg_acc:.4f}")
    elif r_acc_real is not None:
        print(f"{dataset_name}/{sub_dataset} - Real Acc: {r_acc_real:.4f}")
    elif f_acc_fake is not None:
        print(f"{dataset_name}/{sub_dataset} - Fake Acc: {f_acc_fake:.4f}")
    else:
        print(f"{dataset_name}/{sub_dataset} - No accuracy data")
    
    return r_acc_real, f_acc_fake

def validate_and_save_results(model, config_path, result_file, result_folder, iteration=None, epoch=None, 
                             max_sample=500, batch_size=128, 
                             gpu_id=0, save_bad_case=False, 
                             jpeg_quality=None, gaussian_sigma=None, resolution_thres=None, 
                             arch='res50', crop_size=CROP_SIZE):
    """
    Validates model on datasets defined in config file and saves results
    
    Args:
        model: The model to validate
        config_path: Path to YAML config file with dataset definitions
        result_file: Path to the CSV file for saving results
        result_folder: Folder to save additional results (like patch predictions)
        iteration: Current training iteration (optional)
        epoch: Current epoch (optional)
        max_sample: Maximum number of samples to test for each dataset
        batch_size: Batch size for testing
        gpu_id: GPU ID to use
        save_bad_case: Whether to save misclassified examples
        jpeg_quality: JPEG quality for robustness testing
        gaussian_sigma: Gaussian sigma for robustness testing
        resolution_thres: Resolution threshold range
        arch: Model architecture name
        crop_size: Size of each crop/patch
    
    Returns:
        Dictionary containing validation results
    """
    # Load config from YAML file
    with open(config_path, 'r') as f:
        dataset_configs = yaml.safe_load(f)
    
    # Dictionary to store results
    results_dict = {}
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    
    # Create result directory if it doesn't exist
    result_dir = os.path.dirname(result_file)
    os.makedirs(result_dir, exist_ok=True)
    
    # Also create a directory for patch predictions
    patch_pred_dir = os.path.join(result_folder, 'patch_predictions')
    os.makedirs(patch_pred_dir, exist_ok=True)
    
    # Process dataset validation
    def process_dataset(dataset_name, sub_dataset, real_path, fake_path):
        print(f"\n{'-' * 50}")
        print(f"Evaluating {dataset_name}/{sub_dataset}")
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
                crop_size=crop_size,
            )
            
            # Skip if not enough images
            if len(real_dataset) == 0:
                print(f"Warning: No real images found for {dataset_name}/{sub_dataset}")
                r_acc_real = None
            else:
                # Validate real dataset
                real_loader = torch.utils.data.DataLoader(
                    real_dataset, batch_size=batch_size, shuffle=False, 
                    num_workers=8, collate_fn=collate_fn_patches
                )
                ap_real, r_acc_real, f_acc_real, acc_real = validate(
                    model, real_loader, dataset_name, f"{sub_dataset}_real", result_folder,
                    find_thres=False, gpu_id=gpu_id, save_incorrect=False
                )
        else:
            # If no real path, set accuracy to None
            r_acc_real = None
        
        # Create and validate fake dataset if provided
        if fake_path:
            is_genimage_fake = (dataset_name in ['GenImage', 'GenEval', 'AIGCDetectionBenchmark', 'ForenSynth'])
            fake_dataset = RealFakeDataset(
                data_path=fake_path,
                label=1,  # Fake images have label 1
                max_sample=max_sample,
                arch=arch,
                jpeg_quality=jpeg_quality,
                gaussian_sigma=gaussian_sigma,
                resolution_thres=resolution_thres,
                is_genimage_fake=is_genimage_fake,
                crop_size=crop_size,
            )
            
            # Skip if not enough images
            if len(fake_dataset) == 0:
                print(f"Warning: No fake images found for {dataset_name}/{sub_dataset}")
                f_acc_fake = None
            else:
                # Set up bad case saving directory if needed
                save_dir = None
                if save_bad_case:
                    save_dir = os.path.join(result_dir, f"bad_case/{dataset_name}_{sub_dataset}")
                
                # Validate fake dataset
                fake_loader = torch.utils.data.DataLoader(
                    fake_dataset, batch_size=batch_size, shuffle=False, 
                    num_workers=8, collate_fn=collate_fn_patches
                )
                ap_fake, r_acc_fake, f_acc_fake, acc_fake = validate(
                    model, fake_loader, dataset_name, f"{sub_dataset}_fake", result_folder,
                    find_thres=False, gpu_id=gpu_id, save_incorrect=save_bad_case, 
                    save_dir=save_dir
                )
        else:
            # If no fake path, set accuracy to None
            f_acc_fake = None
        
        # Calculate average accuracy if both real and fake are available
        if r_acc_real is not None and f_acc_fake is not None:
            avg_acc = (r_acc_real + f_acc_fake) / 2
            print(f"{dataset_name}/{sub_dataset} - Real Acc: {r_acc_real:.4f}, Fake Acc: {f_acc_fake:.4f}, Avg: {avg_acc:.4f}")
        elif r_acc_real is not None:
            avg_acc = None  # Don't calculate average if only real is available
            print(f"{dataset_name}/{sub_dataset} - Real Acc: {r_acc_real:.4f}")
        elif f_acc_fake is not None:
            avg_acc = None  # Don't calculate average if only fake is available
            print(f"{dataset_name}/{sub_dataset} - Fake Acc: {f_acc_fake:.4f}")
        else:
            avg_acc = None
            print(f"{dataset_name}/{sub_dataset} - No accuracy data")
        
        return (r_acc_real, f_acc_fake, avg_acc)
    
    # Process datasets according to config
    
    # Process DRCT dataset
    if 'DRCT' in dataset_configs:
        drct_config = dataset_configs['DRCT']
        real_path = drct_config.get('real')
        
        for sub_dataset, sub_path in drct_config.items():
            if sub_dataset == 'real':
                continue  # Skip the real path entry
            if isinstance(sub_path, dict) and 'fake' in sub_path:
                fake_path = sub_path['fake']
                results = process_dataset('DRCT', sub_dataset, real_path, fake_path)
                if results:
                    dataset_key = f"DRCT-{sub_dataset}"
                    results_dict[dataset_key] = results
    
    # Process synbuster dataset
    if 'synbuster' in dataset_configs:
        synbuster_config = dataset_configs['synbuster']
        real_path = synbuster_config.get('real')
        
        for sub_dataset, sub_path in synbuster_config.items():
            if sub_dataset == 'real':
                continue  # Skip the real path entry
            if isinstance(sub_path, dict) and 'fake' in sub_path:
                fake_path = sub_path['fake']
                results = process_dataset('synbuster', sub_dataset, real_path, fake_path)
                if results:
                    dataset_key = f"synbuster-{sub_dataset}"
                    results_dict[dataset_key] = results
    
    # Process synthwildx dataset
    if 'synthwildx' in dataset_configs:
        synthwildx_config = dataset_configs['synthwildx']
        real_path = synthwildx_config.get('real')
        
        for sub_dataset, sub_path in synthwildx_config.items():
            if sub_dataset == 'real':
                continue  # Skip the real path entry
            if isinstance(sub_path, dict) and 'fake' in sub_path:
                fake_path = sub_path['fake']
                results = process_dataset('synthwildx', sub_dataset, real_path, fake_path)
                if results:
                    dataset_key = f"synthwildx-{sub_dataset}"
                    results_dict[dataset_key] = results
                    
    # Process GenImage dataset
    if 'GenImage' in dataset_configs:
        genimage_config = dataset_configs['GenImage']
        
        for sub_dataset, sub_config in genimage_config.items():
            if isinstance(sub_config, dict):
                real_path = sub_config.get('real')
                fake_path = sub_config.get('fake')
                results = process_dataset('GenImage', sub_dataset, real_path, fake_path)
                if results:
                    dataset_key = f"GenImage-{sub_dataset}"
                    results_dict[dataset_key] = results

    if 'AIGCDetectionBenchmark' in dataset_configs:
        genimage_config = dataset_configs['AIGCDetectionBenchmark']
        
        for sub_dataset, sub_config in genimage_config.items():
            if isinstance(sub_config, dict):
                real_path = sub_config.get('real')
                fake_path = sub_config.get('fake')
                results = process_dataset('AIGCDetectionBenchmark', sub_dataset, real_path, fake_path)
                if results:
                    dataset_key = f"AIGCDetectionBenchmark-{sub_dataset}"
                    results_dict[dataset_key] = results

    if 'ForenSynth' in dataset_configs:
        genimage_config = dataset_configs['ForenSynth']
        
        for sub_dataset, sub_config in genimage_config.items():
            if isinstance(sub_config, dict):
                real_path = sub_config.get('real')
                fake_path = sub_config.get('fake')
                results = process_dataset('ForenSynth', sub_dataset, real_path, fake_path)
                if results:
                    dataset_key = f"ForenSynth-{sub_dataset}"
                    results_dict[dataset_key] = results
    
    # Process Chameleon dataset
    if 'Chameleon' in dataset_configs:
        chameleon_config = dataset_configs['Chameleon']
        
        for sub_dataset, sub_config in chameleon_config.items():
            if isinstance(sub_config, dict):
                real_path = sub_config.get('real')
                fake_path = sub_config.get('fake')
                results = process_dataset('Chameleon', sub_dataset, real_path, fake_path)
                if results:
                    dataset_key = f"Chameleon-{sub_dataset}"
                    results_dict[dataset_key] = results
    
    # Process GenEval dataset
    if 'GenEval' in dataset_configs:
        geneval_config = dataset_configs['GenEval']
        for sub_dataset, sub_path in geneval_config.items():
            if isinstance(sub_path, dict) and 'fake' in sub_path:
                fake_path = sub_path['fake']
                results = process_dataset('GenEval', sub_dataset, None, fake_path)
                if results:
                    dataset_key = f"GenEval-{sub_dataset}"
                    results_dict[dataset_key] = results
    
    # Process WildRF dataset
    if 'WildRF' in dataset_configs:
        wildrf_config = dataset_configs['WildRF']
        for sub_dataset, sub_config in wildrf_config.items():
            if isinstance(sub_config, dict):
                real_path = sub_config.get('real')
                fake_path = sub_config.get('fake')
                results = process_dataset('WildRF', sub_dataset, real_path, fake_path)
                if results:
                    dataset_key = f"WildRF-{sub_dataset}"
                    results_dict[dataset_key] = results
                          
    # Process FullAligned dataset
    if 'FullAligned' in dataset_configs:
        fullaligned_config = dataset_configs['FullAligned']
        for sub_dataset, sub_path in fullaligned_config.items():
            if isinstance(sub_path, dict) and 'fake' in sub_path:
                fake_path = sub_path['fake']
                results = process_dataset('FullAligned', sub_dataset, None, fake_path)
                if results:
                    dataset_key = f"FullAligned-{sub_dataset}"
                    results_dict[dataset_key] = results
            elif isinstance(sub_path, str):  # Direct path to fake data
                fake_path = sub_path
                results = process_dataset('FullAligned', sub_dataset, None, fake_path)
                if results:
                    dataset_key = f"FullAligned-{sub_dataset}"
                    results_dict[dataset_key] = results
    
    
    # Process DEMO dataset
    if 'DEMO' in dataset_configs:
        demo_config = dataset_configs['DEMO']
        for sub_dataset, sub_config in demo_config.items():
            if isinstance(sub_config, dict):
                real_path = sub_config.get('real')
                fake_path = sub_config.get('fake')
                results = process_dataset('DEMO', sub_dataset, real_path, fake_path)
                if results:
                    dataset_key = f"DEMO-{sub_dataset}"
                    results_dict[dataset_key] = results
    
           
    # Get sorted dataset keys for consistent column order
    dataset_keys = list(results_dict.keys())
    
    # Check if results file exists, create with header if not
    file_exists = os.path.isfile(result_file)
    
    # Save results to CSV file with transposed format (metrics as rows)
    with open(result_file, 'a', newline='') as f:
        writer = csv.writer(f)
        
        # Write header if file is new
        if not file_exists:
            header = ['Timestamp', 'Iteration', 'Epoch', 'MetricType'] + dataset_keys
            writer.writerow(header)
        
        # Write real accuracy row
        real_row = [timestamp, iteration if iteration is not None else '', epoch if epoch is not None else '', 'RealAcc']
        for key in dataset_keys:
            r_acc, _, _ = results_dict[key]
            real_row.append(r_acc if r_acc is not None else '')
        writer.writerow(real_row)
        
        # Write fake accuracy row
        fake_row = [timestamp, iteration if iteration is not None else '', epoch if epoch is not None else '', 'FakeAcc']
        for key in dataset_keys:
            _, f_acc, _ = results_dict[key]
            fake_row.append(f_acc if f_acc is not None else '')
        writer.writerow(fake_row)
        
        # Write average accuracy row
        avg_row = [timestamp, iteration if iteration is not None else '', epoch if epoch is not None else '', 'AvgAcc']
        for key in dataset_keys:
            _, _, avg_acc = results_dict[key]
            # Only write average if we have both real and fake accuracies
            if avg_acc is not None:
                avg_row.append(avg_acc)
            else:
                # Calculate average for datasets with both metrics
                r_acc, f_acc, _ = results_dict[key]
                if r_acc is not None and f_acc is not None:
                    avg_row.append((r_acc + f_acc) / 2)
                else:
                    avg_row.append('')
        writer.writerow(avg_row)
    
    print(f"Results saved to {result_file}")
    
    return results_dict


def main():
    """Main function for model validation"""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    parser.add_argument('--max_sample', type=int, default=-1, help='Only check this number of images for both fake/real')
    parser.add_argument('--arch', type=str, default='res50')
    parser.add_argument('--ckpt', type=str, default='./pretrained_weights/fc_weights.pth')
    parser.add_argument('--result_folder', type=str, default='./result')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--jpeg_quality', type=int, default=None, help="100-30. Used to test robustness")
    parser.add_argument('--gaussian_sigma', type=int, default=None, help="0-4. Used to test robustness")
    parser.add_argument('--resolution_thres', type=int, nargs=2, default=None)
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use')
    parser.add_argument('--save_bad_case', action="store_true")
    parser.add_argument('--skip_path_check', action="store_true", help="Skip checking if paths exist")
    parser.add_argument('--lora_rank', type=int, default=8, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=float, default=1.0, help='LoRA scaling factor')
    parser.add_argument('--lora_targets', type=str, default=None, help='LoRA trainable targets')

    parser.add_argument('--jpeg', type=int, default=None)
    parser.add_argument('--resize', type=float, default=None)
    parser.add_argument('--blur', type=float, default=None)

    parser.add_argument('--crop_size', type=int, default=224)

    opt = parser.parse_args()

    global JPEG, RESIZE, BLUR, CROP_SIZE
    JPEG = opt.jpeg
    RESIZE = opt.resize
    BLUR = opt.blur
    CROP_SIZE = opt.crop_size
    
    # Create result directory
    # if os.path.exists(opt.result_folder):
    #     shutil.rmtree(opt.result_folder)
    os.makedirs(opt.result_folder, exist_ok=True)

    # Load config
    with open(opt.config, 'r') as f:
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
            missing_paths_file = os.path.join(opt.result_folder, 'missing_paths.txt')
            with open(missing_paths_file, 'w') as f:
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
    model = get_model(opt.arch)
    state_dict = torch.load(opt.ckpt, map_location='cpu')['model']
    model.load_state_dict(state_dict)
    print("Model loaded..")
    model.eval()
    model.cuda(opt.gpu_id)

    # Set random seed for reproducibility
    set_seed()
    
    # Use the validate_and_save_results function to process all datasets
    result_file = os.path.join(opt.result_folder, 'results.csv')
    results_dict = validate_and_save_results(
        model=model,
        config_path=opt.config,
        result_file=result_file,
        result_folder=opt.result_folder,
        max_sample=opt.max_sample,
        batch_size=opt.batch_size,
        gpu_id=opt.gpu_id,
        save_bad_case=opt.save_bad_case,
        jpeg_quality=opt.jpeg_quality,
        gaussian_sigma=opt.gaussian_sigma,
        resolution_thres=opt.resolution_thres,
        arch=opt.arch,
        crop_size=opt.crop_size
    )
    
    print("\nValidation complete. Patch predictions are saved and can be used for custom aggregation methods.")

# Add a utility to analyze saved patch predictions
def analyze_patch_predictions(patch_preds_file, aggregation_method='mean'):
    """
    Analyze patch predictions with different aggregation methods
    
    Args:
        patch_preds_file: Path to the NPZ file with patch predictions
        aggregation_method: Method to aggregate patch predictions ('mean', 'median', 'max', etc.)
        
    Returns:
        Dictionary with analysis results
    """
    # Load patch predictions
    data = np.load(patch_preds_file, allow_pickle=True)
    patch_predictions = data['patch_predictions'].item()
    
    # Prepare data structures
    image_labels = []
    image_predictions = []
    
    # Aggregate predictions based on the specified method
    for img_path, img_data in patch_predictions.items():
        patch_preds = np.array(img_data['patches'])
        label = img_data['label']
        
        # Apply aggregation method
        if aggregation_method == 'mean':
            img_pred = np.mean(patch_preds)
        elif aggregation_method == 'median':
            img_pred = np.median(patch_preds)
        elif aggregation_method == 'max':
            img_pred = np.max(patch_preds)
        elif aggregation_method == 'min':
            img_pred = np.min(patch_preds)
        elif aggregation_method == 'majority_vote':
            votes = (patch_preds > 0.5).sum()
            img_pred = 1.0 if votes > len(patch_preds) / 2 else 0.0
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")
        
        image_labels.append(label)
        image_predictions.append(img_pred)
    
    # Convert to numpy arrays
    image_labels = np.array(image_labels)
    image_predictions = np.array(image_predictions)
    
    # Calculate metrics
    ap = average_precision_score(image_labels, image_predictions)
    r_acc, f_acc, acc = calculate_acc(image_labels, image_predictions, 0.5)
    
    results = {
        'aggregation_method': aggregation_method,
        'average_precision': ap,
        'real_acc': r_acc,
        'fake_acc': f_acc,
        'overall_acc': acc,
        'num_images': len(image_labels),
        'real_images': (image_labels == 0).sum(),
        'fake_images': (image_labels == 1).sum()
    }
    
    return results

if __name__ == '__main__':
    print("Working with validate_patch.py")
    main()
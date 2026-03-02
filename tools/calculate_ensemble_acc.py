# Example code to analyze with different aggregation methods
import numpy as np
import os
import csv
from sklearn.metrics import accuracy_score

# METHODS = ["MEAN", "MEDIAN", "MAJORITY_VOTE", "WEIGHTED_MEAN", 
#            "MAX", "MIN", "PERCENTILE_75", "PERCENTILE_25", "TRIMMED_MEAN", "HARMONIC_MEAN",
#            "GEOMETRIC_MEAN", "TOP_K_MEAN", "ADAPTIVE_THRESHOLD", "CONFIDENCE_WEIGHTED"]

METHODS = ["HARMONIC_MEAN"]

EMSEMBLE_METHODS = {
    "MEAN": lambda x: np.mean(x),
    "MEDIAN": lambda x: np.median(x),
    "MAJORITY_VOTE": lambda x: 1 if np.sum(x > 0.5) > np.sum(x < 0.5) else 0,
    "WEIGHTED_MEAN": lambda x: np.average(x, weights=np.abs(x - 0.5) + 0.01),  # 权重基于与0.5的距离
    "MAX": lambda x: np.max(x),  # 最大值策略
    "MIN": lambda x: np.min(x),  # 最小值策略
    "PERCENTILE_75": lambda x: np.percentile(x, 75),  # 75%分位数
    "PERCENTILE_25": lambda x: np.percentile(x, 25),  # 25%分位数
    "TRIMMED_MEAN": lambda x: np.mean(np.sort(x)[1:-1] if len(x) > 2 else x),  # 裁剪平均（去掉最大最小值）
    "HARMONIC_MEAN": lambda x: len(x) / np.sum(1 / (x + 1e-6)),  # 调和平均数（添加小常数避免除零）
    "GEOMETRIC_MEAN": lambda x: np.exp(np.mean(np.log(x + 1e-6))),  # 几何平均数
    "TOP_K_MEAN": lambda x: np.mean(np.sort(x)[-3:]) if len(x) >= 3 else np.mean(x),  # 取top-k（这里k=3）预测值的平均
    "ADAPTIVE_THRESHOLD": lambda x: np.mean(x) > (np.max(x) + np.min(x)) / 2,  # 自适应阈值
    "CONFIDENCE_WEIGHTED": lambda x: np.sum(x * np.abs(x - 0.5)) / np.sum(np.abs(x - 0.5) + 1e-6)  # 基于置信度加权
}

datasets = {
    # "DRCT": ["ldm-t2i", "sd14", "sd15", "sd21", "sdxl", "sdxl-refiner", "sd-turbo", "sdxl-turbo", "lcm-sd15", "lcm-sdxl", "sd-cn", "sd21-cn", "sdxl-cn", "sd-inpainting", "sd21-inpainting", "sdxl-inpainting"],
    # "GenImage": ["MidJourney", "sd14", "sd15", "ADM", "GLIDE", "wukong", "VQDM", "BigGAN"],
    # "Chameleon": ["chameleon"],
    # "GenEval": ["Flux", "GoT", "Infinity", "NOVA", "OmniGen"],
    # "FullAligned": ["sdxl-vae", "sd-vae-ft-ema", "sd-vae-ft-mse", "stable-diffusion-2-1", "stable-diffusion-3.5-large"],
    # "synthwildx": ["dalle3", "firefly", "midjourney_v5"],
    # "synbuster": ['dalle2', "dalle3", "firefly", "sdxl", "midjourney-v5"],
    "WildRF": ["facebook", "reddit", "twitter"]
}

# source_dir = "result/align/emsemble/patch_predictions/"
source_dir = "result/dinov3_ensemble/patch_predictions"

# 创建CSV文件存储所有结果
csv_filename = "ensemble_results.csv"
with open(csv_filename, 'w', newline='') as csvfile:
    # 创建CSV写入器并写入表头
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['Dataset', 'Subset', 'Method', 'Real_Acc', 'Fake_Acc', 'Avg_Acc'])
    
    # 第一层循环：数据集
    for dataset, subsets in datasets.items():
        # 创建字典存储每个方法的结果
        method_results = {method: {'real_accs': [], 'fake_accs': [], 'avg_accs': []} for method in METHODS}
        
        # 第二层循环：数据集的子集
        for subset in subsets:
            real_npz = os.path.join(source_dir, f"{dataset}_{subset}_real_patch_preds.npz")
            fake_npz = os.path.join(source_dir, f"{dataset}_{subset}_fake_patch_preds.npz")

            contain_real = os.path.exists(real_npz)

            real_data = np.load(real_npz, allow_pickle=True) if contain_real else None
            fake_data = np.load(fake_npz, allow_pickle=True)

            real_patch_predictions = real_data["patch_predictions"].item() if contain_real else None
            fake_patch_predictions = fake_data["patch_predictions"].item()
            
            # 第三层循环：方法
            for current_method in METHODS:
                emsemble = EMSEMBLE_METHODS[current_method]
                accs = []
                
                for patch_predictions in [real_patch_predictions, fake_patch_predictions]:

                    if patch_predictions is not None:
                        # 准备分析数据
                        labels = []
                        preds = []

                        # 应用聚合方法
                        for img_path, img_data in patch_predictions.items():
                            patch_preds = np.array(img_data['patches'])
                            labels.append(img_data['label'])
                            emsemble_pred = emsemble(patch_preds)
                            preds.append(emsemble_pred)

                        # 计算准确率
                        acc = accuracy_score(labels, np.array(preds) > 0.5)
                        accs.append(acc)
                    else:
                        accs.append(0)

                real_acc = accs[0]
                fake_acc = accs[1]
                avg_acc = (real_acc + fake_acc) / 2 if contain_real else fake_acc
                
                # 存储此方法的结果
                method_results[current_method]['real_accs'].append(real_acc)
                method_results[current_method]['fake_accs'].append(fake_acc)
                method_results[current_method]['avg_accs'].append(avg_acc)
                
                # 将单个结果写入CSV
                csvwriter.writerow([dataset, subset, current_method, f"{real_acc:.4f}", f"{fake_acc:.4f}", f"{avg_acc:.4f}"])
                
                # 打印结果
                print(f"{dataset}/{subset}/{current_method}: Real Acc {real_acc:.4f}, Fake Acc {fake_acc:.4f}, Avg Acc {avg_acc:.4f}")
        
        # 打印并写入每个方法在此数据集上的汇总
        for method in METHODS:
            avg_real_acc = np.mean(method_results[method]['real_accs'])
            avg_fake_acc = np.mean(method_results[method]['fake_accs'])
            avg_acc = np.mean(method_results[method]['avg_accs'])
            
            # 将数据集汇总写入CSV
            csvwriter.writerow([dataset, 'AVERAGE', method, f"{avg_real_acc:.4f}", f"{avg_fake_acc:.4f}", f"{avg_acc:.4f}"])
            
            print(f"\n-- {dataset} / {method}: Real Acc {avg_real_acc:.4f}, Fake Acc {avg_fake_acc:.4f}, Avg Acc {avg_acc:.4f} --\n")

print(f"结果已保存至 {csv_filename}")
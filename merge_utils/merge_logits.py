import os
import json
import pandas as pd
from tqdm import tqdm


def load_json_safe(path):
    """安全读取 JSON；文件不存在则返回空字典。"""
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def evaluate_detectors(detector1_dir, detector2_dir, thresh1, thresh2, save_csv):
    """
    - detector1_dir / detector2_dir: 两个检测器的根目录
      每个子目录名形如 <dataset>_<subset>，内部包含 fake.json / real.json（real.json 可缺失）
    - thresh1 / thresh2: 两个检测器的阈值
    - save_csv: 结果保存路径
    规则：
      - 单检测器：logit > 阈值 → 该检测器判 fake，否则判 real
      - 只有当两个检测器都判 fake 时，最终判 fake；否则判 real
      - 样本 GT 由来源文件决定：出自 fake.json → gt=fake；出自 real.json → gt=real
      - 同名图片若同时出现在 fake/real 中，会当作两个独立样本（因为来源不同）
      - 仅统计两个检测器都对该样本有 logit 的样本；否则跳过
      - real_acc/fake_acc 分别按各自 GT 计算；avg_acc=(real_acc+fake_acc)/2；
        若 real.json 缺失导致 real_acc 不存在，则 avg_acc=fake_acc
    """
    results = []

    # 遍历 detector1_dir 下的子数据集（基于 A 目录做主控）
    for subset_name in tqdm(
        sorted(os.listdir(detector1_dir)), desc="Processing subsets"
    ):
        subset_path_1 = os.path.join(detector1_dir, subset_name)
        subset_path_2 = os.path.join(detector2_dir, subset_name)

        # 两边都得有这个子目录才评估（避免错位）
        if not (os.path.isdir(subset_path_1) and os.path.isdir(subset_path_2)):
            continue

        # 读取两个检测器的 real/fake 结果（缺失 real.json 时得到 {}）
        fake1 = load_json_safe(os.path.join(subset_path_1, "fake.json"))
        real1 = load_json_safe(os.path.join(subset_path_1, "real.json"))
        fake2 = load_json_safe(os.path.join(subset_path_2, "fake.json"))
        real2 = load_json_safe(os.path.join(subset_path_2, "real.json"))

        # 统计量
        correct_fake, total_fake = 0, 0
        correct_real, total_real = 0, 0

        # ============ 第一批：来自 fake.json 的样本（GT = fake） ============
        # 样本集合为两个检测器 fake.json 的并集；要求两个检测器在各自 fake.json 中都有该样本的 logit 才评估
        fake_imgs_union = set(fake1.keys()) | set(fake2.keys())
        for img in fake_imgs_union:
            logit1 = fake1.get(img, None)
            logit2 = fake2.get(img, None)
            if logit1 is None or logit2 is None:
                continue  # 两个检测器都需要有该样本的 logit

            pred1_fake = logit1 > thresh1
            pred2_fake = logit2 > thresh2
            final_pred_fake = pred1_fake or pred2_fake  # 双假才假

            total_fake += 1
            if final_pred_fake:  # GT=fake，预测也为 fake
                correct_fake += 1

        # ============ 第二批：来自 real.json 的样本（GT = real） ============
        # 注意 real.json 可能缺失，这里用并集；依然要求两个检测器在各自 real.json 中都有该样本
        real_imgs_union = set(real1.keys()) | set(real2.keys())
        for img in real_imgs_union:
            logit1 = real1.get(img, None)
            logit2 = real2.get(img, None)
            if logit1 is None or logit2 is None:
                continue  # 两个检测器都需要有该样本的 logit
                print("Warning: Missing logit for real image:", img)

            pred1_fake = logit1 > thresh1
            pred2_fake = logit2 > thresh2
            final_pred_fake = pred1_fake or pred2_fake  # 双假才假

            total_real += 1
            if not final_pred_fake:  # GT=real，预测应为 real
                correct_real += 1

        # 计算各类 ACC
        fake_acc = (correct_fake / total_fake) if total_fake > 0 else None
        real_acc = (correct_real / total_real) if total_real > 0 else None
        if real_acc is None:
            avg_acc = fake_acc
        elif fake_acc is None:
            avg_acc = real_acc
        else:
            avg_acc = (real_acc + fake_acc) / 2

        # 解析 dataset / subset 字段
        if "_" in subset_name:
            dataset_name, subset_short = subset_name.split("_", 1)
        else:
            dataset_name, subset_short = subset_name, subset_name

        results.append(
            {
                "dataset": dataset_name,
                "subset": subset_short,
                "real_acc": real_acc,
                "fake_acc": fake_acc,
                "avg_acc": avg_acc,
            }
        )

    # 汇总 & 排序 & 导出
    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values(by=["dataset", "subset"], kind="mergesort")
    df.to_csv(save_csv, index=False)
    print(f"✅ Detailed results saved to {save_csv}")
    return df


# def get_avg_acc_from_csv(csv_path, save_path):
#     # 读取结果文件
#     df = pd.read_csv(csv_path)

#     # 若存在意外空行或NaN，进行清理
#     df = df.dropna(subset=["dataset", "subset", "avg_acc"])

#     # Step 1️⃣: 对 subset 内部的相同 dataset 做平均（例如 cyclegan-apple, cyclegan-horse）
#     # 先统一 dataset 名称，例如 cyclegan-apple -> cyclegan
#     df["dataset_group"] = df["dataset"].apply(lambda x: x.split("-")[0])

#     # Step 2️⃣: 对每个 dataset_group 下的所有 subset 的 avg_acc 求平均
#     dataset_avg = (
#         df.groupby("dataset_group", as_index=False)["avg_acc"]
#         .mean()
#         .sort_values(by="avg_acc", ascending=False)
#     )

#     # Step 3️⃣: 保存结果
#     dataset_avg.to_csv(save_path, index=False)
#     print(f"✅ 汇总完成，已保存至{save_path}")


def get_avg_acc_from_csv(csv_path, save_path):
    # 定义哪些 subset 前缀要被合并为一个逻辑 subset
    MERGE_GROUPS = {
        # —— CycleGAN 家族：把这些子集先合并为 "cyclegan" ——
        "cyclegan": [
            "cyclegan-apple",
            "cyclegan-horse",
            "cyclegan-orange",
            "cyclegan-summer",
            "cyclegan-winter",
            "cyclegan-zebra",
        ],
        # —— ProGAN 家族（你截图中列出的 VOC20 类别）→ 合并为 "progan" ——
        "progan": [
            "progan-airplane",
            "progan-bicycle",
            "progan-bird",
            "progan-boat",
            "progan-bottle",
            "progan-bus",
            "progan-car",
            "progan-cat",
            "progan-chair",
            "progan-cow",
            "progan-diningtable",
            "progan-dog",
            "progan-horse",
            "progan-motorbike",
            "progan-person",
            "progan-pottedplant",
            "progan-sheep",
            "progan-sofa",
            "progan-train",
            "progan-tvmonitor",
        ],
        # —— StyleGAN（v1）家族 → 合并为 "stylegan" ——
        "stylegan": [
            "stylegan-bedroom",
            "stylegan-car",
            "stylegan-cat",
            # 若还有其它，如 "stylegan-church"、"stylegan-horse" 等，也可继续补充
        ],
        # —— StyleGAN2 家族 → 合并为 "stylegan2" ——
        "stylegan2": [
            "stylegan2-car",
            "stylegan2-cat",
            "stylegan2-church",
            "stylegan2-horse",
            # 如果你还有 "stylegan2-bedroom" 等，也一并加入
        ],
    }

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["dataset", "subset", "avg_acc"])

    # 建立 subset → 合并名 的映射
    subset_to_group = {}
    for merged_name, members in MERGE_GROUPS.items():
        for m in members:
            subset_to_group[m] = merged_name

    # 若 subset 在映射表里，就替换为其合并名
    df["subset_merged"] = df["subset"].apply(lambda x: subset_to_group.get(x, x))

    # 第一步：在 subset 层（同一个 dataset + subset_merged）内做平均
    subset_avg = df.groupby(["dataset", "subset_merged"], as_index=False).agg(
        {
            "real_acc": "mean" if "real_acc" in df.columns else "first",
            "fake_acc": "mean" if "fake_acc" in df.columns else "first",
            "avg_acc": "mean",
        }
    )

    # 第二步：在 dataset 层（同一个 dataset）内再做平均
    dataset_avg = (
        subset_avg.groupby("dataset", as_index=False)
        .agg(
            {
                "real_acc": "mean" if "real_acc" in subset_avg.columns else "first",
                "fake_acc": "mean" if "fake_acc" in subset_avg.columns else "first",
                "avg_acc": "mean",
            }
        )
        .sort_values("avg_acc", ascending=False)
    )

    dataset_avg.to_csv(save_path, index=False)
    print("✅ 数据集平均结果已保存到", save_path)
    # print(dataset_avg.head(10))


# datasets="all_datasets"
# detector1_dir = "/data1/junwei/DDA/result/dda-all-datasets/prediction_results/scores"
# detector2_dir = "/data1/junwei/DDA/result/flux-dda-all-datasets/prediction_results/scores"

# FLUX family
# datasets="flux_family"
# detector1_dir = "/data1/junwei/DDA/result/dda-flux-family/prediction_results/scores"
# detector2_dir = "result/flux-dda-flux-family/prediction_results/scores"

datasets = "bfree-aigibench"
detector1_dir = "/data1/junwei/DDA/result/dda-bfree-aigibench/prediction_results/scores"
detector2_dir = (
    "/data1/junwei/DDA/result/flux-dda-bfree-aigibench/prediction_results/scores"
)

# thresh_list = [0.5, 0.6, 0.7, 0.8, 0.9]
# sd_thresh_list = [0.5, 0.55, 0.6, 0.65, ..., 0.9]
sd_thresh_list = range(0.5, 0.9, 0.05)
flux_thresh_list = [0.9, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0]

for thresh1 in sd_thresh_list:
    for thresh2 in flux_thresh_list:
        print(f"Evaluating with thresholds: Detector1={thresh1}, Detector2={thresh2}")
        save_csv = f"double_vae_result/{datasets}_thres_sd_{thresh1}_thres_flux_{thresh2}_results.csv"
        save_avg_csv = f"double_vae_result/avg_{datasets}_thres_sd_{thresh1}_thres_flux_{thresh2}_results.csv"
        df = evaluate_detectors(
            detector1_dir, detector2_dir, thresh1, thresh2, save_csv
        )
        get_avg_acc_from_csv(save_csv, save_avg_csv)

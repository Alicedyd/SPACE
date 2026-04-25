import os, json, pandas as pd
from tqdm import tqdm
from itertools import product

# -------------------------------------------------
# 1. 你的文件夹列表 —— 现在是三元组 (Tuple of 3)
#    (detector1_dir, detector2_dir, detector3_dir)
# -------------------------------------------------

FOLDER_TRIPLETS = [
    # (
    #     "/root/autodl-tmp/codes/DDA/result/dda-all-datasets-v2/prediction_results/scores",
    #     "/root/autodl-tmp/codes/DDA/result/flux-dda-all-datasets-v2/prediction_results/scores",
    #     "/root/autodl-tmp/codes/DDA/result/clip_jpeg90_nomixup_all_datasets/prediction_results/scores/",
    # ),
    (
        "/root/autodl-tmp/codes/SPACE/result/SPACE/SD_REM_20260421_081600/prediction_results/scores",
        "/root/autodl-tmp/codes/SPACE/result/SPACE/FLUX_Dinov3_20260423_153847/prediction_results/scores",
        "/root/autodl-tmp/codes/SPACE/result/SPACE/CLIP_REM_20260421_162034/prediction_results/scores",
    ),
]

# -------------------------------------------------
# 2. 阈值组合 (增加第三个模型的阈值列表)
# -------------------------------------------------
SD_THRESH_LIST = [0.5]
FLUX_THRESH_LIST = [0.98]
# FLUX_THRESH_LIST = [i / 100 for i in range(0, 100)]
MODEL3_THRESH_LIST = [0.5]
# FLUX_THRESH_LIST = [
#     0.1,
#     0.2,
#     0.3,
#     0.4,
#     0.5,
#     0.6,
#     0.7,
#     0.8,
#     0.9,
# ]  # <--- 请根据需要修改第三个模型的阈值
# MODEL3_THRESH_LIST = [
#     0.1,
#     0.2,
#     0.3,
#     0.4,
#     0.5,
#     0.6,
#     0.7,
#     0.8,
#     0.9,
# ]  # <--- 请根据需要修改第三个模型的阈值


# -------------------------------------------------
# 3. 辅助函数
# -------------------------------------------------
def load_json_safe(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def evaluate_detectors(
    detector1_dir, detector2_dir, detector3_dir, thresh1, thresh2, thresh3
):
    """
    修改为支持三个模型：
    逻辑：Model1 > T1 OR Model2 > T2 OR Model3 > T3
    """
    results = []

    # 获取子文件夹名称（假设三个文件夹下的 subset 结构一致，以 detector1 为准）
    if not os.path.exists(detector1_dir):
        print(f"Warning: {detector1_dir} does not exist.")
        return pd.DataFrame()

    for subset_name in tqdm(
        sorted(os.listdir(detector1_dir)), desc="Processing subsets"
    ):
        subset_path_1 = os.path.join(detector1_dir, subset_name)
        subset_path_2 = os.path.join(detector2_dir, subset_name)
        subset_path_3 = os.path.join(detector3_dir, subset_name)

        # 确保三个路径都是文件夹
        if not (
            os.path.isdir(subset_path_1)
            and os.path.isdir(subset_path_2)
            and os.path.isdir(subset_path_3)
        ):
            continue

        fake1 = load_json_safe(os.path.join(subset_path_1, "fake.json"))
        real1 = load_json_safe(os.path.join(subset_path_1, "real.json"))

        fake2 = load_json_safe(os.path.join(subset_path_2, "fake.json"))
        real2 = load_json_safe(os.path.join(subset_path_2, "real.json"))

        fake3 = load_json_safe(os.path.join(subset_path_3, "fake.json"))
        real3 = load_json_safe(os.path.join(subset_path_3, "real.json"))

        correct_fake = total_fake = 0
        correct_real = total_real = 0

        # ----------- fake GT -----------
        # 取三个字典 key 的并集
        all_fake_imgs = set(fake1.keys()) | set(fake2.keys()) | set(fake3.keys())
        for img in all_fake_imgs:
            logit1 = fake1.get(img)
            logit2 = fake2.get(img)
            logit3 = fake3.get(img)

            # 只有当三个模型都有结果时才统计（保证数据对齐）
            if logit1 is None or logit2 is None or logit3 is None:
                continue

            # 三个模型取“或”逻辑
            final_fake = (logit1 > thresh1) or (logit2 > thresh2) or (logit3 > thresh3)
            # final_fake = (min(logit1, logit2) > thresh1) or (logit3 > thresh3)

            total_fake += 1
            if final_fake:
                correct_fake += 1

        # ----------- real GT -----------
        all_real_imgs = set(real1.keys()) | set(real2.keys()) | set(real3.keys())
        for img in all_real_imgs:
            logit1 = real1.get(img)
            logit2 = real2.get(img)
            logit3 = real3.get(img)

            if logit1 is None or logit2 is None or logit3 is None:
                continue

            # 三个模型取“或”逻辑
            final_fake = (logit1 > thresh1) or (logit2 > thresh2) or (logit3 > thresh3)
            # final_fake = (min(logit1, logit2) > thresh1) or (logit3 > thresh3)

            total_real += 1
            # Real 图片，判定为 False (Not Fake) 才是正确
            if not final_fake:
                correct_real += 1

        fake_acc = correct_fake / total_fake if total_fake else None
        real_acc = correct_real / total_real if total_real else None

        if real_acc is None and fake_acc is None:
            avg_acc = None
        elif real_acc is None:
            avg_acc = fake_acc
        elif fake_acc is None:
            avg_acc = real_acc
        else:
            avg_acc = (real_acc + fake_acc) / 2

        dataset_name, subset_short = (
            subset_name.split("_", 1)
            if "_" in subset_name
            else (subset_name, subset_name)
        )
        results.append(
            {
                "dataset": dataset_name,
                "subset": subset_short,
                "real_acc": real_acc,
                "fake_acc": fake_acc,
                "avg_acc": avg_acc,
            }
        )

    return pd.DataFrame(results)


def get_avg_acc_from_csv(df):
    """保持原逻辑不变，负责聚合 Subset 到 Dataset"""
    MERGE_GROUPS = {
        "cyclegan": [
            "cyclegan-apple",
            "cyclegan-horse",
            "cyclegan-orange",
            "cyclegan-summer",
            "cyclegan-winter",
            "cyclegan-zebra",
        ],
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
        "stylegan": ["stylegan-bedroom", "stylegan-car", "stylegan-cat"],
        "stylegan2": [
            "stylegan2-car",
            "stylegan2-cat",
            "stylegan2-church",
            "stylegan2-horse",
        ],
    }
    subset_to_group = {v: k for k, vv in MERGE_GROUPS.items() for v in vv}

    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    df = df.dropna(subset=["dataset", "subset", "avg_acc"]).copy()
    df["subset_merged"] = df["subset"].map(subset_to_group).fillna(df["subset"])

    # subset 层平均
    subset_avg = df.groupby(["dataset", "subset_merged"], as_index=False).agg(
        {"real_acc": "mean", "fake_acc": "mean", "avg_acc": "mean"}
    )

    # dataset 层平均
    dataset_avg = subset_avg.groupby("dataset", as_index=False).agg(
        {"real_acc": "mean", "fake_acc": "mean", "avg_acc": "mean"}
    )
    return dataset_avg, subset_avg


# -------------------------------------------------
# 4. 主循环：收集所有结果
# -------------------------------------------------
records = []  # 最后一次性构造 DataFrame

# 使用 product 遍历三个阈值列表
for thresh1, thresh2, thresh3 in product(
    SD_THRESH_LIST, FLUX_THRESH_LIST, MODEL3_THRESH_LIST
):
    print(f"Processing thresholds: SD={thresh1}, Flux={thresh2}, Model3={thresh3}")

    dataset_row = {"real_acc": {}, "fake_acc": {}, "avg_acc": {}}  # 存 dataset 层
    subset_row = {"real_acc": {}, "fake_acc": {}, "avg_acc": {}}  # 存 subset  层

    # 遍历三个文件夹路径的组合
    for det1_dir, det2_dir, det3_dir in FOLDER_TRIPLETS:
        df_detail = evaluate_detectors(
            det1_dir, det2_dir, det3_dir, thresh1, thresh2, thresh3
        )

        if df_detail.empty:
            continue

        df_dataset, df_subset = get_avg_acc_from_csv(df_detail)

        # 把 dataset 层 avg_acc 记录下来
        for _, r in df_dataset.iterrows():
            ds = r["dataset"]
            dataset_row["avg_acc"][ds] = r["avg_acc"]
            if not pd.isna(r["real_acc"]):
                dataset_row["real_acc"][ds] = r["real_acc"]
            if not pd.isna(r["fake_acc"]):
                dataset_row["fake_acc"][ds] = r["fake_acc"]

        # 把 subset 层 avg_acc 记录下来
        for _, r in df_subset.iterrows():
            key = f"{r['dataset']}_{r['subset_merged']}"
            subset_row["avg_acc"][key] = r["avg_acc"]
            if not pd.isna(r["real_acc"]):
                subset_row["real_acc"][key] = r["real_acc"]
            if not pd.isna(r["fake_acc"]):
                subset_row["fake_acc"][key] = r["fake_acc"]

    # 拼成三行
    for metric in ["real_acc", "fake_acc", "avg_acc"]:
        # 记录三个阈值
        base = {
            "thresh1": thresh1,
            "thresh2": thresh2,
            "thresh3": thresh3,
            "metric": metric,
        }
        base.update(dataset_row[metric])
        base.update(subset_row[metric])
        records.append(base)

# -------------------------------------------------
# 5. 生成最终 csv
# -------------------------------------------------
if records:
    final_df = pd.DataFrame(records)
    # 把 metric 列展开成三行，且顺序固定
    final_df = final_df.sort_values(["thresh1", "thresh2", "thresh3", "metric"])
    final_csv = "SPACE_merged.csv"
    final_df.to_csv(final_csv, index=False)
    print(f"✅ All done! 三个模型融合结果已保存至 {final_csv}")
else:
    print("❌ No records to save. Please check your folder paths.")

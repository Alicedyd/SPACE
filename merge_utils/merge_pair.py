import os, json, pandas as pd
import argparse
from tqdm import tqdm
from itertools import product


# -------------------------------------------------
# 0. 参数解析 (实现通过输入自定义输出文件名)
#    使用方法: python script.py --name custom_output.csv
# -------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Merge Logits for 2 Models")
    parser.add_argument("--all", action="store_true")
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        default="default",
    )
    return parser.parse_args()


args = parse_args()

# -------------------------------------------------
# 1. 你的文件夹列表 —— 现在是二元组 (Tuple of 2)
#    (detector1_dir, detector2_dir)
# -------------------------------------------------

# FOLDER_PAIRS = [
#     # (
#     #     "/root/autodl-tmp/codes/DDA/result/dda-all-datasets-v2/prediction_results/scores",
#     #     "/root/autodl-tmp/codes/comparison_code/CO-SPY/results/all_datasets/prediction_results/scores/",
#     # ),
#     (
#         "/root/autodl-tmp/codes/DDA/result/dda-all-datasets-v2/prediction_results/scores",
#         "/root/autodl-tmp/codes/comparison_code/UniversalFakeDetect/results/all_datasets/prediction_results/scores/",
#     ),
#     # 你可以在这里添加更多对组合
# ]
FOLDER_PAIRS_DICT = {
    "dda_cospy": (
        "/root/autodl-tmp/codes/DDA/result/dda-all-datasets-v2/prediction_results/scores",
        "/root/autodl-tmp/codes/comparison_code/CO-SPY/results/all_datasets/prediction_results/scores/",
    ),
    "dda_univfd": (
        "/root/autodl-tmp/codes/DDA/result/dda-all-datasets-v2/prediction_results/scores",
        "/root/autodl-tmp/codes/comparison_code/UniversalFakeDetect/results/all_datasets/prediction_results/scores/",
    ),
    "dda_drct": (
        "/root/autodl-tmp/codes/DDA/result/dda-all-datasets-v2/prediction_results/scores",
        "/root/autodl-tmp/codes/comparison_code/DRCT/results/all_datasets/prediction_results/scores/",
    ),
    "dda_bfree": (
        "/root/autodl-tmp/codes/DDA/result/dda-all-datasets-v2/prediction_results/scores",
        "/root/autodl-tmp/codes/comparison_code/B-Free/code/results/all_datasets/prediction_results/scores/",
    ),
}

if args.all:
    selected_keys = FOLDER_PAIRS_DICT.keys()
elif args.name != "default":
    selected_keys = args.name.split(",")
else:
    print("Error: Please select all or provide precise seletion")
    exit()

DATASETS = [
    "Chameleon",
    "WildRF",
    "cospy-inthewild",
    "BFree-Online",
    "AIGCDetectionBenchmark",
    "GenImage",
    "DRCT",
    "AIGIBench",
]

# -------------------------------------------------
# 2. 阈值组合 (保留两个模型的阈值列表)
# -------------------------------------------------
THRESH_LIST_1 = [0.5]
# THRESH_LIST_1 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

THRESH_LIST_2 = [0.5]
# THRESH_LIST_2 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


# -------------------------------------------------
# 3. 辅助函数
# -------------------------------------------------
def load_json_safe(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def evaluate_detectors(detector1_dir, detector2_dir, thresh1, thresh2, DATASETS):
    """
    支持两个模型：
    逻辑：Model1 > T1 OR Model2 > T2
    """
    results = []

    # 获取子文件夹名称（假设两个文件夹下的 subset 结构一致，以 detector1 为准）
    if not os.path.exists(detector1_dir):
        print(f"Warning: {detector1_dir} does not exist.")
        return pd.DataFrame()

    for subset_name in tqdm(
        sorted(os.listdir(detector1_dir)), desc="Processing subsets"
    ):
        subset_path_1 = os.path.join(detector1_dir, subset_name)
        subset_path_2 = os.path.join(detector2_dir, subset_name)

        dataset, subset = subset_name.split("_", 1)

        if dataset not in DATASETS:
            continue

        # 确保两个路径都是文件夹
        if not (os.path.isdir(subset_path_1) and os.path.isdir(subset_path_2)):
            continue

        fake1 = load_json_safe(os.path.join(subset_path_1, "fake.json"))
        real1 = load_json_safe(os.path.join(subset_path_1, "real.json"))

        fake2 = load_json_safe(os.path.join(subset_path_2, "fake.json"))
        real2 = load_json_safe(os.path.join(subset_path_2, "real.json"))

        correct_fake = total_fake = 0
        correct_real = total_real = 0

        # ----------- fake GT -----------
        # 取两个字典 key 的并集
        all_fake_imgs = set(fake1.keys()) | set(fake2.keys())

        for img in all_fake_imgs:
            logit1 = fake1.get(img)
            logit2 = fake2.get(img)

            # 只有当两个模型都有结果时才统计（保证数据对齐，求交集）
            if logit1 is None or logit2 is None:
                continue

            # 两个模型取“或”逻辑
            final_fake = (logit1 > thresh1) or (logit2 > thresh2)

            total_fake += 1
            if final_fake:
                correct_fake += 1

        # ----------- real GT -----------
        all_real_imgs = set(real1.keys()) | set(real2.keys())

        for img in all_real_imgs:
            logit1 = real1.get(img)
            logit2 = real2.get(img)

            if logit1 is None or logit2 is None:
                continue

            # 两个模型取“或”逻辑
            final_fake = (logit1 > thresh1) or (logit2 > thresh2)

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
for method_key, (dect1_dir, dect2_dir) in FOLDER_PAIRS_DICT.items():
    if method_key not in selected_keys:
        continue
    else:
        print(f"Processing {method_key}")

    FOLDER_PAIRS = [(dect1_dir, dect2_dir)]
    records = []  # 最后一次性构造 DataFrame

    # 使用 product 遍历两个阈值列表
    for thresh1, thresh2 in product(THRESH_LIST_1, THRESH_LIST_2):
        print(f"Processing thresholds: Model1={thresh1}, Model2={thresh2}")

        dataset_row = {"real_acc": {}, "fake_acc": {}, "avg_acc": {}}  # 存 dataset 层
        subset_row = {"real_acc": {}, "fake_acc": {}, "avg_acc": {}}  # 存 subset  层

        # 遍历两个文件夹路径的组合
        for det1_dir, det2_dir in FOLDER_PAIRS:
            df_detail = evaluate_detectors(
                det1_dir, det2_dir, thresh1, thresh2, DATASETS
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

        # 拼成三行 (Real/Fake/Avg)
        for metric in ["real_acc", "fake_acc", "avg_acc"]:
            base = {
                "thresh1": thresh1,
                "thresh2": thresh2,
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
        # 按照阈值和 metric 排序
        final_df = final_df.sort_values(["thresh1", "thresh2", "metric"])

        # 使用命令行参数指定的文件名
        final_csv = method_key + ".csv"
        final_df.to_csv(final_csv, index=False)
        print(f"✅ All done! 两个模型融合结果已保存至: {final_csv}")
    else:
        print("❌ No records to save. Please check your folder paths.")

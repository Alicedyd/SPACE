import os, json, pandas as pd
from tqdm import tqdm
from itertools import product

# -------------------------------------------------
# 1. 你的文件夹对列表 —— 按顺序写即可，不会出现在最终 csv
#    (detector1_dir, detector2_dir)
# -------------------------------------------------
# FOLDER_PAIRS = [
#     ("/data1/junwei/DDA/result/dda-all-datasets/prediction_results/scores",
#         "/data1/junwei/DDA/result/flux-dda-all-datasets/prediction_results/scores",),

#     ("/data1/junwei/DDA/result/dda-bfree-aigibench/prediction_results/scores",
#         "/data1/junwei/DDA/result/flux-dda-bfree-aigibench/prediction_results/scores",),

#     ("/data1/junwei/DDA/result/dda-flux-family/prediction_results/scores",
#         "/data1/junwei/DDA/result/flux-dda-flux-family/prediction_results/scores",),
# ]
FOLDER_PAIRS = [
    (
        "/root/autodl-tmp/codes/DDA/result/dda-cospy-inthewild/prediction_results/scores/",
        "/root/autodl-tmp/codes/DDA/result/flux-cospy-inthewild/prediction_results/scores/",
    ),
]

# -------------------------------------------------
# 2. 阈值组合
# -------------------------------------------------
SD_THRESH_LIST = [0.5]
FLUX_THRESH_LIST = [1.0]


# -------------------------------------------------
# 3. 原脚本函数（完全不变，仅去掉中间保存）
# -------------------------------------------------
def load_json_safe(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def evaluate_detectors(detector1_dir, detector2_dir, thresh1, thresh2):
    """与原函数逻辑完全一致，仅不再保存 csv，而是返回 DataFrame"""
    results = []
    for subset_name in tqdm(
        sorted(os.listdir(detector1_dir)), desc="Processing subsets"
    ):
        subset_path_1 = os.path.join(detector1_dir, subset_name)
        subset_path_2 = os.path.join(detector2_dir, subset_name)
        if not (os.path.isdir(subset_path_1) and os.path.isdir(subset_path_2)):
            continue

        fake1 = load_json_safe(os.path.join(subset_path_1, "fake.json"))
        real1 = load_json_safe(os.path.join(subset_path_1, "real.json"))
        fake2 = load_json_safe(os.path.join(subset_path_2, "fake.json"))
        real2 = load_json_safe(os.path.join(subset_path_2, "real.json"))

        correct_fake = total_fake = 0
        correct_real = total_real = 0

        # ----------- fake GT -----------
        for img in set(fake1.keys()) | set(fake2.keys()):
            logit1, logit2 = fake1.get(img), fake2.get(img)
            if logit1 is None or logit2 is None:
                continue
            final_fake = (logit1 > thresh1) or (logit2 > thresh2)
            total_fake += 1
            if final_fake:
                correct_fake += 1

        # ----------- real GT -----------
        for img in set(real1.keys()) | set(real2.keys()):
            logit1, logit2 = real1.get(img), real2.get(img)
            if logit1 is None or logit2 is None:
                continue
            final_fake = (logit1 > thresh1) or (logit2 > thresh2)
            total_real += 1
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
    """原函数逻辑，只不过输入变成 DataFrame，不再读写文件"""
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

for thresh1, thresh2 in product(SD_THRESH_LIST, FLUX_THRESH_LIST):
    print(f"Processing thresholds: SD={thresh1}, Flux={thresh2}")
    dataset_row = {"real_acc": {}, "fake_acc": {}, "avg_acc": {}}  # 存 dataset 层
    subset_row = {"real_acc": {}, "fake_acc": {}, "avg_acc": {}}  # 存 subset  层

    for det1_dir, det2_dir in FOLDER_PAIRS:
        df_detail = evaluate_detectors(det1_dir, det2_dir, thresh1, thresh2)
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
        base = {"thresh1": thresh1, "thresh2": thresh2, "metric": metric}
        base.update(dataset_row[metric])
        base.update(subset_row[metric])
        records.append(base)

# -------------------------------------------------
# 5. 生成最终 csv
# -------------------------------------------------
final_df = pd.DataFrame(records)
# 把 metric 列展开成三行，且顺序固定
final_df = final_df.sort_values(["thresh1", "thresh2", "metric"])
final_csv = "all_results_in_one.csv"
final_df.to_csv(final_csv, index=False)
print(f"✅ All done! 唯一总结果已保存至 {final_csv}")

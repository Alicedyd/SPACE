import os, json
import re
from tqdm import tqdm
import pandas as pd
import yaml

dir_sd = "/data1/junwei/DDA/result/align-EvalFlaw/prediction_results/scores"
dir_flux = "/data1/junwei/DDA/result/flux-EvalFlaw/prediction_results/scores"

fake_txt = "/data1/junwei/DDA/MLLM/EvalFlaw/fake_predictions.txt"
real_txt = "/data1/junwei/DDA/MLLM/EvalFlaw/real_predictions.txt"

with open("./MLLM/dataset.yaml", "r") as f:
    small_path = yaml.safe_load(f)

SD_THRES = 0.51
FLUX_THRES = 0.98

DATASETS = ["Chameleon", "WildRF", "BFree-Online", "AIGIBench", "synthwildx"]

def load_json_safe(path):
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return {}

def load_prediction_txt(fake_txt, real_txt):
    fake_set, real_set = set(), set()
    if os.path.exists(fake_txt):
        with open(fake_txt, "r") as f:
            fake_set = set(x.strip() for x in f if x.strip())
    if os.path.exists(real_txt):
        with open(real_txt, "r") as f:
            real_set = set(x.strip() for x in f if x.strip())

    big_result = {}
    for img in fake_set:
        if "Bfree_viral" in img:
            img = re.sub(r'(REAL|FAKE)/[^/]+/(img)', r'\1/\2', img)
        big_result[img] = 1
    for img in real_set:
        if "Bfree_viral" in img:
            img = re.sub(r'(REAL|FAKE)/[^/]+/(img)', r'\1/\2', img)
        big_result[img] = 0

    return big_result


def evaluate_detectors(detector1_dir, detector2_dir, thresh1, thresh2, real_txt, fake_txt):
    """与原函数逻辑完全一致，仅不再保存 csv，而是返回 DataFrame"""
    big_result = load_prediction_txt(fake_txt, real_txt)

    results = []
    for subset_name in tqdm(sorted(os.listdir(detector1_dir)), desc="Processing subsets"):
        dataset, subset = subset_name.split("_", 1)
        if dataset not in DATASETS:
            continue

        if "real" in small_path[dataset]:
            real_prefix = small_path[dataset].get("real")
        else:
            real_prefix = small_path[dataset][subset].get("real", None)
        fake_prefix = small_path[dataset][subset].get("fake")

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
            full_img = os.path.join(fake_prefix, img)

            big_pre = big_result[full_img]

            if logit1 is None or logit2 is None:
                continue
            final_fake = ( (logit1 > thresh1) or (logit2 > thresh2) ) or (big_pre == 1)
            total_fake += 1
            if final_fake:
                correct_fake += 1

        # ----------- real GT -----------
        for img in set(real1.keys()) | set(real2.keys()):
            logit1, logit2 = real1.get(img), real2.get(img)
            if real_prefix:
                full_img = os.path.join(real_prefix, img)
                big_pre = big_result[full_img]
            else:
                big_pre is None

            if logit1 is None or logit2 is None or big_pre is None:
                continue
            final_fake = ((logit1 > thresh1) or (logit2 > thresh2)) or (big_pre == 1)
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

        dataset_name, subset_short = subset_name.split("_", 1) if "_" in subset_name else (subset_name, subset_name)
        results.append({
            "dataset": dataset_name,
            "subset": subset_short,
            "real_acc": real_acc,
            "fake_acc": fake_acc,
            "avg_acc": avg_acc
        })

    return pd.DataFrame(results)

def get_avg_acc_from_csv(df):
    """原函数逻辑，只不过输入变成 DataFrame，不再读写文件"""
    MERGE_GROUPS = {
        "cyclegan": ["cyclegan-apple", "cyclegan-horse", "cyclegan-orange", "cyclegan-summer", "cyclegan-winter", "cyclegan-zebra"],
        "progan": ["progan-airplane", "progan-bicycle", "progan-bird", "progan-boat", "progan-bottle", "progan-bus", "progan-car", "progan-cat", "progan-chair", "progan-cow", "progan-diningtable", "progan-dog", "progan-horse", "progan-motorbike", "progan-person", "progan-pottedplant", "progan-sheep", "progan-sofa", "progan-train", "progan-tvmonitor"],
        "stylegan": ["stylegan-bedroom", "stylegan-car", "stylegan-cat"],
        "stylegan2": ["stylegan2-car", "stylegan2-cat", "stylegan2-church", "stylegan2-horse"],
    }
    subset_to_group = {v: k for k, vv in MERGE_GROUPS.items() for v in vv}
    df = df.dropna(subset=["dataset", "subset", "avg_acc"]).copy()
    df["subset_merged"] = df["subset"].map(subset_to_group).fillna(df["subset"])

    # subset 层平均
    subset_avg = (df.groupby(["dataset", "subset_merged"], as_index=False)
                    .agg({"real_acc": "mean", "fake_acc": "mean", "avg_acc": "mean"}))

    # dataset 层平均
    dataset_avg = (subset_avg.groupby("dataset", as_index=False)
                     .agg({"real_acc": "mean", "fake_acc": "mean", "avg_acc": "mean"}))
    return dataset_avg, subset_avg

small_result = evaluate_detectors(dir_sd, dir_flux, SD_THRES, FLUX_THRES, real_txt, fake_txt)
dataset_avg, subset_avg = get_avg_acc_from_csv(small_result)

dataset_row = {"real_acc": {}, "fake_acc": {}, "avg_acc": {}}   # 存 dataset 层
subset_row  = {"real_acc": {}, "fake_acc": {}, "avg_acc": {}}   # 存 subset  层

records = []

# 把 dataset 层 avg_acc 记录下来
for _, r in dataset_avg.iterrows():
    ds = r["dataset"]
    dataset_row["avg_acc"][ds] = r["avg_acc"]
    if not pd.isna(r["real_acc"]):
        dataset_row["real_acc"][ds] = r["real_acc"]
    if not pd.isna(r["fake_acc"]):
        dataset_row["fake_acc"][ds] = r["fake_acc"]

# 把 subset 层 avg_acc 记录下来
for _, r in subset_avg.iterrows():
    key = f"{r['dataset']}_{r['subset_merged']}"
    subset_row["avg_acc"][key] = r["avg_acc"]
    if not pd.isna(r["real_acc"]):
        subset_row["real_acc"][key] = r["real_acc"]
    if not pd.isna(r["fake_acc"]):
        subset_row["fake_acc"][key] = r["fake_acc"]

# 拼成三行
for metric in ["real_acc", "fake_acc", "avg_acc"]:
    base = {"thresh1": SD_THRES, "thresh2": FLUX_THRES, "metric": metric}
    base.update(dataset_row[metric])
    base.update(subset_row[metric])
    records.append(base)

final_df = pd.DataFrame(records)
# 把 metric 列展开成三行，且顺序固定
final_df = final_df.sort_values(["thresh1", "thresh2", "metric"])
final_csv = "all_results_in_one_mllm.csv"
final_df.to_csv(final_csv, index=False)
print(f"✅ All done! 唯一总结果已保存至 {final_csv}")
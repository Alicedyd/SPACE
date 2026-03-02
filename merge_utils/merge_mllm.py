import os, json, re, yaml
import pandas as pd
from tqdm import tqdm


# ========== 配置函数 ==========
def _norm(s: str) -> str:
    # 统一 -/_ 差异，避免 "cospy-inthewild" vs "cospy_inthewild" 这种匹配不上
    return s.lower().replace("-", "_")


def auto_build_tasks(MLLM_DICT, DATASETS_DICT, run_keys=None, strict=False):
    """
    run_keys: 只生成这些 dataset_key 的 task，例如 ["manual", "aiginow"]
    strict:  True 时，若某个 dataset_key 匹配不到任何 mllm_key 就报错；否则跳过
    """
    tasks = []
    ds_keys = list(DATASETS_DICT.keys())
    if run_keys is not None:
        ds_keys = [k for k in ds_keys if k in set(run_keys)]

    for dataset_key in ds_keys:
        dsn = _norm(dataset_key)

        matched_mllm_keys = []
        for mllm_key in MLLM_DICT.keys():
            if dsn in _norm(mllm_key):
                matched_mllm_keys.append(mllm_key)

        if not matched_mllm_keys:
            if strict:
                raise ValueError(f"No MLLM matched for dataset_key='{dataset_key}'")
            continue

        tasks.append(
            {
                "name": dataset_key,
                "mllm": matched_mllm_keys,  # 这里 aiginow 会自动得到两个
                "datasets": [
                    dataset_key
                ],  # dataset group 仍然按 key 取 DATASETS_DICT[dataset_key]
            }
        )

    # 可选：排序让输出稳定
    tasks.sort(key=lambda x: x["name"])
    # 也可以把每个 task 内的 mllm 排序稳定
    for t in tasks:
        t["mllm"].sort()
    return tasks


# ========== 配置部分 ==========
EXPERT_DICT = {
    "dda": "/root/autodl-tmp/codes/DDA/result/dda-AIGI_Now/prediction_results/scores/",
    "dda_mixed": "/root/autodl-tmp/codes/DDA/result/align-AIGI-Now/prediction_results/scores/",
    "DRCT": "/root/autodl-tmp/codes/comparison_code/DRCT/results/all_datasets/prediction_results/scores/",
    "BFree": "/root/autodl-tmp/codes/comparison_code/B-Free/code/results/all_datasets/prediction_results/scores/",
    "cospy": "/root/autodl-tmp/codes/comparison_code/CO-SPY/results/all_datasets/prediction_results/scores/",
    "univfd": "/root/autodl-tmp/codes/comparison_code/UniversalFakeDetect/results/all_datasets/prediction_results/scores/",
}

MLLM_DICT = {
    "mixed_aiginow": "../MLLM/VLM-mixed/",
    "ori_aiginow": "../MLLM/AIGI-Now/qwen25-7B/",
    "manual": "../MLLM/qwen2.5-7B-AllBenchmarks/",
, "univfd"    "inthewild": "../MLLM/inthewild/qwen25-7B/",
    "cospywild": "../MLLM/cospy-wild/qwen25-7B/",
}

DATASETS_DICT = {
    "manual": [
        "DRCT",
        "GenImage",
        "AIGCDetectionBenchmark",
    ],
    "inthewild": ["BFree-Online", "WildRF", "Chameleon", "AIGIBench"],
    "aiginow": ["AIGI-Now"],
    "cospywild": ["cospy-inthewild"],
}

TASKS = auto_build_tasks(MLLM_DICT, DATASETS_DICT)

EXPERT_USED = ["BFree", "DRCT", "cospy", "univfd"]
TASKS_USED = ["inthewild", "manual", "cospywild"]

# EXPERT_USED = ["dda", "dda_mixed"]
# TASKS_USED = ["aiginow"]

for task in TASKS:
    if task["name"] not in TASKS_USED:
        continue

    MLLM_USED = task["mllm"]
    DATASETS_USED = task["datasets"]
    for mllm_key, path_mllm in MLLM_DICT.items():
        if mllm_key not in MLLM_USED:
            continue

        for datasets_key, DATASETS in DATASETS_DICT.items():
            if datasets_key not in DATASETS_USED:
                continue

            for expert_key, dir_sd in EXPERT_DICT.items():
                if expert_key not in EXPERT_USED:
                    continue
                print(
                    f"================ 处理单模型 Expert {expert_key} + MLLM {mllm_key} 数据集组合 {datasets_key} =================="
                )

                fake_txt = f"{path_mllm}/fake_predictions.txt"
                real_txt = f"{path_mllm}/real_predictions.txt"

                # 加载 dataset path 配置
                # 注意：请确保此路径存在，或者根据你的环境修改
                if os.path.exists("../MLLM/dataset.yaml"):
                    with open("../MLLM/dataset.yaml", "r") as f:
                        small_path = yaml.safe_load(f)
                else:
                    # Fallback 或者报错，这里给个空字典防止直接崩，但实际运行需要该文件
                    print("Warning: ../MLLM/dataset.yaml not found.")
                    small_path = {}

                # 阈值配置 (仅保留第一个模型的阈值列表)
                SD_THRES_LIST = [0.5]
                # ============================

                def load_json_safe(path):
                    if os.path.exists(path):
                        with open(path, "r") as f:
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
                            img = re.sub(r"(REAL|FAKE)/[^/]+/(img)", r"\1/\2", img)
                        big_result[img] = 1
                    for img in real_set:
                        if "Bfree_viral" in img:
                            img = re.sub(r"(REAL|FAKE)/[^/]+/(img)", r"\1/\2", img)
                        big_result[img] = 0
                    return big_result

                def evaluate_detectors(detector1_dir, thresh1, real_txt, fake_txt):
                    """
                    修改为：单模型 (Model1) 与 大模型 (MLLM) 结合
                    逻辑：Model1 > T1 OR MLLM == Fake
                    """
                    big_result = load_prediction_txt(fake_txt, real_txt)
                    results = []

                    if not os.path.exists(detector1_dir):
                        print(f"Warning: {detector1_dir} does not exist.")
                        return pd.DataFrame()

                    for subset_name in tqdm(
                        sorted(os.listdir(detector1_dir)),
                        desc=f"th1={thresh1:.2f}",
                    ):
                        # 防止非文件夹文件报错
                        if "_" not in subset_name:
                            continue

                        dataset, subset = subset_name.split("_", 1)
                        if dataset not in DATASETS:
                            continue

                        if dataset == "AIGIBench" and subset not in [
                            "CommunityAI",
                            "SocialRF",
                        ]:
                            continue

                        if dataset not in small_path:
                            continue

                        if "real" in small_path[dataset]:
                            real_prefix = small_path[dataset].get("real")
                        else:
                            real_prefix = (
                                small_path[dataset].get(subset, {}).get("real", None)
                            )
                        fake_prefix = small_path[dataset].get(subset, {}).get("fake")

                        subset_path_1 = os.path.join(detector1_dir, subset_name)

                        # 移除了 subset_path_2 的检查
                        if not os.path.isdir(subset_path_1):
                            continue

                        fake1 = load_json_safe(os.path.join(subset_path_1, "fake.json"))
                        real1 = load_json_safe(os.path.join(subset_path_1, "real.json"))
                        # 移除了 fake2, real2

                        correct_fake = total_fake = correct_real = total_real = 0

                        # ----------- fake GT -----------
                        # 仅遍历 fake1 的 keys
                        for img in fake1.keys():
                            logit1 = fake1.get(img)

                            root, ext = os.path.splitext(img)
                            if fake_prefix:
                                full_img = os.path.join(fake_prefix, root + ext.lower())
                                big_pre = big_result.get(full_img)
                            else:
                                big_pre = None

                            # 只要 logit1 或 big_pre 缺失，根据实际情况决定是否跳过
                            # 原逻辑是：只要有一个缺失就 continue。这里保持一致。
                            # 注意：如果 big_pre 没找到(None)，也会跳过。
                            if logit1 is None or big_pre is None:
                                continue

                            # 融合逻辑：模型1 > 阈值  OR  大模型认为是Fake
                            final_fake = (logit1 > thresh1) or (big_pre == 1)

                            total_fake += 1
                            if final_fake:
                                correct_fake += 1

                        # ----------- real GT -----------
                        for img in real1.keys():
                            logit1 = real1.get(img)

                            if real_prefix:
                                full_img = os.path.join(real_prefix, img)
                                big_pre = big_result.get(full_img)
                            else:
                                big_pre = None

                            if logit1 is None or big_pre is None:
                                continue

                            # 融合逻辑
                            final_fake = (logit1 > thresh1) or (big_pre == 1)

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

                        results.append(
                            {
                                "dataset": dataset,
                                "subset": subset,
                                "real_acc": real_acc,
                                "fake_acc": fake_acc,
                                "avg_acc": avg_acc,
                            }
                        )

                    return pd.DataFrame(results)

                def get_avg_acc_from_csv(df):
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
                        "stylegan": [
                            "stylegan-bedroom",
                            "stylegan-car",
                            "stylegan-cat",
                        ],
                        "stylegan2": [
                            "stylegan2-car",
                            "stylegan2-cat",
                            "stylegan2-church",
                            "stylegan2-horse",
                        ],
                    }
                    subset_to_group = {
                        v: k for k, vv in MERGE_GROUPS.items() for v in vv
                    }

                    if df.empty:
                        return pd.DataFrame(), pd.DataFrame()

                    df = df.dropna(subset=["dataset", "subset", "avg_acc"]).copy()
                    df["subset_merged"] = (
                        df["subset"].map(subset_to_group).fillna(df["subset"])
                    )

                    subset_avg = df.groupby(
                        ["dataset", "subset_merged"], as_index=False
                    ).agg({"real_acc": "mean", "fake_acc": "mean", "avg_acc": "mean"})
                    dataset_avg = subset_avg.groupby("dataset", as_index=False).agg(
                        {"real_acc": "mean", "fake_acc": "mean", "avg_acc": "mean"}
                    )
                    return dataset_avg, subset_avg

                # ========== 主循环 ==========
                all_records = []

                # 仅遍历 SD_THRES_LIST
                for SD_THRES in SD_THRES_LIST:
                    small_result = evaluate_detectors(
                        dir_sd, SD_THRES, real_txt, fake_txt
                    )

                    if small_result.empty:
                        continue

                    dataset_avg, subset_avg = get_avg_acc_from_csv(small_result)

                    dataset_row = {"real_acc": {}, "fake_acc": {}, "avg_acc": {}}
                    subset_row = {"real_acc": {}, "fake_acc": {}, "avg_acc": {}}

                    # dataset-level
                    for _, r in dataset_avg.iterrows():
                        ds = r["dataset"]
                        dataset_row["avg_acc"][ds] = r["avg_acc"]
                        if not pd.isna(r["real_acc"]):
                            dataset_row["real_acc"][ds] = r["real_acc"]
                        if not pd.isna(r["fake_acc"]):
                            dataset_row["fake_acc"][ds] = r["fake_acc"]

                    # subset-level
                    for _, r in subset_avg.iterrows():
                        key = f"{r['dataset']}_{r['subset_merged']}"
                        subset_row["avg_acc"][key] = r["avg_acc"]
                        if not pd.isna(r["real_acc"]):
                            subset_row["real_acc"][key] = r["real_acc"]
                        if not pd.isna(r["fake_acc"]):
                            subset_row["fake_acc"][key] = r["fake_acc"]

                    # 三行结果：real_acc, fake_acc, avg_acc
                    for metric in ["real_acc", "fake_acc", "avg_acc"]:
                        base = {
                            "thresh1": SD_THRES,
                            # "thresh2": FLUX_THRES, # 已移除
                            "metric": metric,
                        }
                        base.update(dataset_row[metric])
                        base.update(subset_row[metric])
                        all_records.append(base)

                # 汇总输出
                if all_records:
                    final_df = pd.DataFrame(all_records)
                    # 按照 thresh1 和 metric 排序
                    final_df = final_df.sort_values(["thresh1", "metric"])

                    # 输出文件名改为 1model
                    final_csv = f"../MLLM_results/dataset_{datasets_key}_mllm_{mllm_key}_merge_{expert_key}.csv"

                    # 确保输出目录存在
                    os.makedirs(os.path.dirname(final_csv), exist_ok=True)

                    final_df.to_csv(final_csv, index=False)
                    print(
                        f"✅ 已完成单模型+MLLM融合，共 {len(SD_THRES_LIST)} 组，每组三行，结果保存在 {final_csv}"
                    )
                else:
                    print("❌ 未生成任何记录，请检查路径配置。")

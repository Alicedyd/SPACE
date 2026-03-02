import os, json, re, yaml
import pandas as pd
from tqdm import tqdm

# ========== 配置部分 ==========
SD_DICT = {
    "824": "/root/autodl-tmp/codes/DDA/result/dda-all-datasets-v2/prediction_results/scores/",
    # "832": "/data1/junwei/DDA/result/dda-832-all_datasets/prediction_results/scores",
    # "837": "/data1/junwei/DDA/result/dda-837-all_datasets/prediction_results/scores",
    # "840": "/data1/junwei/DDA/result/dda-840-all_datasets/prediction_results/scores",
    # "846": "/data1/junwei/DDA/result/dda-846-all_datasets/prediction_results/scores",
}
dir_flux = "/root/autodl-tmp/codes/DDA/result/flux-dda-all-datasets-v2/prediction_results/scores/"


for sd_key, dir_sd in SD_DICT.items():
    print(f"================ 处理 SD 模型 DDA {sd_key} ==================")
    # fake_txt = "../MLLM/qwen2.5-7B-AllBenchmarks/fake_predictions_converted.txt"
    # real_txt = "../MLLM/qwen2.5-7B-AllBenchmarks/real_predictions_converted.txt"
    # fake_txt = "../MLLM/inthewild/qwen25-7B/fake_predictions.txt"
    # real_txt = "../MLLM/inthewild/qwen25-7B/real_predictions.txt"
    fake_txt = "../MLLM/cospy-wild/qwen25-7B/fake_predictions.txt"
    real_txt = "../MLLM/cospy-wild/qwen25-7B/real_predictions.txt"

    with open("../MLLM/dataset.yaml", "r") as f:
        small_path = yaml.safe_load(f)

    # DATASETS = ["GenImage"]
    # DATASETS = [
    #     "Chameleon",
    #     "WildRF",
    #     "BFree-Online",
    #     "AIGIBench",
    #     "synthwildx",
    #     "cospy-inthewild",
    # ]
    # DATASETS = ["DRCT", "GenImage", "AIGCDetectionBenchmark", "synbuster", "ForenSynth"]
    DATASETS = ["cospy-inthewild"]
    # DATASETS = ["AIGI-Now"]

    # 多组阈值配置
    # SD_THRES_LIST = [0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.7]
    SD_THRES_LIST = [1.0]
    FLUX_THRES_LIST = [1.0]
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

    def evaluate_detectors(
        detector1_dir, detector2_dir, thresh1, thresh2, real_txt, fake_txt
    ):
        big_result = load_prediction_txt(fake_txt, real_txt)
        results = []

        for subset_name in tqdm(
            sorted(os.listdir(detector1_dir)),
            desc=f"th1={thresh1:.2f}, th2={thresh2:.2f}",
        ):
            dataset, subset = subset_name.split("_", 1)
            if dataset not in DATASETS:
                continue

            if dataset == "AIGIBench" and subset not in ["CommunityAI", "SocialRF"]:
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

            # print(
            #     f"Dataset: {dataset}, fake1: {len(fake1)}, real1: {len(real1)}, fake2: {len(fake2)}, real2: {len(real2)}"
            # )

            correct_fake = total_fake = correct_real = total_real = 0

            # fake GT
            # print(set(fake1.keys()) | set(fake2.keys()))
            for img in set(fake1.keys()) | set(fake2.keys()):
                logit1, logit2 = fake1.get(img), fake2.get(img)
                root, ext = os.path.splitext(img)
                full_img = os.path.join(fake_prefix, root + ext.lower())
                big_pre = big_result.get(full_img)

                # print(
                #     f"Evaluating image: {full_img}, logit1: {logit1}, logit2: {logit2}, big_pre: {big_pre}"
                # )

                if logit1 is None or logit2 is None or big_pre is None:
                    continue
                final_fake = ((logit1 > thresh1) or (logit2 > thresh2)) or (
                    big_pre == 1
                )
                total_fake += 1
                if final_fake:
                    correct_fake += 1

            # real GT
            for img in set(real1.keys()) | set(real2.keys()):
                logit1, logit2 = real1.get(img), real2.get(img)
                if real_prefix:
                    full_img = os.path.join(real_prefix, img)
                    big_pre = big_result.get(full_img)
                else:
                    big_pre = None

                # print(f"Evaluating image: {full_img}, logit1: {logit1}, logit2: {logit2}, big_pre: {big_pre}")

                if logit1 is None or logit2 is None or big_pre is None:
                    continue
                final_fake = ((logit1 > thresh1) or (logit2 > thresh2)) or (
                    big_pre == 1
                )
                total_real += 1
                if not final_fake:
                    correct_real += 1

            fake_acc = correct_fake / total_fake if total_fake else None
            real_acc = correct_real / total_real if total_real else None
            avg_acc = (
                (real_acc + fake_acc) / 2
                if (real_acc and fake_acc)
                else real_acc or fake_acc
            )

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

        subset_avg = df.groupby(["dataset", "subset_merged"], as_index=False).agg(
            {"real_acc": "mean", "fake_acc": "mean", "avg_acc": "mean"}
        )
        dataset_avg = subset_avg.groupby("dataset", as_index=False).agg(
            {"real_acc": "mean", "fake_acc": "mean", "avg_acc": "mean"}
        )
        return dataset_avg, subset_avg

    # ========== 主循环 ==========
    all_records = []

    for SD_THRES in SD_THRES_LIST:
        for FLUX_THRES in FLUX_THRES_LIST:
            small_result = evaluate_detectors(
                dir_sd, dir_flux, SD_THRES, FLUX_THRES, real_txt, fake_txt
            )
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
                    "thresh2": FLUX_THRES,
                    "metric": metric,
                }
                base.update(dataset_row[metric])
                base.update(subset_row[metric])
                all_records.append(base)

    # 汇总输出
    final_df = pd.DataFrame(all_records)
    final_df = final_df.sort_values(["thresh1", "thresh2", "metric"])
    final_csv = f"../MLLM_results/inthewild/all_results_in_one_mllm_DDA_{sd_key}.csv"
    final_df.to_csv(final_csv, index=False)
    print(
        f"✅ 已完成所有组合，共 {len(SD_THRES_LIST) * len(FLUX_THRES_LIST)} 组，每组三行，结果保存在 {final_csv}"
    )

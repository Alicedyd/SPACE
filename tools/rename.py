#!/usr/bin/env python3
import os
import sys
import argparse


def find_targets(base_dir: str, recursive: bool, pattern: str):
    targets = []
    base_dir = os.path.abspath(base_dir)

    if not recursive:
        # 仅处理第一层子目录
        try:
            for name in os.listdir(base_dir):
                full = os.path.join(base_dir, name)
                if os.path.isdir(full) and name.startswith(f"{pattern}_"):
                    targets.append(full)
        except FileNotFoundError:
            print(f"❌ 路径不存在：{base_dir}", file=sys.stderr)
            sys.exit(1)
    else:
        # 递归收集，按路径深度从深到浅重命名，避免父目录先改动导致路径失效
        for root, dirs, _ in os.walk(base_dir):
            for d in dirs:
                if d.startswith(f"{pattern}_"):
                    targets.append(os.path.join(root, d))
        # 深度优先（先改最深的）
        targets.sort(key=lambda p: p.count(os.sep), reverse=True)

    return targets


def main():
    ap = argparse.ArgumentParser(description="批量把 Any_* 文件夹重命名为 AIGI-Now_*")
    ap.add_argument("directory", help="要处理的根目录")
    ap.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="递归处理所有子目录（默认仅处理根目录下一层）",
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="只预览不改动（打印将要进行的重命名）"
    )
    ap.add_argument("-s", "--s_pattern", type=str, default="Any")
    ap.add_argument("-t", "--t_pattern", type=str, default="cospy-inthewild")
    args = ap.parse_args()

    base = os.path.abspath(args.directory)
    if not os.path.isdir(base):
        print(f"❌ 不是有效目录：{base}", file=sys.stderr)
        sys.exit(1)

    targets = find_targets(base, args.recursive, args.s_pattern)
    if not targets:
        print(f"ℹ️ 未找到需要重命名的文件夹（以 {args.s_pattern}_ 开头）。")
        return

    changed, skipped = 0, 0
    for src in targets:
        parent = os.path.dirname(src)
        old = os.path.basename(src)
        new = f"{args.t_pattern}_" + old[len(f"{args.s_pattern}_") :]
        dst = os.path.join(parent, new)

        if os.path.exists(dst):
            print(f"⚠️ 跳过（目标已存在）：{src}  ->  {dst}")
            skipped += 1
            continue

        print(f"{'DRY-RUN ' if args.dry_run else ''}重命名：{src}  ->  {dst}")
        if not args.dry_run:
            os.rename(src, dst)
        changed += 1

    print(f"\n✅ 完成。重命名 {changed} 个，跳过 {skipped} 个。")


if __name__ == "__main__":
    main()

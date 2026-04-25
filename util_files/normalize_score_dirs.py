#!/usr/bin/env python3
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('scores_dir', type=Path)
args = parser.parse_args()

scores_dir = args.scores_dir
if not scores_dir.is_dir():
    raise SystemExit(f'{scores_dir} is not a directory')

for child in sorted(scores_dir.iterdir()):
    if not child.is_dir():
        continue
    name = child.name
    if name == 'Any-BFree-Online' or name == 'Any_BFree-Online':
        target = scores_dir / 'BFree-Online_BFree-Online'
    elif name.startswith('Any-'):
        target = scores_dir / f"cospy-inthewild_{name[4:]}"
    elif name.startswith('Any_'):
        target = scores_dir / f"cospy-inthewild_{name[4:]}"
    else:
        continue
    if target.exists():
        print(f'Skip existing target: {target.name}')
        continue
    print(f'Rename {name} -> {target.name}')
    child.rename(target)

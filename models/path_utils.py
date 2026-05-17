import os
from pathlib import Path


DEFAULT_CODES_ROOTS = (
    "/root/autodl-tmp/codes",
    "/gsdata/home/crx/jw/codes",
)


def get_codes_root():
    env_root = os.environ.get("SPACE_CODES_ROOT")
    if env_root:
        return env_root

    for root in DEFAULT_CODES_ROOTS:
        if Path(root).exists():
            return root

    return DEFAULT_CODES_ROOTS[0]


def get_model_root():
    return os.environ.get("SPACE_MODEL_ROOT", os.path.join(get_codes_root(), "model_pth"))


def model_path(*parts):
    return os.path.join(get_model_root(), *parts)

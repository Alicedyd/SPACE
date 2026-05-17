import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from .path_utils import model_path


PE_DIMS = {
    "PE-Core-T16-384": 512,
    "PE-Core-S16-384": 512,
    "PE-Core-B16-224": 1024,
    "PE-Core-L14-336": 1024,
    "PE-Core-G14-448": 1280,
}


def _import_pe():
    perception_models_path = os.environ.get("PERCEPTION_MODELS_PATH")
    if perception_models_path and perception_models_path not in sys.path:
        sys.path.insert(0, perception_models_path)

    try:
        import core.vision_encoder.pe as pe
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Could not import perception_models. Set PERCEPTION_MODELS_PATH to the "
            "cloned facebookresearch/perception_models directory, or add it to "
            "PYTHONPATH. Example: export "
            f"PERCEPTION_MODELS_PATH={model_path('perception_models')}"
        ) from exc

    return pe


class PEModel(nn.Module):
    """Perception Encoder with a frozen image tower and a linear probing head."""

    def __init__(self, name, num_classes=1, pretrained=True, checkpoint_path=None):
        super().__init__()
        if name not in PE_DIMS:
            raise ValueError(f"Unsupported PE model: {name}. Valid names: {list(PE_DIMS)}")

        pe = _import_pe()
        self.model = pe.CLIP.from_config(
            name,
            pretrained=pretrained,
            checkpoint_path=checkpoint_path,
        )

        for param in self.model.parameters():
            param.requires_grad = False

        self.fc = nn.Linear(PE_DIMS[name], num_classes)

    def forward(self, x, return_feature=False, return_tokens=False):
        with torch.no_grad():
            features = self.model.encode_image(x, normalize=False)
        features = F.normalize(features.float(), dim=-1)
        output = self.fc(features)

        if return_feature and return_tokens:
            return features, None, output
        if return_feature:
            return features, output
        if return_tokens:
            return None, None, output
        return output

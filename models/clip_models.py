from .clip import clip 
from PIL import Image
import torch.nn as nn


CHANNELS = {
    "RN50": 1024,
    "RN101": 1024,
    "RN50x4": 640,
    "RN50x16": 768,
    "RN50x64": 1024,
    "ViT-B/32": 512,
    "ViT-B/16": 512,
    "ViT-L/14": 768,
    "ViT-L/14@336px": 768
}

class CLIPModel(nn.Module):
    def __init__(self, name, num_classes=1):
        super(CLIPModel, self).__init__()

        self.model, self.preprocess = clip.load(name, device="cpu") # self.preprecess will not be used during training, which is handled in Dataset class 
        # CRITICAL: Freeze ALL CLIP parameters first
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.feature_projector = nn.Linear(CHANNELS[name], 1024)
        # Final classification layer uses 1024-dim features
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x, return_feature=False, return_tokens=False):
        features = self.feature_projector(self.model.encode_image(x))
        output = self.fc(features)
        
        if return_feature and return_tokens:
            # For consistency with DINOv2, return features, tokens (None for CLIP), and output
            return features, None, output
        
        elif return_feature:
            # Return features and output
            return features, output
        elif return_tokens:
            # CLIP doesn't support token-wise features, return None for tokens
            return None, None, output
        else:
            # Return only output
            return output
    
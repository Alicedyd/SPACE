from threading import local
import torch
import torch.nn as nn
from transformers import AutoModel
import os

CHANNELS = {
    "dinov3_vitl16": 1024,
    "dinov3_vith16plus": 1280,
}

PATH = {
    "dinov3_vitl16": "/root/autodl-tmp/codes/model_pth/dinov3_backup/pth-vitl16-lvd/",
    "dinov3_vith16plus": "/root/autodl-tmp/codes/model_pth/dinov3_backup/pth-vith16plus-lvd/",
}


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.drop = nn.Dropout(0.3)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_features, out_features)

        # self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.fc1(self.drop(x))
        x = self.act(x)
        x = self.fc2(self.drop(x))
        return x


class DINOv3Model(nn.Module):
    def __init__(self, name, num_classes=1):
        super(DINOv3Model, self).__init__()
        print(f"Loading DINOv3 from local: {name}")

        local_dir = PATH[name]
        self.model = AutoModel.from_pretrained(local_dir, weights_only=False)
        self.fc = Mlp(CHANNELS[name], 512, num_classes)

    def forward(self, x, return_feature=False, return_tokens=False):
        features = self.model(pixel_values=x)

        if return_feature:
            return features[1], self.fc(features[1])

        return self.fc(features[1])

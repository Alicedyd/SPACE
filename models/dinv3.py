class DINOv3(nn.Module):
    def __init__(self, dinov3_path, finetune=True):
        super(DINOv3, self).__init__()
        self.dinov3 = AutoModel.from_pretrained(dinov3_path, weights_only=False)

        if not finetune:  # 冻结版本
            self.dinov3.requires_grad_(False)
        else:  # LoRA 微调版本
            self.dinov3.requires_grad_(False)
            self._lora()

    def _lora(self):
        config = LoraConfig(
            r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj"]
        )
        for i in range(0, 24):
            layer = self.dinov3.layer[i]
            wrapped = get_peft_model(layer, config)
            self.dinov3.layer[i] = wrapped

    def forward(self, x):
        feats = self.dinov3(pixel_values=x)
        feat_last = feats[0]  # token-level features
        feat_cls = feats[1]  # CLS token
        return feat_last, feat_cls


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
        # x = self.fc(x)
        return x


class REM_Model(nn.Module):
    def __init__(self):
        super(REM_Model, self).__init__()
        self.norm = Norm()

        mode = "d3_h"  # 'd3_7b', 'd3_h', 'd3_l', 'clip'
        if mode == "d3_7b":
            dinov3_path = "/data/liurq_data/weights/dinov3-7b"
            self.fc = Mlp(4096, 512, 2)  # 用微调后的 cls 特征做分类
            self.model_finetune = DINOv3(dinov3_path, finetune=True)
            self.norm = Norm(mode="imagenet")
        elif mode == "d3_h":
            dinov3_path = "/data/liurq_data/weights/dinov3-h"
            self.fc = Mlp(1280, 512, 2)  # 用微调后的 cls 特征做分类
            self.model_finetune = DINOv3(dinov3_path, finetune=True)
            self.norm = Norm(mode="imagenet")
        elif mode == "d3_l":  # 'l'
            dinov3_path = "/data/liurq_data/weights/dinov3-large"
            self.fc = Mlp(1024, 512, 2)  # 用微调后的 cls 特征做分类
            self.model_finetune = DINOv3(dinov3_path, finetune=True)
            self.norm = Norm(mode="imagenet")
        elif mode == "clip":
            clip_path = "/data/liurq_data/weights/clip-vit-large-patch14"
            self.fc = Mlp(1024, 512, 2)  # 用微调后的 cls 特征做分类
            self.model_finetune = CLIPVit(clip_path)
            self.norm = Norm(mode="clip")
        elif mode == "d2_l":  # 'd2_l'
            dinov2_path = "/data/liurq_data/weights/dinov2-large"
            self.fc = Mlp(1024, 512, 2)  # 用微调后的 cls 特征做分类
            self.model_finetune = DINOv2(dinov2_path)
            self.norm = Norm(mode="imagenet")
        # 冻结版
        # self.dinov3_freeze = DINOv3(dinov3_path, finetune=False)
        self._set_trainable_params()

    def _set_trainable_params(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"[Trainable] {name}: {param.shape}")

    def forward(self, x):
        # 输入: [B, 3, 224, 224]
        # x = self.norm(x, mode='imagenet')
        x = self.norm(x)

        # 特征提取
        # feat_last_frozen, feat_cls_frozen = self.dinov3_freeze(x)      # 冻结特征

        feat_last_tuned, feat_cls_tuned = self.model_finetune(x)  # 微调特征
        feat_last_frozen, feat_cls_frozen = feat_last_tuned, feat_cls_tuned
        # 分类输出 (只用微调后的 cls)
        out = self.fc(feat_cls_tuned)

        # 返回三部分：分类输出、冻结特征、微调特征
        return out, feat_cls_frozen, feat_cls_tuned

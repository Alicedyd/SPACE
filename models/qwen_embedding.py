import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import LoraConfig, get_peft_model, TaskType

# 定义默认路径，你可以根据实际情况修改
DEFAULT_PATHS = {
    "2B": "/root/autodl-tmp/codes/model_pth/qwen3-vl-embedding/",
}


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=512, out_features=1, p=0.3):
        super().__init__()
        self.drop = nn.Dropout(p)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(self.drop(x))
        x = self.act(x)
        x = self.fc2(self.drop(x))
        return x


class Qwen3VLEmbedder(nn.Module):
    """
    Qwen3-VL-Embedding Base Model (No LoRA).
    Can be used for Frozen Feature Extraction or Full Fine-tuning.
    """

    def __init__(
        self,
        local_path: str,
        num_classes: int = 1,
        hidden_dim: int = 512,
        dtype=torch.bfloat16,
        head_dropout: float = 0.3,
        normalize_feats: bool = False,
        freeze_backbone: bool = True,  # 默认为 Embedding 模式，冻结骨干
    ):
        super().__init__()
        self.local_path = local_path
        self.hidden_dim = hidden_dim
        self.normalize_feats = normalize_feats
        self.num_classes = num_classes

        print(f"Loading Qwen3-VL-Embedding (Base) from local: {local_path}")
        self.processor = AutoProcessor.from_pretrained(local_path)

        # 加载基础模型
        self.backbone = AutoModelForImageTextToText.from_pretrained(
            local_path,
            dtype=dtype,
            device_map=None,
        )

        # 自动推断 feature dim
        self.feature_dim = getattr(self.backbone.config, "hidden_size", None)
        if self.feature_dim is None and hasattr(self.backbone.config, "text_config"):
            self.feature_dim = getattr(
                self.backbone.config.text_config, "hidden_size", None
            )
        if self.feature_dim is None:
            raise ValueError("Cannot infer feature_dim from model config.")

        # 冻结或解冻骨干网络
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
            print("Backbone frozen.")
        else:
            print("Backbone trainable (Full Fine-tuning).")

        # 分类头 (始终可训练)
        self.classifier = Mlp(
            in_features=self.feature_dim,
            hidden_features=self.hidden_dim,
            out_features=self.num_classes,
            p=head_dropout,
        )

        self.eos_token_id = self.processor.tokenizer.eos_token_id
        print(f"[Init] feature_dim={self.feature_dim}, hidden_dim={self.hidden_dim}")

    @property
    def device(self):
        return self.backbone.device

    def extract_features(self, images, text_prompt=" "):
        # 构造对话格式
        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": text_prompt},
                    ],
                }
            ]
            for _ in images
        ]

        texts = [
            self.processor.apply_chat_template(
                m, tokenize=False, add_generation_prompt=True
            )
            for m in messages
        ]

        # Tokenize and process images
        inputs = self.processor(
            text=texts, images=images, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Forward pass
        # 注意：如果全量微调，这里会计算梯度
        outputs = self.backbone(**inputs, output_hidden_states=True, return_dict=True)

        hs = outputs.hidden_states[-1]  # [B, seq, D]
        input_ids = inputs["input_ids"]  # [B, seq]

        # 找到最后一个 EOS token 的位置进行 Pooling
        eos_mask = (input_ids == self.eos_token_id).int()
        pos = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
        # 这是一个简单的找最后一个EOS的逻辑，如果某些样本没有EOS可能会报错，但在chat模板下通常会有
        eos_idx = (eos_mask * pos).max(dim=1).values

        feats = hs[
            torch.arange(hs.size(0), device=hs.device), eos_idx
        ].float()  # [B, D]

        if self.normalize_feats:
            feats = F.normalize(feats, p=2, dim=-1)

        return feats

    def forward(
        self, images, text_prompt=" ", return_feature=False, return_tokens=False
    ):
        # 兼容 dinov3 的接口 return_tokens (虽然这里没用到)
        feats = self.extract_features(images, text_prompt=text_prompt)
        logits = self.classifier(feats).squeeze(-1)  # [B]

        if return_feature:
            return feats, logits  # 注意通常 Dinov3 返回的是 (features, logits)
        return logits

    def get_preprocessing_transforms(self):
        # Qwen VL 内部处理 transform，这里返回 None 或做特定处理
        return None


class Qwen3VLEmbedderLoRA(Qwen3VLEmbedder):
    """
    Qwen3-VL-Embedding with LoRA adapters.
    Inherits extraction logic from Base, adds LoRA initialization.
    """

    def __init__(
        self,
        local_path: str,
        num_classes: int = 1,
        hidden_dim: int = 512,
        dtype=torch.bfloat16,
        lora_r: int = 32,
        lora_alpha: int = 32,
        lora_dropout: float = 0.0,
        target_modules=(
            "q_proj",
            "v_proj",
            "k_proj",
            "up_proj",
            "down_proj",
            "gate_proj",
        ),
        head_dropout: float = 0.3,
        normalize_feats: bool = False,
    ):
        # 初始化 Base，强制 freeze_backbone=False (因为我们需要 PEFT 来接管 freezing 逻辑)
        # 但为了避免重复加载，我们可以先调用 super，然后 apply LoRA
        super().__init__(
            local_path=local_path,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dtype=dtype,
            head_dropout=head_dropout,
            normalize_feats=normalize_feats,
            freeze_backbone=False,  # 先不冻结，让 PEFT 处理
        )

        print(f"Applying LoRA to Qwen3-VL: r={lora_r}, alpha={lora_alpha}")

        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=list(target_modules),
            bias="none",
        )

        # 将 backbone 包装成 PeftModel
        self.backbone = get_peft_model(self.backbone, lora_cfg)

        # 确保只有 LoRA 参数和 Classifier 是可训练的
        for n, p in self.backbone.named_parameters():
            if "lora_" not in n:
                p.requires_grad_(False)
            else:
                p.requires_grad_(True)

        # 确保 Classifier 在正确的设备上
        self.classifier.to(self.device)

    def print_trainable(self):
        trainable, total = 0, 0
        # 统计 backbone (LoRA) + classifier
        for p in self.parameters():
            n = p.numel()
            total += n
            if p.requires_grad:
                trainable += n
        print(
            f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.4f}%)"
        )

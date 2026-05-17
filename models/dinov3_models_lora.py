import torch
import torch.nn as nn
from .dinov3_models import DINOv3Model


class DINOv3ModelWithLoRA(nn.Module):
    def __init__(
        self,
        name,
        num_classes=1,
        lora_rank=8,
        lora_alpha=16,
        lora_targets=None,
        local_files_only=False,
        model_dir=None,
        huggingface_path=None,
    ):
        super(DINOv3ModelWithLoRA, self).__init__()
        # Initialize the base DINOv3 model
        self.base_model = DINOv3Model(
            name,
            num_classes,
            huggingface_path=huggingface_path,
        )
        # self.base_model = DINOv3Model(
        #     name=name,
        #     num_classes=num_classes,
        #     # local_files_only=local_files_only,
        #     model_dir=model_dir,
        #     huggingface_path=huggingface_path
        # )
        # Store name for reference
        self.name = name
        # Import LoRA utilities
        try:
            from .lora import apply_lora_to_linear_layers, get_lora_params
        except ImportError:
            raise ImportError(
                "LoRA module not found. Please ensure the LoRA utilities are available."
            )
        # Default target modules for LoRA if not provided (target key substrings in module names)
        if lora_targets is None:
            # lora_targets = ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"]
            lora_targets = ["attention.q_proj", "attention.k_proj", "attention.v_proj"]

        # lora_targets = [
        #     # Attention projections
        #     "attention.q_proj",
        #     "attention.k_proj",
        #     "attention.v_proj",
        #     # "attention.o_proj",
        #     # MLP projections
        #     "mlp.up_proj",
        #     "mlp.down_proj",
        # ]
        print(
            f"Adding LoRA to DINOv3 model '{name}' (rank={lora_rank}, alpha={lora_alpha})"
        )
        print(f"LoRA target modules: {lora_targets}")
        # Apply LoRA to the base model's linear layers
        self.base_model.model = apply_lora_to_linear_layers(
            self.base_model.model,
            rank=lora_rank,
            alpha=lora_alpha,
            target_modules=lora_targets,
            trainable_orig=False,
        )
        # Store a callable to retrieve LoRA params
        self._get_lora_params = lambda: get_lora_params(self.base_model.model)

    def get_trainable_params(self):
        """
        Get the list of trainable parameters (LoRA parameters + classification head).
        Returns:
            list[nn.Parameter]: Parameters to be optimized during training.
        """
        lora_params = self._get_lora_params()
        fc_params = list(self.base_model.fc.parameters())
        total_lora = sum(p.numel() for p in lora_params)
        total_fc = sum(p.numel() for p in fc_params)
        print(f"LoRA parameters: {total_lora:,}, Classifier parameters: {total_fc:,}")
        return list(lora_params) + fc_params

    def forward(self, x, return_feature=False, return_tokens=False):
        """
        Forward pass through the LoRA-augmented DINOv3 model.
        Args:
            x (torch.Tensor): Input tensor.
            return_feature (bool): Whether to return backbone features along with output.
            return_tokens (bool): (Unused) Whether to return token-level features.
        """
        # Simply forward to the base model (which handles return_feature logic)
        return self.base_model(
            x, return_feature=return_feature, return_tokens=return_tokens
        )

    def get_preprocessing_transforms(self):
        """
        Get preprocessing transforms (normalization) from the base model.
        """
        return self.base_model.get_preprocessing_transforms()

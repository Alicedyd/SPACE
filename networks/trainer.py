import functools
import math
import torch
import torch.nn as nn
from networks.base_model import BaseModel, init_weights
import sys
from models import get_model
import torchvision.transforms.functional as F
import torch.nn.functional

from pytorch_metric_learning import losses


class SoftContrastiveLoss(nn.Module):
    def __init__(self, pos_margin=0.0, neg_margin=1.0, similarity_threshold=0.5):
        """Contrastive loss with soft labels based on similarity scores.

        Args:
            pos_margin (float, optional): target margin of positive sample pair. Defaults to 0.0.
            neg_margin (float, optional): target margin of negative sample pair. Defaults to 1.0.
            similarity_threshold (float, optional): positive sample pair threshold, pair with similarity over this threshold
                                                    are treated as positive sample pair. Defaults to 0.5.

        Returns:
            loss (torch.Tensor): computed loss value.
        """

        super(SoftContrastiveLoss, self).__init__()
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.similarity_threshold = similarity_threshold

    def forward(self, features, labels):
        """Forward pass for the soft contrastive loss.

        Args:
            features : [N, D] features of samples, where N is the number of samples and D is the feature dimension
            labels : [N,] tensor of labels corresponding to each feature
        """
        # Normalize features
        features = torch.nn.functional.normalize(features, p=2, dim=1)

        # Compute pairwise cosine similarity
        distances = torch.cdist(features, features, p=2)  # [N, N]

        # Compute label similarities (1 - absoulute difference)
        label_diff = torch.abs(labels.unsqueeze(0) - labels.unsqueeze(1))  # [N, N]
        label_sim = 1 - label_diff

        # Get upper triangular indices to avoid double counting
        upper_tri_indices = torch.triu_indices(
            len(labels), len(labels), offset=1, device=features.device
        )
        pair_distances = distances[
            upper_tri_indices[0], upper_tri_indices[1]
        ]  # [M,] where M is number of pairs
        pair_similarities = label_sim[
            upper_tri_indices[0], upper_tri_indices[1]
        ]  # [M,]

        # Define positive and negative pairs based on similarity threshold
        pos_mask = pair_similarities >= self.similarity_threshold
        neg_mask = pair_similarities <= (1 - self.similarity_threshold)

        pos_loss = torch.tensor(0.0, device=features.device)
        neg_loss = torch.tensor(0.0, device=features.device)

        # Compute positive loss
        if pos_mask.sum() > 0:
            pos_distances = pair_distances[pos_mask]
            pos_weights = pair_similarities[pos_mask]
            pos_loss = (
                pos_weights * torch.relu(pos_distances - self.pos_margin)
            ).mean()

        # Compute negative loss
        if neg_mask.sum() > 0:
            neg_distances = pair_distances[neg_mask]
            neg_weights = pair_similarities[neg_mask]
            neg_loss = (
                neg_weights * torch.relu(neg_distances - self.neg_margin)
            ).mean()

        return pos_loss + neg_loss


class TokenWiseContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(TokenWiseContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, token_features, token_masks):
        """
        Args:
            token_features: Tensor of shape [B, N, D] where B is batch size,
                          N is number of tokens, D is feature dim
            token_masks: Binary tensor of shape [B, N] where 1 indicate
                        fake tokens and 0 indicates real tokens
        """
        # Ensure proper shape
        B, N, D = token_features.shape
        token_features = token_features.reshape(-1, D)  # [B*N, D]

        # Create token labels based on image labels
        token_labels = token_masks.reshape(-1)  # [B*N]

        # Normalize features
        token_features = torch.nn.functional.normalize(
            token_features, p=2, dim=1
        )  # Use full path and correct parameters

        # Get real and fake token indices
        real_indices = (token_labels == 0).nonzero(as_tuple=True)[0]
        fake_indices = (token_labels == 1).nonzero(as_tuple=True)[0]

        # If either real or fake tokens are missing, return zero loss
        if len(real_indices) == 0 or len(fake_indices) == 0:
            return torch.tensor(0.0, device=token_features.device)

        # Get features of real and fake tokens
        real_features = token_features[real_indices]
        fake_features = token_features[fake_indices]

        # Compute pairwise similarities
        similarity_matrix = (
            torch.matmul(real_features, fake_features.T) / self.temperature
        )

        # Compute contrastive loss
        real_to_fake_max, _ = torch.max(similarity_matrix, dim=1)
        real_to_fake_loss = -torch.log(
            1 - torch.sigmoid(real_to_fake_max) + 1e-6
        ).mean()

        fake_to_real_max, _ = torch.max(similarity_matrix.T, dim=1)
        fake_to_real_loss = -torch.log(
            1 - torch.sigmoid(fake_to_real_max) + 1e-6
        ).mean()

        # Combined loss
        total_loss = (real_to_fake_loss + fake_to_real_loss) / 2

        return total_loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        ce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = self.alpha * (1 - p_t) ** self.gamma * ce
        return loss.mean()


class Trainer(BaseModel):
    def name(self):
        return "Trainer"

    def __init__(self, opt):
        super(Trainer, self).__init__(opt)
        self.opt = opt

        # Add gradient accumulation parameters
        self.accumulation_steps = (
            opt.accumulation_steps if hasattr(opt, "accumulation_steps") else 1
        )
        self.current_step = 0  # Track steps for gradient accumulation

        lora_args = {}
        if hasattr(opt, "lora_rank"):
            lora_args["lora_rank"] = opt.lora_rank
        if hasattr(opt, "lora_alpha"):
            lora_args["lora_alpha"] = opt.lora_alpha
        if hasattr(opt, "lora_targets"):
            lora_args["lora_targets"] = (
                opt.lora_targets.split(",") if opt.lora_targets else None
            )
        if hasattr(opt, "model_dir") and opt.model_dir:
            lora_args["huggingface_path"] = opt.model_dir

        self.model = get_model(name=opt.arch, **lora_args)

        # Determine training mode based on model name
        using_lora = (
            opt.arch.startswith("CLIP-LoRA:")
            or opt.arch.startswith("DINOv2-LoRA:")
            or opt.arch.startswith("DINOv3-LoRA:")
            or opt.arch.startswith("Qwen3-LoRA:")
        )
        using_fullfinetune = opt.arch.startswith(
            "CLIP-FullFinetune:"
        ) or opt.arch.startswith("DINOv2-FullFinetune:")

        # Initialize final layer weights
        if opt.arch.startswith("DINOv2-LoRA:"):
            torch.nn.init.normal_(
                self.model.base_model.fc.weight.data, 0.0, opt.init_gain
            )
            if opt.use_logit_adjustment:
                torch.nn.init.constant_(self.model.base_model.fc.bias.data, -3.89)
        elif opt.arch.startswith("DINOv3-LoRA"):
            pass
        elif opt.arch.startswith("Qwen3"):
            pass
        else:
            torch.nn.init.normal_(self.model.fc.weight.data, 0.0, opt.init_gain)

        # Parameter selection based on training mode
        if opt.fix_backbone and not using_fullfinetune:
            # Fixed backbone training (original behavior)
            params = []
            for name, p in self.model.named_parameters():
                if name == "fc.weight" or name == "fc.bias":
                    params.append(p)
                else:
                    p.requires_grad = False
            print(
                "Training with fixed backbone - only final layer parameters will be updated"
            )

        elif using_lora:
            # LoRA training
            if hasattr(self.model, "get_trainable_params"):
                params = self.model.get_trainable_params()
                print(
                    "Training with LoRA - only LoRA and final layer parameters will be updated"
                )
            else:
                raise ValueError("LoRA model should have get_trainable_params method")

        elif using_fullfinetune:
            # Full fine-tuning (all parameters)
            params = self.model.parameters()
            print(
                "Training with full fine-tuning - all model parameters will be updated"
            )

        else:
            # Legacy behavior for regular models without explicit full fine-tuning
            if not opt.fix_backbone:
                print(
                    "Warning: Your backbone is not fixed. Are you sure you want to proceed?"
                )
                print(
                    "If this is a mistake, enable the --fix_backbone command during training and rerun"
                )
                print(
                    "For explicit full fine-tuning, consider using 'CLIP-FullFinetune:' or 'DINOv2-FullFinetune:' model names"
                )
                import time

                time.sleep(3)
                params = self.model.parameters()
            else:
                params = []
                for name, p in self.model.named_parameters():
                    if name == "fc.weight" or name == "fc.bias":
                        params.append(p)
                    else:
                        p.requires_grad = False

        # Print trainable parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        print(f"\nTrainable parameters summary:")
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Trainable parameters: {trainable_params:,}")
        print(f"  - Trainable ratio: {trainable_params / total_params * 100:.2f}%")

        print(f"\nTrainable parameter names:")
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                print(f"  - {name}: {p.numel():,} parameters")

        if opt.optim == "adam":
            self.optimizer = torch.optim.AdamW(
                params,
                lr=opt.lr,
                betas=(opt.beta1, 0.999),
                weight_decay=opt.weight_decay,
            )
        elif opt.optim == "sgd":
            self.optimizer = torch.optim.SGD(
                params, lr=opt.lr, momentum=0.0, weight_decay=opt.weight_decay
            )
        else:
            raise ValueError("optim should be [adam, sgd]")

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, eta_min=opt.lr * 0.001, T_max=1000
        )
        if opt.use_focal_loss:
            self.loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        else:
            self.loss_fn = nn.BCEWithLogitsLoss()

        # Regular Contrastive Learning
        self.contrastive = opt.contrastive
        if self.contrastive:
            self.contrastive_loss_fn = losses.ContrastiveLoss(
                pos_margin=0.0, neg_margin=1.0
            )
            # self.contrastive_loss_fn = SoftContrastiveLoss(pos_margin=0.0, neg_margin=1.0, similarity_threshold=0.5)
            self.contrastive_alpha = 0.5

        # Token-wise Contrastive Learning
        self.token_contrastive = (
            opt.token_contrastive if hasattr(opt, "token_contrastive") else False
        )
        if self.token_contrastive:
            self.token_contrastive_loss_fn = TokenWiseContrastiveLoss(
                temperature=opt.temperature if hasattr(opt, "temperature") else 0.07
            )
            self.token_contrastive_alpha = (
                opt.token_contrastive_alpha
                if hasattr(opt, "token_contrastive_alpha")
                else 0.3
            )

        if hasattr(opt, "device"):
            self.device = opt.device
        self.model.to(self.device)

    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] /= 10.0
            if param_group["lr"] < min_lr:
                return False
        return True

    def set_input(self, input):
        components = ["real", "real_resized", "fake", "fake_resized"]

        input_stack = []
        for key in components:
            if input[key] is not None:
                input_stack.append(input[key])
        self.input = torch.cat(input_stack, dim=0).to(self.device)

        # if "soft_label" in input.keys():
        #     self.label = torch.cat([input['soft_label'][key] for key in components], dim=0).to(self.device).float()

        # else:
        #     LABELS = {
        #         "real": 0,
        #         "real_resized": 0,
        #         "fake": 1,
        #         "fake_resized": 1,
        #     }
        #     label_stack = []
        #     for key in components:
        #         if input[key] is not None:
        #             label_stack += [LABELS[key]] * len(input[key])
        #     self.label = torch.tensor(label_stack).to(self.device).float()

        LABELS = {
            "real": 0,
            "real_resized": 0,
            "fake": 1,
            "fake_resized": 1,
        }
        label_stack = []
        for key in components:
            if input[key] is not None:
                label_stack += [LABELS[key]] * len(input[key])
        self.label = torch.tensor(label_stack).to(self.device).float()

    def forward(self):
        if self.contrastive and self.token_contrastive:
            # Get global features, token features, and output
            self.feature, self.token_features, self.output = self.model(
                self.input, return_feature=True, return_tokens=True
            )
        elif self.contrastive:
            # Get only global features and output
            self.feature, self.output = self.model(self.input, return_feature=True)
        elif self.token_contrastive:
            # Get only token features and output
            _, self.token_features, self.output = self.model(
                self.input, return_feature=False, return_tokens=True
            )
        else:
            # Get only output
            self.output = self.model(self.input)

        if hasattr(self.output, "view"):
            self.output = self.output.view(-1).unsqueeze(1)

        self.output = self.output

    def get_loss(self):
        return self.loss_fn(self.output.squeeze(1), self.label)

    def optimize_parameters(self):
        self.current_step += 1
        self.forward()

        # Calculate classification loss
        cls_loss = self.loss_fn(self.output.squeeze(1), self.label)

        # Initialize total loss with classification loss
        total_loss = cls_loss

        # Add global contrastive loss if enabled
        if self.contrastive:
            contrastive_loss = self.contrastive_loss_fn(self.feature, self.label)
            total_loss = (
                1 - self.contrastive_alpha
            ) * total_loss + self.contrastive_alpha * contrastive_loss
            # contrastive_alpha = min(self.contrastive_alpha * (self.current_step / 10000), self.contrastive_alpha)
            # total_loss = (1 - contrastive_alpha) * cls_loss + contrastive_alpha * contrastive_loss

        # Add token-wise contrastive loss if enabled
        if self.token_contrastive:
            # Create token masks based on image labels
            B = self.label.size(0)
            N = self.token_features.size(1)  # Number of tokens per image

            # Repeat each image label for all of its tokens
            token_masks = self.label.unsqueeze(1).repeat(1, N)

            token_contrastive_loss = self.token_contrastive_loss_fn(
                self.token_features, token_masks
            )
            total_loss = (
                1 - self.token_contrastive_alpha
            ) * total_loss + self.token_contrastive_alpha * token_contrastive_loss

        # Save the final loss
        self.loss = total_loss

        # Apply gradient accumulation
        self.loss = self.loss / self.accumulation_steps
        self.loss.backward()

        if self.current_step % self.accumulation_steps == 0:
            # Update parameters
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()  # Reset gradients after update

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    # Handle remaining gradients at the end of epoch
    def finalize_epoch(self):
        if self.current_step % self.accumulation_steps != 0:
            # Update with remaining gradients
            self.optimizer.step()
            self.optimizer.zero_grad()

from .clip_models import CLIPModel
from .imagenet_models import ImagenetModel
from .dinov2_models import DINOv2Model
from .clip_models_lora import CLIPModelWithLoRA
from .dinov2_models_lora import DINOv2ModelWithLoRA
from .dinov3_models import DINOv3Model
from .dinov3_models_lora import DINOv3ModelWithLoRA
from .qwen_embedding import Qwen3VLEmbedderLoRA, Qwen3VLEmbedder
from .pe_models import PEModel
from .path_utils import model_path

from .align import resnet50

VALID_NAMES = [
    "align",
    "Imagenet:resnet18",
    "Imagenet:resnet34",
    "Imagenet:resnet50",
    "Imagenet:resnet101",
    "Imagenet:resnet152",
    "Imagenet:vgg11",
    "Imagenet:vgg19",
    "Imagenet:swin-b",
    "Imagenet:swin-s",
    "Imagenet:swin-t",
    "Imagenet:vit_b_16",
    "Imagenet:vit_b_32",
    "Imagenet:vit_l_16",
    "Imagenet:vit_l_32",
    "CLIP:RN50",
    "CLIP:RN101",
    "CLIP:RN50x4",
    "CLIP:RN50x16",
    "CLIP:RN50x64",
    "CLIP:ViT-B/32",
    "CLIP:ViT-B/16",
    "CLIP:ViT-L/14",
    "CLIP:ViT-L/14@336px",
    "DINOv2:dinov2_vits14",
    "DINOv2:dinov2_vitb14",
    "DINOv2:dinov2_vitl14",
    "DINOv2:dinov2_vitg14",
    "DINOv2:dinov2_vitl14_register",  # Added support for register model
    # Full Fine-tuning Models
    "CLIP-FullFinetune:RN50",
    "CLIP-FullFinetune:RN101",
    "CLIP-FullFinetune:RN50x4",
    "CLIP-FullFinetune:RN50x16",
    "CLIP-FullFinetune:RN50x64",
    "CLIP-FullFinetune:ViT-B/32",
    "CLIP-FullFinetune:ViT-B/16",
    "CLIP-FullFinetune:ViT-L/14",
    "CLIP-FullFinetune:ViT-L/14@336px",
    "DINOv2-FullFinetune:dinov2_vits14",
    "DINOv2-FullFinetune:dinov2_vitb14",
    "DINOv2-FullFinetune:dinov2_vitl14",
    "DINOv2-FullFinetune:dinov2_vitg14",
    "DINOv2-FullFinetune:dinov2_vitl14_register",
    "CLIP-LoRA:RN50",
    "CLIP-LoRA:RN101",
    "CLIP-LoRA:ViT-B/32",
    "CLIP-LoRA:ViT-B/16",
    "CLIP-LoRA:ViT-L/14",
    "CLIP-LoRA:ViT-L/14@336px",
    "DINOv2-LoRA:dinov2_vits14",
    "DINOv2-LoRA:dinov2_vitb14",
    "DINOv2-LoRA:dinov2_vitl14",
    "DINOv2-LoRA:dinov2_vitg14",
    "DINOv2-LoRA:dinov2_vitl14_register",  # Added support for register model with LoRA
    "DINOv3:dinov3_vitl16",
    "DINOv3:dinov3_vith16plus",
    "DINOv3-LoRA:dinov3_vitl16",
    "DINOv3-LoRA:dinov3_vith16plus",
    "Qwen3-Embedding:2B",
    "Qwen3-LoRA:2B",
    "PE:PE-Core-T16-384",
    "PE:PE-Core-S16-384",
    "PE:PE-Core-B16-224",
    "PE:PE-Core-L14-336",
    "PE:PE-Core-G14-448",
]


def get_model(
    name, lora_rank=8, lora_alpha=1.0, lora_targets=None, huggingface_path=None
):
    """
    Get a model instance based on model name and parameters.

    Args:
        name (str): Model name from VALID_NAMES list
        lora_rank (int, optional): Rank for LoRA models. Defaults to 8.
        lora_alpha (float, optional): Alpha value for LoRA models. Defaults to 1.0.
        lora_targets (list, optional): Target modules for LoRA. Defaults to None.
        huggingface_path (str, optional): Path to HuggingFace model. Defaults to None.
                                   For DINOv2 models: 224, 504 (register model), or 518 (standard models).

    Returns:
        Model instance
    """
    assert name in VALID_NAMES, (
        f"Invalid model name: {name}. Valid names: {VALID_NAMES}"
    )

    # Print initialization information
    print("\n" + "=" * 50)
    print(f"Initializing model: {name}")
    print(f"Parameters:")
    if "LoRA" in name:
        print(f"  - LoRA rank: {lora_rank}")
        print(f"  - LoRA alpha: {lora_alpha}")
        if lora_targets:
            print(f"  - LoRA targets: {lora_targets}")
        else:
            print(f"  - LoRA targets: default")
    elif "FullFinetune" in name:
        print(f"  - Training mode: Full Fine-tuning (all parameters)")

    # Print huggingface path information if applicable
    if huggingface_path:
        print(f"  - HuggingFace path: {huggingface_path}")

    if name == "align":
        return resnet50(num_classes=1, stride0=1, dropout=0.5)

    if name.startswith("PE:"):
        model_family = "Perception Encoder"
        model_type = name[3:]
        print(f"  - Model family: {model_family}")
        print(f"  - Model type: {model_type}")
        if huggingface_path:
            print(f"  - PE checkpoint path: {huggingface_path}")

        model = PEModel(model_type, checkpoint_path=huggingface_path)
        print("  - Successfully initialized PE linear probing model")
        return model

    if name.startswith("Imagenet:"):
        model_family = "ImageNet"
        model_type = name[9:]
        print(f"  - Model family: {model_family}")
        print(f"  - Model type: {model_type}")

        model = ImagenetModel(model_type)
        print(f"  - Successfully initialized with custom image size")
        return model

    elif name.startswith("CLIP:"):
        model_family = "CLIP"
        model_type = name[5:]
        print(f"  - Model family: {model_family}")
        print(f"  - Model type: {model_type}")

        model = CLIPModel(model_type)
        print(f"  - Successfully initialized with custom image size")
        return model

    elif name.startswith("CLIP-LoRA:"):
        model_family = "CLIP with LoRA"
        model_type = name[10:]
        print(f"  - Model family: {model_family}")
        print(f"  - Model type: {model_type}")

        model = CLIPModelWithLoRA(
            model_type,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_targets=lora_targets,
        )
        print(f"  - Successfully initialized with custom image size")
        return model

    elif name.startswith("CLIP-FullFinetune:"):
        model_family = "CLIP Full Fine-tuning"
        model_type = name[18:]
        print(f"  - Model family: {model_family}")
        print(f"  - Model type: {model_type}")

        model = CLIPModel(model_type)
        print(f"  - Successfully initialized for full fine-tuning")
        return model

    elif name.startswith("DINOv2:"):
        model_family = "DINOv2"
        model_type = name[7:]
        print(f"  - Model family: {model_family}")
        print(f"  - Model type: {model_type}")

        if model_type == "dinov2_vitl14" and huggingface_path is None:
            huggingface_path = "/Youtu_Pangu_Security/cusmochen/pretrained/huggingface/models/models--facebook--dinov2-large/snapshots/47b73eefe95e8d44ec3623f8890bd894b6ea2d6c/"
            print(
                f"  - Using default HuggingFace path for dinov2_vitl14: {huggingface_path}"
            )

        # Handle specific configurations for DINOv2 models
        if model_type == "dinov2_vitl14_register" and huggingface_path is None:
            # For register model, default to the specific HuggingFace path if not specified
            huggingface_path = "/Youtu_Pangu_Security/cusmochen/pretrained/huggingface/models/models--facebook--dinov2-with-registers-large/snapshots/e4c89a4e05589de9b3e188688a303d0f3c04d0f3/"
            print(
                f"  - Using default HuggingFace path for register model: {huggingface_path}"
            )

        # Create DINOv2 model with optional parameters
        model = DINOv2Model(name=model_type, huggingface_path=huggingface_path)

        print(f"  - Successfully initialized DINOv2 model")
        return model

    elif name.startswith("DINOv2-LoRA:"):
        model_family = "DINOv2 with LoRA"
        model_type = name[12:]
        print(f"  - Model family: {model_family}")
        print(f"  - Model type: {model_type}")

        # Handle specific configurations for DINOv2-LoRA models
        if model_type == "dinov2_vitl14_register" and huggingface_path is None:
            # For register model, default to the specific HuggingFace path if not specified
            huggingface_path = "/Youtu_Pangu_Security/cusmochen/pretrained/huggingface/models/models--facebook--dinov2-with-registers-large/snapshots/e4c89a4e05589de9b3e188688a303d0f3c04d0f3/"
            print(
                f"  - Using default HuggingFace path for register model: {huggingface_path}"
            )

        # Create DINOv2 model with LoRA and optional parameters
        model = DINOv2ModelWithLoRA(
            name=model_type,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_targets=lora_targets,
            huggingface_path=huggingface_path,
        )

        print(f"  - Successfully initialized DINOv2-LoRA model")

        # Report trainable parameters information
        trainable_params = model.get_trainable_params()
        total_trainable_params = sum(p.numel() for p in trainable_params)
        print(f"  - Total trainable parameters: {total_trainable_params:,}")

        return model

    elif name.startswith("DINOv3:"):
        model_family = "DINOv3"
        model_type = name[7:]
        print(f"  - Model family: {model_family}")
        print(f"  - Model type: {model_type}")

        if model_type == "dinov3_vitl16" and huggingface_path is None:
            huggingface_path = model_path("dinov3", "pth-vitl16-lvd")
            print(
                f"  - Using default HuggingFace path for dinov3_vitl16: {huggingface_path}"
            )

        # Create DINOv2 model with optional parameters
        model = DINOv3Model(name=model_type, huggingface_path=huggingface_path)

        print(f"  - Successfully initialized DINOv3 model")
        return model

    elif name.startswith("DINOv3-LoRA:"):
        model_family = "DINOv3 with LoRA"
        model_type = name[12:]
        print(f"  - Model family: {model_family}")
        print(f"  - Model type: {model_type}")

        if model_type == "dinov3_vitl16" and huggingface_path is None:
            huggingface_path = model_path("dinov3", "pth-vitl16-lvd")
            print(
                f"  - Using default HuggingFace path for dinov3_vitl16: {huggingface_path}"
            )

        # Create DINOv2 model with LoRA and optional parameters
        model = DINOv3ModelWithLoRA(
            name=model_type,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_targets=lora_targets,
            huggingface_path=huggingface_path,
        )

        print(f"  - Successfully initialized DINOv3-LoRA model")

        # Report trainable parameters information
        trainable_params = model.get_trainable_params()
        total_trainable_params = sum(p.numel() for p in trainable_params)
        print(f"  - Total trainable parameters: {total_trainable_params:,}")

        return model

    elif name.startswith("Qwen3-Embedding:"):
        model_family = "Qwen3-VL Embedding"
        model_type = name.split(":")[-1]  # 获取 2B 或 7B
        print(f"  - Model family: {model_family}")
        print(f"  - Model type: {model_type}")

        # 如果没有指定 huggingface_path，设置默认路径
        if huggingface_path is None:
            # 请根据你的实际路径修改这里
            if model_type == "2B":
                huggingface_path = model_path("qwen3-vl-embedding")

            print(f"  - Using default path for Qwen: {huggingface_path}")

        model = Qwen3VLEmbedder(
            local_path=huggingface_path,
            num_classes=1,  # 假设是回归或二分类
            freeze_backbone=True,  # Embedding 模式默认冻结
        )
        print(f"  - Successfully initialized Qwen3 Embedding model")
        return model

    elif name.startswith("Qwen3-LoRA:"):
        model_family = "Qwen3-VL LoRA"
        model_type = name.split(":")[-1]
        print(f"  - Model family: {model_family}")
        print(f"  - Model type: {model_type}")

        # 如果没有指定 huggingface_path，设置默认路径
        if huggingface_path is None:
            # 请根据你的实际路径修改这里
            if model_type == "2B":
                huggingface_path = model_path("qwen3-vl-embedding")

            print(f"  - Using default path for Qwen: {huggingface_path}")

        # 设置 LoRA 参数，如果有传入的参数则使用，否则使用默认

        model = Qwen3VLEmbedderLoRA(
            local_path=huggingface_path,
            num_classes=1,
            lora_r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=lora_targets
            if lora_targets
            else ("q_proj", "v_proj", "k_proj", "up_proj", "down_proj", "gate_proj"),
        )

        print(f"  - Successfully initialized Qwen3 LoRA model")
        model.print_trainable()
        return model

    elif name.startswith("DINOv2-FullFinetune:"):
        model_family = "DINOv2 Full Fine-tuning"
        model_type = name[20:]
        print(f"  - Model family: {model_family}")
        print(f"  - Model type: {model_type}")

        if model_type == "dinov2_vitl14" and huggingface_path is None:
            huggingface_path = "/Youtu_Pangu_Security/cusmochen/pretrained/huggingface/models/models--facebook--dinov2-large/snapshots/47b73eefe95e8d44ec3623f8890bd894b6ea2d6c/"
            print(
                f"  - Using default HuggingFace path for dinov2_vitl14: {huggingface_path}"
            )

        # Handle specific configurations for DINOv2 models
        if model_type == "dinov2_vitl14_register" and huggingface_path is None:
            # For register model, default to the specific HuggingFace path if not specified
            huggingface_path = "/Youtu_Pangu_Security/cusmochen/pretrained/huggingface/models/models--facebook--dinov2-with-registers-large/snapshots/e4c89a4e05589de9b3e188688a303d0f3c04d0f3/"
            print(
                f"  - Using default HuggingFace path for register model: {huggingface_path}"
            )

        # Create DINOv2 model with optional parameters
        model = DINOv2Model(name=model_type, huggingface_path=huggingface_path)

        print(f"  - Successfully initialized DINOv2 model for full fine-tuning")
        return model

    else:
        raise ValueError(f"Unsupported model prefix in name: {name}")

    print("=" * 50 + "\n")

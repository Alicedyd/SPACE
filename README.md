# SPACE

SPACE is an AI-generated image detection training and evaluation project. It supports DINOv3, DINOv2, CLIP, Qwen, and Perception Encoder based detectors, plus merged evaluation over multiple checkpoints.

## Environment Paths

This project is used on multiple servers with different root directories. Source `setup_space_env.sh` before training or evaluation.

For the `gsdata` server:

```bash
cd /gsdata/home/crx/jw/codes/SPACE
source setup_space_env.sh gsdata
```

For the autodl server:

```bash
cd /root/autodl-tmp/codes/SPACE
source setup_space_env.sh autodl
```

Auto-detect mode:

```bash
source setup_space_env.sh
```

You can override any path when needed:

```bash
export SPACE_DATA_ROOT=/path/to/datasets
export SPACE_CKPT_ROOT=/path/to/ckpt
source setup_space_env.sh gsdata
```

The script exports:

```bash
SPACE_CODES_ROOT
SPACE_PROJECT_ROOT
SPACE_DATA_ROOT
SPACE_MODEL_ROOT
SPACE_CKPT_ROOT
SPACE_VAL_CONFIG_ROOT
PERCEPTION_MODELS_PATH
```

## Expected Layout

Typical `gsdata` layout:

```text
/gsdata/home/crx/jw/
  codes/
    SPACE/
    DDA/
    ckpt/
    model_pth/
    val_configs/
  datasets/
```

Typical autodl layout:

```text
/root/autodl-tmp/
  codes/
    SPACE/
    DDA/
    ckpt/
    model_pth/
    val_configs/
  datasets/
```

## Training

Activate the Python environment first:

```bash
conda activate DDA
cd "$SPACE_PROJECT_ROOT"
```

Train the SD branch:

```bash
bash scripts/train_dinov3_sd.sh -g 0 -c RGB -a 16 -n SD_RUN
```

Train the FLUX branch:

```bash
bash scripts/train_dinov3_flux.sh -g 0 -c RGB -a 16 -n FLUX_RUN
```

Train the CLIP branch:

```bash
bash scripts/train_clip.sh -g 0 -c RGB -a 16 -n CLIP_RUN
```

Checkpoints are written under:

```bash
$SPACE_CKPT_ROOT/checkpoints_SPACE
```

Find recent checkpoints:

```bash
find "$SPACE_CKPT_ROOT/checkpoints_SPACE" -name "*.pth" | sort | tail -20
```

## Single Model Evaluation

Evaluate a DINOv3 checkpoint:

```bash
bash scripts/validate_dinov3.sh \
  -k /path/to/checkpoint.pth \
  -r DINOv3_TEST \
  -g 0
```

Evaluate a CLIP checkpoint:

```bash
bash scripts/validate_clip.sh \
  -k /path/to/checkpoint.pth \
  -r CLIP_TEST \
  -g 0
```

Results are written under:

```bash
$SPACE_PROJECT_ROOT/result/SPACE
```

## Merged Evaluation

Run merged evaluation with SD, FLUX, and CLIP or PE checkpoints:

```bash
bash scripts/validate_merged.sh \
  -s /path/to/sd_checkpoint.pth \
  -f /path/to/flux_checkpoint.pth \
  -c /path/to/clip_or_pe_checkpoint.pth \
  -r SPACE_MERGED_TEST \
  -g 0
```

Results are written under:

```bash
$SPACE_PROJECT_ROOT/result/SPACE_merged
```

Important output files:

```text
merged_results.csv
config_summary.json
```

## Perception Encoder

Clone the Perception Models repo under the model root:

```bash
cd "$SPACE_MODEL_ROOT"
git clone https://github.com/facebookresearch/perception_models.git
```

The full `pip install -e .` may require Python 3.11+. SPACE only needs the PE source code, so `PERCEPTION_MODELS_PATH` is enough:

```bash
export PERCEPTION_MODELS_PATH="$SPACE_MODEL_ROOT/perception_models"
```

Use a PE architecture such as:

```bash
PE:PE-Core-L14-336
```

For PE linear probing, freeze the backbone and train only the classifier head:

```bash
python train.py \
  --arch "PE:PE-Core-L14-336" \
  --fix_backbone \
  --cropSize 336 \
  --is_resize \
  ...
```

For merged evaluation with PE replacing CLIP, set the third branch in `scripts/validate_merged.sh` to the PE architecture and checkpoint.

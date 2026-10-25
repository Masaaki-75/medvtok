# MedVTok Script Setup Guide

This guide explains what to prepare before running the scripts in `scripts/`.

## 1. Prepare the Python environment

Use Linux x86_64 with Conda installed and an NVIDIA GPU for feature extraction and training. Run [build_env.sh](build_env.sh) to configure the Python environment:

```bash
cd /path/to/this/repo
bash build_env.sh
conda activate medvtok
```

The script creates or reuses a Python 3.11 environment, installs PyTorch 2.7.1 with CUDA 12.6 and all project and encoder dependencies from [requirements.txt](requirements.txt). It downloads and installs FlashAttention 2.7.4, caches the verified wheel in the repository root, and checks the installed versions and GPU visibility. Use `ENV_NAME=my_env bash build_env.sh` to choose another environment name.

## 2. Download the encoder weights

The commands below use `/path/to/medvtok_weights` as an example. Replace it with your own absolute directory.

```bash
export WEIGHT_ROOT=/path/to/medvtok_weights
mkdir -p "$WEIGHT_ROOT"
hf auth login
```

### DermLIP

1. Request and receive access at [DermLIP PanDerm](https://huggingface.co/redlessone/DermLIP_PanDerm-base-w-PubMed-256).
2. Download the repository:

```bash
hf download redlessone/DermLIP_PanDerm-base-w-PubMed-256 \
  --local-dir "$WEIGHT_ROOT/DermLIP/DermLIP_PanDerm-base-w-PubMed-256"
```

The directory must contain `open_clip_config.json` and either `open_clip_model.safetensors` or `open_clip_pytorch_model.bin`.

### HLIP

Download [HLIP](https://huggingface.co/zch0414/clip-vit_large-slice_scan_study-dualdinotxt1568):

```bash
hf download zch0414/clip-vit_large-slice_scan_study-dualdinotxt1568 \
  --local-dir "$WEIGHT_ROOT/HLIP/clip-vit_large-slice_scan_study-dualdinotxt1568"
```

The required file is `hlip_vision_weights.pth`.

### MedImageInsight

Download the full [MedImageInsights repository](https://huggingface.co/lion-ai/MedImageInsights/tree/main)
and keep its directory structure:

```bash
hf download lion-ai/MedImageInsights --local-dir "$WEIGHT_ROOT/medimageinsight"
```

The code uses these four items:

```text
medimageinsight/2024.09.27/config.yaml
medimageinsight/2024.09.27/vision_model/medimageinsigt-v1.0.0.pt
medimageinsight/2024.09.27/language_model/clip_tokenizer_4.16.2/
medimageinsight/pytorch_weights/image_encoder.pt
```

### SPECTRE

Download [SPECTRE](https://huggingface.co/cclaess/SPECTRE/tree/main):

```bash
hf download cclaess/SPECTRE --local-dir "$WEIGHT_ROOT/SPECTRE"
```

The code uses these three files:

```text
spectre_backbone_vit_large_patch16_128.pt
spectre_combiner_feature_vit_large.pt
SigLIP_projection_head_image.pt
```

### PeskaVLP

```bash
mkdir -p "$WEIGHT_ROOT/PeskaVLP"
wget -O "$WEIGHT_ROOT/PeskaVLP/PeskaVLP.pth" "https://seafile.unistra.fr/f/65a2b1bf113e428280d0/?dl=1"
```

### CONCH v1.5

1. Request and receive access at [CONCH v1.5](https://huggingface.co/MahmoodLab/conchv1_5).
2. Download it:

```bash
hf download MahmoodLab/conchv1_5 --local-dir "$WEIGHT_ROOT/CONCH/conch_v15"
```

The required file is `CONCH/conch_v15/pytorch_model.bin`.

### RET-CLIP

Follow the checkpoint instructions in the [RET-CLIP repository](https://github.com/sStonemason/RET-CLIP), then place the downloaded checkpoint at `"$WEIGHT_ROOT/retclip/ret-clip.pt"`


## 3. Update the encoder paths

Open `local_constants.py` and point the following values to the files you just
downloaded. Always use absolute paths.

```python
DERMLIP_CKPT = '/path/to/medvtok_weights/DermLIP'
HLIPV_CKPT = '/path/to/medvtok_weights/HLIP/clip-vit_large-slice_scan_study-dualdinotxt1568/hlip_vision_weights.pth'

MII_CONFIG = '/path/to/medvtok_weights/medimageinsight/2024.09.27/config.yaml'
MII_UNICL_CKPT = '/path/to/medvtok_weights/medimageinsight/2024.09.27/vision_model/medimageinsigt-v1.0.0.pt'
MII_TEXT_TOKENIZER = '/path/to/medvtok_weights/medimageinsight/2024.09.27/language_model/clip_tokenizer_4.16.2'
MII_VISION_CKPT = '/path/to/medvtok_weights/medimageinsight/pytorch_weights/image_encoder.pt'

SPECTRE_BACKBONE_CKPT = '/path/to/medvtok_weights/SPECTRE/spectre_backbone_vit_large_patch16_128.pt'
SPECTRE_COMBINER_CKPT = '/path/to/medvtok_weights/SPECTRE/spectre_combiner_feature_vit_large.pt'
SPECTRE_PROJECTION_CKPT = '/path/to/medvtok_weights/SPECTRE/SigLIP_projection_head_image.pt'

PESKAVLP_CKPT = '/path/to/medvtok_weights/PeskaVLP/PeskaVLP.pth'
CONCH_CKPT = '/path/to/medvtok_weights/CONCH'
RETCLIP_CKPT = '/path/to/medvtok_weights/retclip/ret-clip.pt'
```

## 4. Download the two training weights in `external/`

Both files come from
[FoundationVision/unitok_external](https://huggingface.co/FoundationVision/unitok_external/tree/main):

```bash
hf download FoundationVision/unitok_external \
  dinov2_vits14_pretrain.pth lpips_with_vgg.pth \
  --local-dir external
```

The final locations must be:

```text
external/dinov2_vits14_pretrain.pth
external/lpips_with_vgg.pth
```

No path change is needed when the files are placed there. Their defaults are
defined in `utilities/config.py`.

## 5. Prepare the data and CSV files

Each CSV must contain these columns:

```text
identifier,dataset_identifier,modality,submodality,multiwindow
```

`identifier` is the image or volume path relative to `TRAIN_ROOT`. The small example files are `data/examples_2d.csv` and `data/examples_3d.csv`, with sample inputs under `data/vis_imgs/`. The default `TRAIN_ROOT` is `data/`, so the bundled CSV identifiers use `vis_imgs/...`.

To split all source CSVs into single-modality extraction tables and then run the
matching 2D/3D extractor automatically:

```bash
# Preview: write deduplicated tables to data/single_modality/ and print commands.
python scripts/expert_feature_extraction/extract_all_features.py --dry-run

# Full workflow for the bundled examples; already extracted features are skipped.
python scripts/expert_feature_extraction/extract_all_features.py

# Use a different source table and image root.
python scripts/expert_feature_extraction/extract_all_features.py \
  --csv 'data/*.csv' --data-root /path/to/medical_images \
  --metadata-dir data/single_modality --feature-root outputs/alignment
```

The source training CSVs are preserved. The workflow canonicalizes submodality
aliases and deduplicates identifiers across input tables; `ct3d`/`mr3d` use the
3D extractor, while `ct3d2d`/`mr3d2d` use the 2D extractor. The default image
root is `data/`, matching the bundled examples. `--modalities ct3d mr3d`
selects a subset, `--nproc-per-node 4` uses four GPUs, and `--force` regenerates
existing features. Add `--device cpu --num-workers 0 --num-workers-3d 0 --max-samples 1`
for a small CPU run. Default paths are resolved from the script's location;
explicit relative paths are resolved from the caller's working directory.

Before training, extract one alignment feature for every input. Features are written under `outputs/alignment/default/` and keep the same relative path as the input, with the suffix changed to `.pt`.

Use a metadata CSV containing only the modality named in each command. For example:

```bash
python scripts/expert_feature_extraction/extract_alignment_features.py \
  --modality dermoscopy \
  --meta_path /path/to/dermoscopy.csv \
  --data_root /path/to/data
```

The 2D modality-to-encoder mapping is:

| `--modality` | Encoder |
|---|---|
| `dermoscopy` | DermLIP |
| `cfp` | RET-CLIP |
| `histopathology` | CONCH |
| `endoscopy`, `ophthalmoscope`, `ophthalmoscopy`, `surgical` | PeskaVLP |
| `ct2d`, `mr2d`, `ultrasound`, `xray`, `mammography`, or others | MedImageInsight |

For 3D data:

```bash
# 3D CT uses SPECTRE.
python scripts/expert_feature_extraction/extract_alignment_features_3d.py \
  --modality ct3d \
  --meta_path /path/to/ct3d.csv \
  --data_root /path/to/data

# 3D MRI uses HLIP.
python scripts/expert_feature_extraction/extract_alignment_features_3d.py \
  --modality mr3d \
  --meta_path /path/to/mr3d.csv \
  --data_root /path/to/data
```

Use `torchrun --nproc-per-node=N` instead of `python` to extract features with multiple GPUs.

## 6. Prepare the MedVTok checkpoint

Download the released checkpoint and configuration from [massaki75/medvtok](https://huggingface.co/massaki75/medvtok):

```bash
hf download massaki75/medvtok config.json ckpt.pth --local-dir weights
```

Stage 1 can start without an initialization checkpoint. To continue an older Stage 1 run, set `RESUME_FROM`.

For optional SigLIP2 encoder initialization, run the [conversion workflow](scripts/adapt_siglip2_to_medvtok/README.md) beforehand, then pass the resulting file:

```bash
SIGLIP_ENCODER_INIT="$PWD/scripts/adapt_siglip2_to_medvtok/converted/encoder.safetensors" \
NPROC_PER_NODE=2 WORKERS=0 MIXED_RES=False ENABLE_VISUALIZER=False \
bash scripts/training/train_stage1_full_mnodes.sh 1 2
```

`SIGLIP_ENCODER_INIT` defaults to empty and only applies to fresh Stage 1 runs. Training loads the encoder tensors strictly without running conversion or reading the original SigLIP2 weights or conversion manifest. It retains MedVTok's RoPE, LayerNorm epsilon, preprocessing, and other modules; this is parameter initialization, not SigLIP2 forward-equivalence mode. Explicit and automatic checkpoint recovery skip this initialization, even if the converted file has been moved or deleted. Use the default four-frame `first` conversion with `PADDING_MODE=zero`; the `average` conversion needs `PADDING_MODE=replicate`.

Stage 2 needs an initial MedVTok checkpoint. Put it at `weights/ckpt.pth`, or set `INIT_CKPT` to another absolute path. This checkpoint is separate from the encoder weights listed above.

## 7. Run training

Run commands from the repository root. Once alignment features are available, training reads the `.pt` files directly; expert checkpoints and `local_constants.py` paths are only needed to extract those features.

Run Stage 1 with 2D data only:

```bash
TRAIN_DATA=/path/to/train_2d.csv \
TRAIN_ROOT=/path/to/data \
FEATURE_ROOT="$PWD/outputs/alignment" \
NPROC_PER_NODE=1 \
bash scripts/training/train_stage1_full_mnodes.sh 1 10
```

Run Stage 2 with both 2D and 3D data:

```bash
TRAIN_DATA=/path/to/train_2d.csv \
TRAIN_DATA_3D=/path/to/train_3d.csv \
TRAIN_ROOT=/path/to/data \
FEATURE_ROOT="$PWD/outputs/alignment" \
INIT_CKPT="$PWD/weights/ckpt.pth" \
NPROC_PER_NODE=1 \
bash scripts/training/train_stage2_full_mnodes.sh 1 10 2
```

The Stage 1 arguments are `number_of_nodes` and `2D_batch_size`. The Stage 2 arguments are `number_of_nodes`, `2D_batch_size`, and `3D_batch_size`.

Training checkpoints are saved under `outputs/ckpts/` unless `OUTPUT_DIR` is set to another location.

For a smaller single-GPU Stage 1 run using the bundled CSV and existing alignment features:

```bash
NPROC_PER_NODE=1 WORKERS=0 MIXED_RES=False ENABLE_VISUALIZER=False \
bash scripts/training/train_stage1_full_mnodes.sh 1 2
```

`MIXED_RES=True` (the default) samples 256/384/512-pixel resolutions; Stage 2 also samples 16/32-slice depths. The example batch sizes can exceed 48 GiB per GPU. With `MIXED_RES=False`, 2D images use `IMG_SIZE` (256 by default), while 3D volumes retain their stored dimensions. For Stage 2 on limited GPU memory, resize or crop volumes before training. Use an even 3D batch size of at least 2 to include both CT and MRI in each batch.

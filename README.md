# humanoid-spatial-perception

Semantic 3D reconstruction orchestration wrapper around:
- `external/MASt3R-SLAM`
- `external/segmast3r`
- `external/Grounded-SAM-2`

## Installation

### 1. Clone submodules

```bash
git submodule update --init --recursive
```

### 2. Create conda environment

```bash
conda create -n hsp python=3.12 cmake=3.14.0
conda activate hsp

pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
```

### 3. Install MASt3R-SLAM

See [MASt3R-SLAM](https://github.com/JinRhee/MASt3R-SLAM) for full instructions. Summary:

```bash
cd external/MASt3R-SLAM
pip install --no-build-isolation -e thirdparty/mast3r "numpy==1.26.4"  # pin numpy to avoid recompilation
pip install --no-build-isolation -e thirdparty/in3d  # may need: pip install "cython<3.0"
pip install --no-build-isolation -e .
cd ../..
```

Download MASt3R checkpoints:

```bash
mkdir -p external/MASt3R-SLAM/checkpoints
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth \
  -O external/MASt3R-SLAM/checkpoints/mast3r_v1.pth
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth \
  -O external/MASt3R-SLAM/checkpoints/mast3r_v2.pth
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl \
  -O external/MASt3R-SLAM/checkpoints/mast3r_codebook.pkl
```

### 4. Install SegMASt3R

See [SegMASt3R](https://github.com/SegMASt3R/segmast3r) for full instructions. Summary:

```bash
cd external/segmast3r
pip install -e .
cd ../..
```

Download SegMASt3R checkpoint:

```bash
mkdir -p external/segmast3r/checkpoints
wget https://huggingface.co/rjayanti/segmast3r/resolve/main/segmast3r_spp.ckpt \
  -O external/segmast3r/checkpoints/segmast3r_spp.ckpt
```

### 5. Install Grounded-SAM-2

See [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2) for full instructions. Summary:

```bash
cd external/Grounded-SAM-2
pip install -e .
pip install --no-build-isolation -e grounding_dino
cd ../..
```

Download checkpoints:

```bash
cd external/Grounded-SAM-2/checkpoints && bash download_ckpts.sh && cd ../../..
cd external/Grounded-SAM-2/gdino_checkpoints && bash download_ckpts.sh && cd ../../..
```

### 6. Install this package

```bash
pip install -e .
```

## Datasets

Download benchmark datasets using the MASt3R-SLAM scripts (run from the repo root):

```bash
# TUM RGB-D
bash external/MASt3R-SLAM/scripts/download_tum.sh

# 7-Scenes
bash external/MASt3R-SLAM/scripts/download_7_scenes.sh

# ETH3D
bash external/MASt3R-SLAM/scripts/download_eth3d.sh

# EuRoC MAV
bash external/MASt3R-SLAM/scripts/download_euroc.sh
```

Each script creates and populates a `datasets/<name>/` directory at the repo root.

## Convert video to pipeline-ready images

```bash
./video_to_images \
  --input_video /absolute/path/to/video.mp4 \
  --output_dir /absolute/path/to/images \
  --start_sec 0 \
  --start_nsec 0
```

This writes JPEG frames named `images_<sec>_<nsec>.jpg`, matching the ingestion format used by the pipeline.
It requires `ffmpeg` and `ffprobe` to be available on `PATH`.

## Run

```bash
./run_pipeline \
  --images_dir /absolute/path/to/images \
  --config /absolute/path/to/configs/pipeline.yaml \
  --output_dir /absolute/path/to/output \
  --preset balanced
```

## Adapter configuration

The pipeline expects adapter factories for MASt3R backbone + SLAM, Grounded-SAM-2, and SegMASt3R. These are configured in
`configs/pipeline.yaml` under `backbone.factory`, `slam.factory`, `grounded_sam2.factory`, and `segmast3r.factory`.
The default config leaves the SLAM/Grounded-SAM-2/SegMASt3R factories unset, so fill them in when using those modes.

`backbone.factory` is set to `hsp_pipeline.backbone:create_mast3r_backbone`.
`backbone.model_factory` should point to the MASt3R model loader (`module:function`). The built-in
`create_mast3r_backbone` adapter uses unified inference:
- encode once per frame (cached on frame object),
- symmetric decode (i→j and j→i),
- branch outputs into SLAM tensors `(X, C, D, Q)` and SegMASt3R descriptors `(desc_i, desc_j)`.

Set `slam.factory`, `grounded_sam2.factory`, and `segmast3r.factory` to `module:function` callables that return adapters with:
- MASt3R backbone: `run_pair(image_a, image_b)` and `run_single(image)` returning pointmaps + feature grids.
- MASt3R-SLAM: `run(frames, backbone)` returning a trajectory and optional global map.
- Grounded-SAM-2: `predict(image, prompts, max_masks)` returning masks + scores.
- SegMASt3R: `encode_mask(features, mask)` returning a descriptor.

For smoke testing without external dependencies, set `backbone.mode: stub`, `slam.mode: stub`, `grounded_sam2.mode: stub`,
and `segmast3r.mode: stub` in the config.

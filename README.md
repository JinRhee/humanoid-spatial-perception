# humanoid-spatial-perception

Semantic 3D reconstruction orchestration wrapper around:
- `external/MASt3R-SLAM`
- `external/segmast3r`
- `external/sam3`

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

## Installation
```bash
git submodule update --init --recursive

conda create -n hsp python=3.12 cmake=3.14.0
conda activate hsp

pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
```

### Install MASt3R-SLAM
```bash
cd external/MASt3R-SLAM
pip install --no-build-isolation -e thirdparty/mast3r "numpy==1.26.4" # Without numpy version pin, latest numpy will be used causing a recompilation of MASt3R-SLAM
pip install --no-build-isolation -e thirdparty/in3d    # Might need to downgrade cython via $ pip install "cython<3.0"
pip install --no-build-isolation -e .
cd ..
```

Download checkpoints for MASt3R
```bash
mkdir -p checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth -O checkpoints/mast3r_v1.pth
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth -O checkpoints/mast3r_v2.pth
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl -O checkpoints/mast3r_codebook.pkl
```

### Install SegMASt3R
```bash
mkdir -p checkpoints
wget https://huggingface.co/rjayanti/segmast3r/resolve/main/segmast3r_spp.ckpt -O checkpoints/segmast3r_spp.ckpt
```

### Install SAM3
```bash
cd sam3
pip install -e .
cd ../..
```
Optional dependencies for faster inference
```bash
pip install einops ninja && pip install flash-attn-3 --no-deps --index-url https://download.pytorch.org/whl/cu128
pip install git+https://github.com/ronghanghu/cc_torch.git
```

## Adapter configuration

The pipeline expects adapter factories for MASt3R backbone + SLAM, SAM3, and SegMASt3R. These are configured in
`configs/pipeline.yaml` under `backbone.factory`, `slam.factory`, `sam3.factory`, and `segmast3r.factory`.
The default config leaves the SLAM/SAM3/SegMASt3R factories unset, so fill them in when using those modes.

`backbone.factory` defaults to `hsp_pipeline.backbone:create_mast3r_backbone`.
`backbone.model_factory` should point to the MASt3R model loader (`module:function`). The built-in
`create_mast3r_backbone` adapter uses unified inference:
- encode once per frame (cached on frame object),
- symmetric decode (i→j and j→i),
- branch outputs into SLAM tensors `(X, C, D, Q)` and SegMASt3R descriptors `(desc_i, desc_j)`.

Set `slam.factory`, `sam3.factory`, and `segmast3r.factory` to `module:function` callables that return adapters with:
- MASt3R backbone: `run_pair(image_a, image_b)` and `run_single(image)` returning pointmaps + feature grids.
- MASt3R-SLAM: `run(frames, backbone)` returning a trajectory and optional global map.
- SAM3: `predict(image, prompts, max_masks)` returning masks + scores.
- SegMASt3R: `encode_mask(features, mask)` returning a descriptor.

For smoke testing without external dependencies, set `backbone.mode: stub`, `slam.mode: stub`, `sam3.mode: stub`,
and `segmast3r.mode: stub` in the config.

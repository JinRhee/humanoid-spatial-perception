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

```bash
cd external/MASt3R-SLAM
pip install --no-build-isolation -e thirdparty/mast3r
pip install --no-build-isolation -e thirdparty/in3d    # Might need to downgrade cython via $ pip install "cython<3.0"
pip install --no-build-isolation -e .
cd ..

cd SegMASt3R
pip install -r mast3r_src/dust3r/requirements.txt
pip install -e .
cd ..


cd sam3
pip install -e .
cd ../..


```

```bash
mkdir -p checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth -P checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth -P checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl -P checkpoints/

wget https://huggingface.co/rjayanti/segmast3r/resolve/main/segmast3r_spp.ckpt -O checkpoints/segmast3r_spp.ckpt


```
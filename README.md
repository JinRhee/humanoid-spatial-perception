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

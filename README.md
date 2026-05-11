# humanoid-spatial-perception

Semantic 3D reconstruction orchestration wrapper around:
- `external/MASt3R-SLAM`
- `external/segmast3r`
- `external/sam3`

## Run

```bash
/home/runner/work/humanoid-spatial-perception/humanoid-spatial-perception/run_pipeline \
  --images_dir /home/runner/work/humanoid-spatial-perception/humanoid-spatial-perception/images \
  --config /home/runner/work/humanoid-spatial-perception/humanoid-spatial-perception/configs/pipeline.yaml \
  --output_dir /home/runner/work/humanoid-spatial-perception/humanoid-spatial-perception/outputs \
  --preset balanced
```

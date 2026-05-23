import time
from pathlib import Path

import numpy as np

try:
    import torch as _torch
except ImportError:
    _torch = None


def measure_time(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {time.time() - start:.3f}s")
        return result

    return wrapper


class _MasksWrapper:
    def __init__(self, data):
        self.data = data


class SegmentResult:
    """Unified result type returned by all segmentor backends."""

    def __init__(self, masks_tensor, labels=None):
        self.masks = _MasksWrapper(masks_tensor)
        self.labels = labels  # list[str | None] | None


class GroundedSAM2Segmentor:
    """Grounding DINO → SAM 2: detect keyword objects, then segment them."""

    def __init__(
        self,
        sam2_checkpoint: str,
        sam2_config: str,
        gdino_checkpoint: str,
        gdino_config: str,
        keywords: list,
        seg_config=None,
    ):
        import sys as _sys
        _gsam2_root = str(Path(__file__).parents[2] / "external/Grounded-SAM-2")
        if _gsam2_root not in _sys.path:
            _sys.path.insert(0, _gsam2_root)

        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from groundingdino.util.inference import load_model
        import groundingdino.datasets.transforms as _GDT

        print("Loading SAM 2...")
        sam2 = build_sam2(sam2_config, sam2_checkpoint, device="cuda:0")
        self.predictor = SAM2ImagePredictor(sam2)

        print("Loading Grounding DINO...")
        self.gdino = load_model(gdino_config, gdino_checkpoint)

        self.text_prompt = " . ".join(keywords)
        self.keywords = [k.lower() for k in keywords]

        cfg = seg_config or {}
        self.box_threshold  = cfg.get("box_threshold",  0.35)
        self.text_threshold = cfg.get("text_threshold", 0.25)

        self._transform = _GDT.Compose([
            _GDT.RandomResize([800], max_size=1333),
            _GDT.ToTensor(),
            _GDT.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _to_device(self, device: str):
        self.gdino.to(device)
        self.predictor.model.to(device)
        self._device = device

    def release(self):
        """Move models to CPU and free GPU cache so MASt3R-SLAM can use the VRAM."""
        import torch as _t
        self._to_device("cpu")
        _t.cuda.empty_cache()

    def _preprocess(self, image: np.ndarray):
        """Convert uint8 RGB numpy array to the normalised tensor GDINO expects."""
        from PIL import Image as _PIL
        pil = _PIL.fromarray(image)
        tensor, _ = self._transform(pil, None)
        return tensor

    @measure_time
    def segment(self, image: np.ndarray) -> SegmentResult:
        """image: uint8 (H, W, 3) RGB."""
        import torch
        from groundingdino.util.inference import predict
        from torchvision.ops import box_convert

        self._to_device("cuda:0")
        image_tensor = self._preprocess(image)
        boxes, _, phrases = predict(
            self.gdino, image_tensor, self.text_prompt,
            self.box_threshold, self.text_threshold,
            device="cuda:0",
        )

        if boxes is None or len(boxes) == 0:
            empty = torch.zeros(0, image.shape[0], image.shape[1])
            print("Generated 0 instance masks")
            return SegmentResult(empty, labels=[])

        H, W = image.shape[:2]
        boxes_xyxy = box_convert(
            boxes=boxes * torch.tensor([W, H, W, H]),
            in_fmt="cxcywh", out_fmt="xyxy",
        ).numpy()

        self.predictor.set_image(image)
        masks, _, _ = self.predictor.predict(
            box=boxes_xyxy, multimask_output=False,
        )  # (N, 1, H, W)

        labels = [self._match(p) for p in phrases]
        print(f"Generated {len(masks)} instance masks")
        return SegmentResult(_torch.from_numpy(masks[:, 0]).float(), labels=labels)

    def _match(self, phrase: str):
        phrase = phrase.lower()
        for kw in self.keywords:
            if kw in phrase or phrase in kw:
                return kw
        return None


def create_segmentor(seg_config: dict, keywords: list | None = None):
    """Factory: returns the backend named in seg_config['backend']."""
    backend = seg_config.get("backend", "grounded_sam2")
    if backend != "grounded_sam2":
        raise ValueError(f"Unknown segmentation backend: {backend!r}")
    return GroundedSAM2Segmentor(
        sam2_checkpoint  = seg_config["sam2_checkpoint"],
        sam2_config      = seg_config["sam2_config"],
        gdino_checkpoint = seg_config["gdino_checkpoint"],
        gdino_config     = seg_config["gdino_config"],
        keywords         = keywords or [],
        seg_config       = seg_config,
    )
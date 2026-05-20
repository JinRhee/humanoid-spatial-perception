"""
Lightweight Entity Segmentation Pipeline
Image -> FastSAM Segment Everything -> Visualization
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import FastSAM, SAM
import time
from pathlib import Path

try:
    import torch as _torch
except ImportError:
    _torch = None


def measure_time(func):
    """Decorator to measure function execution time"""

    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {time.time() - start:.3f}s")
        return result

    return wrapper


class SegmentationPipeline:
    def __init__(
        self,
        fastsam_model_path="FastSAM-x.pt",
        seg_config=None,
    ):
        """
        Initialize the segmentation pipeline with FastSAM

        Args:
            fastsam_model_path: Path to FastSAM model
        """
        print("Loading FastSAM model...")
        self.model = measure_time(FastSAM)(fastsam_model_path)

        self.last_segments = None

        # Set defaults, optionally override from seg_config
        self.conf = 0.8
        self.iou = 0.9
        self.imgsz = 1024
        
        if seg_config is not None:
            self.conf = seg_config.get("conf", self.conf)
            self.iou = seg_config.get("iou", self.iou)
            self.imgsz = seg_config.get("imgsz", self.imgsz)

    @measure_time
    def segment(self, image, conf=None, iou=None, imgsz=None):
        """
        Run FastSAM segmentation (segment everything)

        Args:
            image: Input image (file path, numpy array, or PIL Image)
            conf: Confidence threshold (default 0.4)
            iou: IoU threshold for NMS (default 0.9)
            imgsz: Input image size (default 1024)

        Returns:
            Segmentation results
        """
        # Run FastSAM with segment everything mode
        # The Ultralytics model handles paths, numpy arrays, and PIL images natively
        results = self.model(
            image,
            device="cuda:0",
            retina_masks=True,
            imgsz=imgsz if imgsz is not None else self.imgsz,
            conf=conf if conf is not None else self.conf,
            iou=iou if iou is not None else self.iou,
            verbose=False,
        )

        self.last_segments = results[0]
        print(
            f"Generated {len(results[0].masks) if results[0].masks is not None else 0} instance masks"
        )

        return results[0]

    def visualize(self, image_path, segments=None, save_path=None, show=True):
        """
        Visualize segmentation masks

        Args:
            image_path: Path to input image
            segments: Segmentation results (uses last_segments if None)
            save_path: Path to save visualization (optional)
            show: Whether to display the plot
        """
        if segments is None:
            segments = self.last_segments

        # Load image
        img = cv2.imread(str(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Original image
        axes[0].imshow(img)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        # Segmentation masks
        if segments is not None and segments.masks is not None:
            img_seg = img.copy()
            masks = segments.masks.data.cpu().numpy()

            # Create colored overlay
            overlay = np.zeros_like(img)
            for i, mask in enumerate(masks):
                # Resize mask to image size if needed
                if mask.shape != img.shape[:2]:
                    mask = cv2.resize(
                        mask.astype(np.uint8), (img.shape[1], img.shape[0])
                    )

                # Generate random color for each instance
                color = np.random.randint(0, 255, 3).tolist()
                overlay[mask > 0.5] = color

            # Blend with original image
            img_seg = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)

            axes[1].imshow(img_seg)
            axes[1].set_title(f"Instance Masks ({len(masks)})")
        else:
            axes[1].imshow(img)
            axes[1].set_title("Instance Masks (0)")
        axes[1].axis("off")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"\nSaved visualization to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def run(self, image_path, conf=0.4, iou=0.9, imgsz=1024, save_path=None, show=True):
        """
        Complete pipeline: segment -> visualize

        Args:
            image_path: Path to input image
            conf: Confidence threshold
            iou: IoU threshold for NMS
            imgsz: Input image size
            save_path: Path to save visualization (optional)
            show: Whether to display the plot

        Returns:
            segments
        """
        segments = self.segment(image_path, conf=conf, iou=iou, imgsz=imgsz)
        self.visualize(image_path, segments, save_path, show)

        return segments


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
    backend = seg_config.get("backend", "fastsam")
    if backend == "grounded_sam2":
        return GroundedSAM2Segmentor(
            sam2_checkpoint  = seg_config["sam2_checkpoint"],
            sam2_config      = seg_config["sam2_config"],
            gdino_checkpoint = seg_config["gdino_checkpoint"],
            gdino_config     = seg_config["gdino_config"],
            keywords         = keywords or [],
            seg_config       = seg_config,
        )
    # default: FastSAM
    return SegmentationPipeline(
        fastsam_model_path=seg_config.get("fastsam_checkpoint", "FastSAM-x.pt"),
        seg_config=seg_config,
    )


class CLIPLabeler:
    """Assigns semantic labels to segments via CLIP text–image cosine similarity."""

    def __init__(self, keywords_path: str, clip_model, tokenizer, device):
        with open(keywords_path) as f:
            self.labels = [line.strip() for line in f if line.strip()]
        tokens = tokenizer(self.labels).to(device)
        with _torch.no_grad():
            text_feats = clip_model.encode_text(tokens)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        self._text_feats = text_feats.float().cpu().numpy()  # (K, 512)

    def assign(self, clip_features: np.ndarray, threshold: float = 0.20) -> list:
        """Return list[str | None] of length M. None when best sim < threshold."""
        if clip_features is None or len(clip_features) == 0:
            return []
        sims = clip_features @ self._text_feats.T          # (M, K)
        best_idx = sims.argmax(axis=1)                     # (M,)
        best_score = sims[np.arange(len(sims)), best_idx]
        return [
            self.labels[int(i)] if float(s) >= threshold else None
            for i, s in zip(best_idx, best_score)
        ]


# Example usage
if __name__ == "__main__":
    CHECKPOINT_ROOT_PATH = Path("./checkpoints/ultralytics")
    fastsam_checkpoint = CHECKPOINT_ROOT_PATH / "mobile_sam.pt"

    # Initialize pipeline
    pipeline = SegmentationPipeline(
        fastsam_model_path=fastsam_checkpoint,
    )

    # Run on single image
    image_path = "./checkpoints/test_image.jpg"
    segments = pipeline.run(
        image_path,
        conf=0.6,
        iou=0.45,
        imgsz=768,
        save_path="./checkpoints/output_visualization.jpg",
        show=False,
    )

    # Or run steps individually for more control
    # segments = pipeline.segment(image_path, conf=0.4, iou=0.9, imgsz=1024)
    # pipeline.visualize(image_path, save_path='custom_output.jpg')
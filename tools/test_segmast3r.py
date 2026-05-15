import torch
import numpy as np
from PIL import Image
from pathlib import Path
import types

# ── Load original SegMASt3R (vanilla) ──────────────────────────────
import sys
segmast3r_root = Path("external/segmast3r").resolve()
vendored_src = segmast3r_root / "src"
original_src_module = sys.modules.get("src")
vendored_src_module = types.ModuleType("src")
vendored_src_module.__path__ = [str(vendored_src)]
sys.modules["src"] = vendored_src_module
sys.path.insert(0, str(segmast3r_root))
from configs.default import cfg
from model_infer import MASt3RSegFeatInfer

if original_src_module is not None:
    sys.modules["src"] = original_src_module
else:
    sys.modules.pop("src", None)

cfg.merge_from_file("external/segmast3r/configs/config_eval_spp_resz.yaml")
vanilla = MASt3RSegFeatInfer(cfg)
vanilla.load_state_dict(
    torch.load("external/segmast3r/checkpoints/segmast3r_spp.ckpt",
               map_location="cpu", weights_only=False)["state_dict"],
    strict=False,
)
vanilla.eval().cuda()

# ── Load your UnifiedMASt3RInfer ───────────────────────────────────
sys.path.insert(0, "src")
from hsp_pipeline.unified_inference import UnifiedMASt3RInfer

unified = UnifiedMASt3RInfer(
    mast3r_ckpt="external/MASt3R-SLAM/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth",
    segmast3r_ckpt="external/segmast3r/checkpoints/heads_only.pt",
)
unified.prepare("cuda")

print()
# Check whether seg_head1 weights match vanilla's downstream_head1
vanilla_w = dict(vanilla.encoder.downstream_head1.named_parameters())
unified_w = dict(unified.seg_head1.named_parameters())

for k in list(vanilla_w.keys()):
    match = torch.allclose(vanilla_w[k], unified_w[k], atol=1e-6)
    if not match:
        print(f"{k}: match={match}, vanilla={vanilla_w[k].mean().item():.6f}, unified={unified_w[k].mean().item():.6f}")
print()


# ── Shared dummy input ─────────────────────────────────────────────
torch.manual_seed(0)
H, W = 512, 512
img = torch.rand(1, 3, H, W).cuda()
true_shape = torch.tensor([[H, W]], dtype=torch.int64).cuda()

view = {"img": img, "true_shape": true_shape, "instance": ["test"]}

# ── Vanilla forward ────────────────────────────────────────────────
with torch.no_grad():
    desc_v1, desc_v2 = vanilla.forward(view, view)
    # desc shape: (1, H, W, 24)
    print("vanilla desc mean:", desc_v1.mean().item())
    print("vanilla desc norm:", desc_v1.norm(dim=-1).mean().item())

# ── Unified forward ────────────────────────────────────────────────
from types import SimpleNamespace

frame = SimpleNamespace(
    img=img,
    img_true_shape=true_shape,
    feat=None,
    pos=None,
)

with torch.no_grad():
    # Run encoder + decoder manually to get dec tokens
    unified._encode_frame(frame)
    dec11, dec21 = unified._decoder(frame.feat, frame.pos, frame.feat, frame.pos)
    res_unified, _ = unified._segmast3r_head(dec11, dec21, true_shape, true_shape)
    desc_u = res_unified["desc"]
    print("unified desc mean:", desc_u.mean().item())
    print("unified desc norm:", desc_u.norm(dim=-1).mean().item())

with torch.no_grad():
    # What does _downstream_head return directly?
    (shape1, shape2), (feat1, feat2), (pos1, pos2) = vanilla.encoder._encode_symmetrized(view, view)
    dec1, dec2 = vanilla.encoder._decoder(feat1, pos1, feat2, pos2)
    res = vanilla.encoder._downstream_head(1, [tok.float() for tok in dec1], shape1)
    print("_downstream_head desc norm:", res["desc"].norm(dim=-1).mean().item())

    # What does MASt3RSegFeatInfer.forward() return?
    desc_v1, desc_v2 = vanilla.forward(view, view)
    print("vanilla.forward() desc norm:", desc_v1.norm(dim=-1).mean().item())

print("dustbin score:", unified.feature_matcher.matching_mat.dustbin_score.item())


# ── Compare ────────────────────────────────────────────────────────
print("max abs diff:", (desc_v1 - desc_u).abs().max().item())
print("descriptors match:", torch.allclose(desc_v1, desc_u, atol=1e-4))
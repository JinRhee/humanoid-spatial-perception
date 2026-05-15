# check_heads.py
import sys
import torch
from pathlib import Path
from types import SimpleNamespace

# Ensure repository root is on sys.path so imports like `src.models` resolve
repo_root = Path(__file__).resolve().parent
# Ensure external/segmast3r is first so its internal `src` package is used
sys.path.insert(0, str((repo_root / "external" / "segmast3r").resolve()))
# Then add repository root so `external.segmast3r` (and other top-level packages) are importable
sys.path.insert(1, str(repo_root))
# Add the workspace `src` directory so `hsp_pipeline` can be imported as a top-level package
sys.path.insert(2, str((repo_root / "src").resolve()))

from configs.default import cfg
cfg.merge_from_file("external/segmast3r/configs/config_eval_spp_resz.yaml")
from model_infer import MASt3RSegFeatInfer
from hsp_pipeline.unified_inference import UnifiedMASt3RInfer

vanilla = MASt3RSegFeatInfer(cfg)
vanilla.load_state_dict(
    torch.load("external/segmast3r/checkpoints/segmast3r_spp.ckpt",
               map_location="cpu", weights_only=False)["state_dict"],
    strict=False,
)

unified = UnifiedMASt3RInfer(
    mast3r_ckpt="external/MASt3R-SLAM/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth",
    segmast3r_ckpt="external/segmast3r/checkpoints/segmast3r_spp.ckpt",
)

vanilla_params = dict(vanilla.encoder.downstream_head1.named_parameters())
unified_params = dict(unified.seg_head1.named_parameters())

print(f"vanilla keys: {len(vanilla_params)}")
print(f"unified keys: {len(unified_params)}")
print(f"key sets match: {set(vanilla_params.keys()) == set(unified_params.keys())}")

for k in list(vanilla_params.keys())[:5]:
    v = vanilla_params[k]
    u = unified_params.get(k)
    if u is None:
        print(f"MISSING in unified: {k}")
    else:
        match = torch.allclose(v, u, atol=1e-6)
        print(f"{k}: match={match} vanilla_mean={v.mean().item():.6f} unified_mean={u.mean().item():.6f}")

torch.manual_seed(0)
H, W = 512, 512
img = torch.rand(1, 3, H, W)
true_shape = torch.tensor([[H, W]], dtype=torch.int64)

view = {"img": img, "true_shape": true_shape, "instance": ["test"]}

# Vanilla decoder tokens
with torch.no_grad():
    (shape1, shape2), (feat1, feat2), (pos1, pos2) = vanilla.encoder._encode_symmetrized(view, view)
    dec1_v, dec2_v = vanilla.encoder._decoder(feat1, pos1, feat2, pos2)

# Unified decoder tokens
frame = SimpleNamespace(img=img, img_true_shape=true_shape, feat=None, pos=None)
with torch.no_grad():
    unified._encode_frame(frame)
    dec1_u, dec2_u = unified._decoder(frame.feat, frame.pos, frame.feat, frame.pos)

# Compare token by token
for i, (tv, tu) in enumerate(zip(dec1_v, dec1_u)):
    match = torch.allclose(tv, tu, atol=1e-4)
    print(f"dec1 token[{i}]: match={match} max_diff={(tv - tu).abs().max().item():.6f}")

with torch.no_grad():
    # With squeeze (your current broken call)
    shape_sq = true_shape.squeeze(0)  # (2,) instead of (1, 2)
    res_sq = unified.seg_head1([tok.float() for tok in dec1_u], shape_sq)
    print("squeezed desc norm:", res_sq["desc"].norm(dim=-1).mean().item())

    # Without squeeze (correct for the head contract: pass a 1D [H, W] tensor)
    res_no_sq = unified.seg_head1([tok.float() for tok in dec1_u], true_shape[0])
    print("unsqueezed desc norm:", res_no_sq["desc"].norm(dim=-1).mean().item())

    # Vanilla reference
    res_v = vanilla.encoder.downstream_head1([tok.float() for tok in dec1_v], shape1[0])
    print("vanilla desc norm:", res_v["desc"].norm(dim=-1).mean().item())

with torch.no_grad():
    shape_sq = true_shape.squeeze(0)  # (2,) — correct input to head directly

    res_sq = unified.seg_head1([tok.float() for tok in dec1_u], shape_sq)
    print("unified desc norm:", res_sq["desc"].norm(dim=-1).mean().item())

    res_v = vanilla.encoder.downstream_head1([tok.float() for tok in dec1_v], shape1.squeeze(0))
    print("vanilla desc norm:", res_v["desc"].norm(dim=-1).mean().item())

    print("max abs diff:", (res_sq["desc"] - res_v["desc"]).abs().max().item())

    print("vanilla head type:", type(vanilla.encoder.downstream_head1))
    print("unified head type:", type(unified.seg_head1))
    print("vanilla head:", vanilla.encoder.downstream_head1)
    print("unified head:", unified.seg_head1)
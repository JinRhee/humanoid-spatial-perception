import torch

if __name__ == "__main__":
    slam_ckpt = torch.load("external/MASt3R-SLAM/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth", map_location="cpu", weights_only=False)
    seg_ckpt  = torch.load("external/segmast3r/checkpoints/segmast3r_spp.ckpt", map_location="cpu", weights_only=False)

    slam_state = slam_ckpt.get("model", slam_ckpt)
    seg_state  = seg_ckpt["state_dict"]

    # Compare a backbone layer
    key = "patch_embed.proj.weight"
    slam_key = key
    seg_key  = f"encoder.{key}"

    if slam_key in slam_state and seg_key in seg_state:
        match = torch.allclose(slam_state[slam_key], seg_state[seg_key], atol=1e-6)
        print(f"Backbone weights match: {match}")
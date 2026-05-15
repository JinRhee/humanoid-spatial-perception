import torch
from pathlib import Path

if __name__ == "__main__":
  ckpt_path = Path("external/segmast3r/checkpoints/segmast3r_spp.ckpt")

  ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
  state = ckpt["state_dict"]

  # Extract downstream heads
  head1_keys = {k: v for k, v in state.items() if "downstream_head1" in k}
  head2_keys = {k: v for k, v in state.items() if "downstream_head2" in k}
  
  # Extract feature matcher head
  matcher_keys = {k: v for k, v in state.items() if "feature_matcher" in k}

  print(f"Head1 parameters: {len(head1_keys)}")
  print(f"Head2 parameters: {len(head2_keys)}")
  print(f"Matcher head parameters: {len(matcher_keys)}")
  
  # Save just the heads to a new checkpoint
  heads_only = {**head1_keys, **head2_keys, **matcher_keys}
  torch.save({"state_dict": heads_only}, "heads_only.pt")
  
  # Or save individual heads
  torch.save({"state_dict": head1_keys}, "head1_only.pt")
  torch.save({"state_dict": head2_keys}, "head2_only.pt")

  heads = torch.load("external/segmast3r/checkpoints/heads_only.pt", map_location="cpu", weights_only=False)
  state = heads["state_dict"]
  print(list(state.keys())[:100])   # what do the keys look like?
  print(f"total keys: {len(state)}")
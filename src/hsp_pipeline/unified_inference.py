import torch
import einops
import mast3r.model as mast3r_model
from mast3r.catmlp_dpt_head import mast3r_head_factory

from mast3r_slam.config import config
import mast3r_slam.matching as matching
from external.segmast3r.src.models.mast3r_segfeat.diff_feature_matcher import featureMatcher
from external.segmast3r.src.models.mast3r_segfeat.diff_masked_pooling import masked_average_pooling


class SegMASt3RHead(torch.nn.Module):
    def __init__(self, checkpoint_path: str, mast3r_model):
        super().__init__()

        head_type   = "catmlp+dpt"
        output_mode = "pts3d+desc24"

        self.head1 = mast3r_head_factory(head_type, output_mode, mast3r_model, has_conf=True)
        self.head2 = mast3r_head_factory(head_type, output_mode, mast3r_model, has_conf=True)

        ckpt  = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = ckpt["state_dict"]

        # Strip prefix and load each head separately

        def extract(prefix):
            matched = {
                k[len(prefix):]: v
                for k, v in state.items()
                if k.startswith(prefix)
            }
            print(f"[SegMASt3RHead] extracted {len(matched)} keys for prefix '{prefix}'")
            return matched
        
        self.head1.load_state_dict(extract("encoder.downstream_head1."))
        self.head2.load_state_dict(extract("encoder.downstream_head2."))

        # Expose the trained dustbin score so UnifiedMASt3RInfer can use it
        if "feature_matcher.matching_mat.dustbin_score" in state:
            self.dustbin_score = state["feature_matcher.matching_mat.dustbin_score"].item()
        else:
            self.dustbin_score = 5.3937  # fallback to default
            
    @torch.no_grad()
    def forward(self, dec1, dec2, shape1, shape2):
        """Forward pass through both heads.
        
        Args:
            dec1, dec2: decoder outputs (list of tokens)
            shape1, shape2: image shapes
            
        Returns:
            Full head outputs (dicts with 'pts3d', 'desc', 'conf', etc.)
        """
        shape1 = shape1.squeeze(0)
        shape2 = shape2.squeeze(0)

        pred1 = self.head1(
            [tok.float() for tok in dec1], shape1
        )
        pred2 = self.head2(
            [tok.float() for tok in dec2], shape2
        )

        return pred1, pred2


class UnifiedMASt3RInfer(torch.nn.Module):
    def __init__(
        self,
        mast3r_ckpt: str,
        segmast3r_ckpt: str,
        feature_matcher_cfg: dict | None = None,
    ):
        super().__init__()

        if feature_matcher_cfg is None:
            feature_matcher_cfg = {
                "TYPE": "Sinkhorn",
                "SINKHORN": {
                    "NUM_IT": 100,
                    "DUSTBIN_SCORE_INIT": 0.50,#5.3937,
                },
            }

        # MASt3R: from_pretrained(), includes both encoder, decoder, and original head
        self.mast3r = mast3r_model.AsymmetricMASt3R.from_pretrained(mast3r_ckpt)

        self.segmast3r_head = SegMASt3RHead(
            checkpoint_path=segmast3r_ckpt,
            mast3r_model=self.mast3r,
        )

        # Use the trained dustbin score from the checkpoint
        feature_matcher_cfg["SINKHORN"]["DUSTBIN_SCORE_INIT"] = self.segmast3r_head.dustbin_score
        self.feature_matcher = featureMatcher(feature_matcher_cfg)


        self._configure_grad()

    def _configure_grad(self):
        self.mast3r.requires_grad_(False)
        self.segmast3r_head.head1.requires_grad_(False)
        self.segmast3r_head.head2.requires_grad_(False)
        self.feature_matcher.requires_grad_(False)

    def _get_model_params(self, data_source: str):
        base_params = {
            "pos_embed": "RoPE100",
            "patch_embed_cls": "ManyAR_PatchEmbed",
            "img_size": (336, 512),
            "head_type": "catmlp+dpt",
            "output_mode": "pts3d+desc24",
            "depth_mode": ("exp", -mast3r_model.inf, mast3r_model.inf),
            "conf_mode": ("exp", 1, mast3r_model.inf),
            "enc_embed_dim": 1024,
            "enc_depth": 24,
            "enc_num_heads": 16,
            "dec_embed_dim": 768,
            "dec_depth": 12,
            "dec_num_heads": 12,
            "two_confs": True,
        }

        dataset_type = data_source.lower()
        if dataset_type in ("mapfree", "hm3d"):
            base_params.update({
                "patch_embed_cls": "PatchEmbedDust3R",
                "img_size": (512, 512),
                "desc_conf_mode": ("exp", 0, mast3r_model.inf),
                "landscape_only": False,
            })

        return base_params

    def prepare(self, device):
        """Call once before inference loop."""
        self.device = device
        self.eval()
        self.to(device)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode_frame(self, frame):
        """Encode a single frame, using cached feat/pos if available."""
        if frame.feat is None:
            frame.feat, frame.pos, _ = self.mast3r._encode_image(
                frame.img, frame.img_true_shape
            )

    @torch.inference_mode()
    def _decoder(self, feat1, pos1, feat2, pos2):
        dec1, dec2 = self.mast3r._decoder(feat1, pos1, feat2, pos2)
        return dec1, dec2

    @torch.inference_mode()
    def _mast3r_head(self, dec1, dec2, shape1, shape2):
        with torch.amp.autocast(enabled=False, device_type="cuda"):
            res1 = self.mast3r._downstream_head(
                1, [tok.float() for tok in dec1], shape1
            )
            res2 = self.mast3r._downstream_head(
                2, [tok.float() for tok in dec2], shape2
            )
        return res1, res2

    @torch.inference_mode()
    def _segmast3r_head(self, dec1, dec2, shape1, shape2):
        with torch.amp.autocast(enabled=False, device_type="cuda"):
            res1, res2 = self.segmast3r_head.forward(dec1, dec2, shape1, shape2)
        return res1, res2  # Returns full head outputs (dicts with 'pts3d', 'desc', 'conf', etc.)

    def _downsample(self, X, C, D, Q):
        factor = config["dataset"]["img_downsample"]
        if factor > 1:
            X = X[..., ::factor, ::factor, :].contiguous()
            C = C[..., ::factor, ::factor].contiguous()
            D = D[..., ::factor, ::factor, :].contiguous()
            Q = Q[..., ::factor, ::factor].contiguous()
        return X, C, D, Q

    @torch.inference_mode()
    def forward(self, frame_i, frame_j):
        """
        Run the full symmetric MASt3R inference once.
        Returns the four raw result dicts for downstream branching.
        """
        self._encode_frame(frame_i)
        self._encode_frame(frame_j)

        feat_i, pos_i = frame_i.feat, frame_i.pos
        feat_j, pos_j = frame_j.feat, frame_j.pos

        # Pass 1: i as reference, j as query
        dec11, dec21 = self._decoder(feat_i, pos_i, feat_j, pos_j)
        # shape_i, shape_j
        # Pass 2: j as reference, i as query
        dec22, dec12 = self._decoder(feat_j, pos_j, feat_i, pos_i)
        # shape_j, shape_i

        return dec11, dec21, dec22, dec12

    # ------------------------------------------------------------------
    # Unified: both branches in one call
    # ------------------------------------------------------------------

    def infer_unified(self, frame_i, frame_j, masks_i, masks_j, debug: bool = False):
        """
        Run both branches from a single MASt3R inference pass.

        Returns:
            slam: (X, C, D, Q) — for matching.match()
            seg: match_result (B, M) — segment correspondences
            agg_desc_i: (B, D, M) pooled per-mask descriptors for frame_i
            agg_desc_j: (B, D, N) pooled per-mask descriptors for frame_j
        """
        assert hasattr(self, "device"), "Call model.prepare(device) before infer_unified"

        dec11, dec21, dec22, dec12 = self.forward(frame_i, frame_j)

        shape1, shape2 = frame_i.img_true_shape, frame_j.img_true_shape

        # Original MASt3R head
        res11, res21 = self._mast3r_head(dec11, dec21, shape1, shape2)
        res22, res12 = self._mast3r_head(dec22, dec12, shape2, shape1)

        X, C, D, Q = zip(*[
            (r["pts3d"][0], r["conf"][0], r["desc"][0], r["desc_conf"][0])
            for r in [res11, res21, res22, res12]
        ])
        X, C, D, Q = torch.stack(X), torch.stack(C), torch.stack(D), torch.stack(Q)
        X, C, D, Q = self._downsample(X, C, D, Q)

        # -- SegMASt3R branch --
        res0, res1 = self._segmast3r_head(dec11, dec21, shape1, shape2)


        # Extract descriptors from head outputs (shape: B, H, W, desc_dim)
        desc_i = res0["desc"].permute(0, 3, 1, 2)  # B, desc_dim, H, W
        desc_j = res1["desc"].permute(0, 3, 1, 2)  # B, desc_dim, H, W
        agg_desc_i = masked_average_pooling(desc_i, masks_i)
        agg_desc_j = masked_average_pooling(desc_j, masks_j)
        print("desc_i raw norm:", desc_i.norm(dim=1).mean().item())   # should be ~31
        print("desc_j raw norm:", desc_j.norm(dim=1).mean().item())   # should be ~31

        assert masks_i.device == desc_i.device, \
            f"Device mismatch: masks {masks_i.device} vs desc {desc_i.device}"
        
        if debug:
            with torch.no_grad():
                # Descriptor quality check: norms, variance, and pairwise similarity
                # agg_desc_i/j shape: (B, D, M/N)
                desc_i_norm = torch.norm(agg_desc_i, dim=1, keepdim=False)  # (B, M)
                desc_j_norm = torch.norm(agg_desc_j, dim=1, keepdim=False)  # (B, N)
                
                # Pairwise cosine similarity within each frame
                # Normalize descriptors
                desc_i_normalized = torch.nn.functional.normalize(agg_desc_i, dim=1)  # (B, D, M)
                desc_j_normalized = torch.nn.functional.normalize(agg_desc_j, dim=1)  # (B, D, N)
                
                # Cosine similarity: (B, M, M) and (B, N, N)
                cos_sim_i = torch.bmm(desc_i_normalized.transpose(1, 2), desc_i_normalized)  # (B, M, M)
                cos_sim_j = torch.bmm(desc_j_normalized.transpose(1, 2), desc_j_normalized)  # (B, N, N)
                
                # Off-diagonal mean similarity (diversity metric)
                mask_i_offdiag = ~torch.eye(cos_sim_i.shape[1], device=cos_sim_i.device, dtype=torch.bool).unsqueeze(0)
                mask_j_offdiag = ~torch.eye(cos_sim_j.shape[1], device=cos_sim_j.device, dtype=torch.bool).unsqueeze(0)
                offdiag_sim_i = cos_sim_i[mask_i_offdiag].mean() if mask_i_offdiag.sum() > 0 else torch.tensor(float('nan'))
                offdiag_sim_j = cos_sim_j[mask_j_offdiag].mean() if mask_j_offdiag.sum() > 0 else torch.tensor(float('nan'))
                
                print(
                    "[descriptor-quality]",
                    f"desc_i_norm_mean={desc_i_norm.mean().item():.4f}",
                    f"desc_i_norm_std={desc_i_norm.std().item():.4f}",
                    f"desc_j_norm_mean={desc_j_norm.mean().item():.4f}",
                    f"desc_j_norm_std={desc_j_norm.std().item():.4f}",
                    f"desc_i_offdiag_cosine={offdiag_sim_i.item():.4f}",
                    f"desc_j_offdiag_cosine={offdiag_sim_j.item():.4f}",
                )

        log_P_dustb = self.feature_matcher(agg_desc_i, agg_desc_j)
        scores = torch.exp(log_P_dustb)

        if debug:
            with torch.no_grad():
                score_stats = scores.detach()
                non_dustbin = score_stats[:, : max(score_stats.shape[1] - 1, 0), : max(score_stats.shape[2] - 1, 0)]
                dustbin_row = score_stats[:, :-1, -1]
                dustbin_col = score_stats[:, -1, :-1]
                print(
                    "[unified-debug]",
                    f"agg_desc_i={tuple(agg_desc_i.shape)}",
                    f"agg_desc_j={tuple(agg_desc_j.shape)}",
                    f"scores={tuple(scores.shape)}",
                    f"non_dustbin_max={non_dustbin.max().item() if non_dustbin.numel() else float('nan'):.4f}",
                    f"non_dustbin_mean={non_dustbin.mean().item() if non_dustbin.numel() else float('nan'):.4f}",
                    f"dustbin_row_mean={dustbin_row.mean().item() if dustbin_row.numel() else float('nan'):.4f}",
                    f"dustbin_col_mean={dustbin_col.mean().item() if dustbin_col.numel() else float('nan'):.4f}",
                )

        B, M_p1, N_p1 = scores.shape
        M, N = M_p1 - 1, N_p1 - 1

        ref_to_target  = scores[:, :M, :].argmax(dim=-1)
        target_to_ref  = scores[:, :, :N].argmax(dim=-2)
        is_not_dustbin = ref_to_target < N
        safe_indices   = ref_to_target.clamp(max=N - 1)
        reciprocal     = torch.gather(target_to_ref, 1, safe_indices)
        row_indices    = torch.arange(M, device=self.device).unsqueeze(0).expand(B, M)
        is_mutual      = reciprocal == row_indices

        if debug:
            with torch.no_grad():
                print(
                    "[unified-debug]",
                    f"ref_to_target_valid={(is_not_dustbin).sum().item()}/{is_not_dustbin.numel()}",
                    f"mutual_valid={(is_mutual).sum().item()}/{is_mutual.numel()}",
                    f"final_valid={((is_not_dustbin & is_mutual).sum().item())}/{is_not_dustbin.numel()}",
                )

        match_result = torch.full((B, M), -1, dtype=torch.int64, device=self.device)
        valid = is_not_dustbin & is_mutual
        match_result[valid] = ref_to_target[valid]

        return (X, C, D, Q), match_result, agg_desc_i, agg_desc_j
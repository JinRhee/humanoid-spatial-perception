import torch
import einops
import mast3r.model as mast3r_model

from mast3r_slam.config import config
import mast3r_slam.matching as matching
from src.models.mast3r_segfeat.diff_feature_matcher import featureMatcher
from src.models.mast3r_segfeat.diff_masked_pooling import masked_average_pooling


class UnifiedMASt3RInfer(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        model_params = self._get_model_params(cfg)
        self.encoder = mast3r_model.AsymmetricMASt3R(**model_params)

        cfg["FEATURE_MATCHER"]["SINKHORN"]["DUSTBIN_SCORE_INIT"] = 5.3937
        self.feature_matcher = featureMatcher(cfg["FEATURE_MATCHER"])

        self._configure_grad()

    def _configure_grad(self):
        self.encoder.requires_grad_(False)
        self.feature_matcher.requires_grad_(False)

    def _get_model_params(self, cfg):
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

        dataset_type = cfg["DATASET"]["DATA_SOURCE"].lower()
        if dataset_type == "mapfree" or "hm3d":
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
            frame.feat, frame.pos, _ = self.encoder._encode_image(
                frame.img, frame.img_true_shape
            )

    @torch.inference_mode()
    def _decoder(self, feat1, pos1, feat2, pos2, shape1, shape2):
        dec1, dec2 = self.encoder._decoder(feat1, pos1, feat2, pos2)
        with torch.amp.autocast(enabled=False, device_type="cuda"):
            res1 = self.encoder._downstream_head(
                1, [tok.float() for tok in dec1], shape1
            )
            res2 = self.encoder._downstream_head(
                2, [tok.float() for tok in dec2], shape2
            )
        return res1, res2

    def _downsample(self, X, C, D, Q):
        factor = config["dataset"]["img_downsample"]
        if factor > 1:
            X = X[..., ::factor, ::factor, :].contiguous()
            C = C[..., ::factor, ::factor].contiguous()
            D = D[..., ::factor, ::factor, :].contiguous()
            Q = Q[..., ::factor, ::factor].contiguous()
        return X, C, D, Q

    # ------------------------------------------------------------------
    # Core forward: shared trunk
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def forward(self, frame_i, frame_j):
        """
        Run the full symmetric MASt3R inference once.
        Returns the four raw result dicts for downstream branching.
        """
        self._encode_frame(frame_i)
        self._encode_frame(frame_j)

        feat_i, pos_i, shape_i = frame_i.feat, frame_i.pos, frame_i.img_true_shape
        feat_j, pos_j, shape_j = frame_j.feat, frame_j.pos, frame_j.img_true_shape

        # Pass 1: i as reference, j as query
        res11, res21 = self._decoder(feat_i, pos_i, feat_j, pos_j, shape_i, shape_j)
        # Pass 2: j as reference, i as query
        res22, res12 = self._decoder(feat_j, pos_j, feat_i, pos_i, shape_j, shape_i)

        return res11, res21, res22, res12

    # ------------------------------------------------------------------
    # SLAM branch
    # ------------------------------------------------------------------

    def infer_slam(self, frame_i, frame_j):
        """
        SLAM branch. Returns X, C, D, Q stacked across all four
        decoder outputs, downsampled per config.

        Output shapes (before downsample): (4, H, W, C).
        Ordering: [res11, res21, res22, res12]
        i.e.      [Xii,   Xji,   Xjj,   Xij]
        """
        res11, res21, res22, res12 = self.forward(frame_i, frame_j)

        X, C, D, Q = zip(*[
            (r["pts3d"][0], r["conf"][0], r["desc"][0], r["desc_conf"][0])
            for r in [res11, res21, res22, res12]
        ])
        X, C, D, Q = torch.stack(X), torch.stack(C), torch.stack(D), torch.stack(Q)
        X, C, D, Q = self._downsample(X, C, D, Q)

        return X, C, D, Q

    def infer_slam_and_match(self, frame_i, frame_j):
        """
        SLAM branch including symmetric matching.
        Mirrors mast3r_match_symmetric() from mast3r_utils.py.
        """
        X, C, D, Q = self.infer_slam(frame_i, frame_j)

        b = X.shape[0] // 4  # batch size

        Xii, Xji, Xjj, Xij = X[0], X[1], X[2], X[3]
        Dii, Dji, Djj, Dij = D[0], D[1], D[2], D[3]
        Qii, Qji, Qjj, Qij = Q[0], Q[1], Q[2], Q[3]

        X11 = torch.cat((Xii, Xjj), dim=0)
        X21 = torch.cat((Xji, Xij), dim=0)
        D11 = torch.cat((Dii, Djj), dim=0)
        D21 = torch.cat((Dji, Dij), dim=0)

        idx_1_to_2, valid_match_2 = matching.match(X11, X21, D11, D21)

        match_b = X11.shape[0] // 2
        idx_i2j,     idx_j2i     = idx_1_to_2[:match_b],    idx_1_to_2[match_b:]
        valid_match_j, valid_match_i = valid_match_2[:match_b], valid_match_2[match_b:]

        return (
            idx_i2j, idx_j2i,
            valid_match_j, valid_match_i,
            Qii.view(b, -1, 1), Qjj.view(b, -1, 1),
            Qji.view(b, -1, 1), Qij.view(b, -1, 1),
        )

    # ------------------------------------------------------------------
    # SegMASt3R branch
    # ------------------------------------------------------------------

    def infer_seg(self, frame_i, frame_j, masks_i, masks_j):
        """
        SegMASt3R branch. Mirrors infer_pair() from model_infer.py.
        Uses only the i->j decoder direction (res11, res21).

        Args:
            masks_i: (B, M, H, W) — segment masks for frame_i
            masks_j: (B, N, H, W) — segment masks for frame_j

        Returns:
            match_result: (B, M) tensor, values in [-1, N-1].
                          -1 indicates unmatched (dustbin or failed MNN).
        """
        assert hasattr(self, "device"), "Call model.prepare(device) before infer_seg"

        res11, res21, _, _ = self.forward(frame_i, frame_j)

        # (B, H, W, 24) -> (B, 24, H, W)
        desc_i = res11["desc"].permute(0, 3, 1, 2)
        desc_j = res21["desc"].permute(0, 3, 1, 2)

        agg_i = masked_average_pooling(desc_i, masks_i)   # (B, 24, M)
        agg_j = masked_average_pooling(desc_j, masks_j)   # (B, 24, N)

        log_P_dustb = self.feature_matcher(agg_i, agg_j)  # (B, M+1, N+1)
        scores = torch.exp(log_P_dustb)

        B, M_p1, N_p1 = scores.shape
        M, N = M_p1 - 1, N_p1 - 1

        # Mutual nearest neighbour matching (from model_infer.py)
        ref_to_target  = scores[:, :M, :].argmax(dim=-1)   # (B, M)
        target_to_ref  = scores[:, :, :N].argmax(dim=-2)   # (B, N)
        is_not_dustbin = ref_to_target < N
        safe_indices   = ref_to_target.clamp(max=N - 1)
        reciprocal     = torch.gather(target_to_ref, 1, safe_indices)
        row_indices    = torch.arange(M, device=self.device).unsqueeze(0).expand(B, M)
        is_mutual      = reciprocal == row_indices

        match_result = torch.full((B, M), -1, dtype=torch.int64, device=self.device)
        valid = is_not_dustbin & is_mutual
        match_result[valid] = ref_to_target[valid]

        return match_result

    # ------------------------------------------------------------------
    # Unified: both branches in one call
    # ------------------------------------------------------------------

    def infer_unified(self, frame_i, frame_j, masks_i, masks_j):
        """
        Run both branches from a single MASt3R inference pass.

        Returns:
            slam:  (X, C, D, Q) — for matching.match()
            seg:   match_result (B, M) — segment correspondences
        """
        assert hasattr(self, "device"), "Call model.prepare(device) before infer_unified"

        res11, res21, res22, res12 = self.forward(frame_i, frame_j)

        # -- SLAM branch --
        X, C, D, Q = zip(*[
            (r["pts3d"][0], r["conf"][0], r["desc"][0], r["desc_conf"][0])
            for r in [res11, res21, res22, res12]
        ])
        X, C, D, Q = torch.stack(X), torch.stack(C), torch.stack(D), torch.stack(Q)
        X, C, D, Q = self._downsample(X, C, D, Q)

        # -- SegMASt3R branch --
        desc_i = res11["desc"].permute(0, 3, 1, 2)
        desc_j = res21["desc"].permute(0, 3, 1, 2)
        agg_i  = masked_average_pooling(desc_i, masks_i)
        agg_j  = masked_average_pooling(desc_j, masks_j)

        log_P_dustb = self.feature_matcher(agg_i, agg_j)
        scores = torch.exp(log_P_dustb)

        B, M_p1, N_p1 = scores.shape
        M, N = M_p1 - 1, N_p1 - 1

        ref_to_target  = scores[:, :M, :].argmax(dim=-1)
        target_to_ref  = scores[:, :, :N].argmax(dim=-2)
        is_not_dustbin = ref_to_target < N
        safe_indices   = ref_to_target.clamp(max=N - 1)
        reciprocal     = torch.gather(target_to_ref, 1, safe_indices)
        row_indices    = torch.arange(M, device=self.device).unsqueeze(0).expand(B, M)
        is_mutual      = reciprocal == row_indices

        match_result = torch.full((B, M), -1, dtype=torch.int64, device=self.device)
        valid = is_not_dustbin & is_mutual
        match_result[valid] = ref_to_target[valid]

        return (X, C, D, Q), match_result
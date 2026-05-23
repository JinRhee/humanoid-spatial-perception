import torch
import einops
import mast3r.model as mast3r_model
# from mast3r.catmlp_dpt_head import mast3r_head_factory
from mast3r_src.mast3r.catmlp_dpt_head import mast3r_head_factory


from pathlib import Path
from mast3r_slam.config import config
import mast3r_slam.matching as matching
from external.segmast3r.src.models.mast3r_segfeat.diff_feature_matcher import featureMatcher
from external.segmast3r.src.models.mast3r_segfeat.diff_masked_pooling import masked_average_pooling

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
                    "DUSTBIN_SCORE_INIT": 5.3937,
                },
            }

        # ── Shared trunk ──────────────────────────────────────────
        # Load from MASt3R-SLAM checkpoint.
        # Encoder + decoder + original SLAM heads (pts3d, conf, desc).
        self.mast3r = mast3r_model.AsymmetricMASt3R.from_pretrained(mast3r_ckpt)

        # ── SegMASt3R heads ───────────────────────────────────────
        # Built with the same factory as the original heads so the
        # architecture is identical, then loaded with SegMASt3R weights.
        # These produce the fine-tuned descriptors for segment matching.
        head_type   = "catmlp+dpt"
        output_mode = "pts3d+desc24"

        self.seg_head1 = mast3r_head_factory(
            head_type, output_mode, self.mast3r, has_conf=True
        )
        self.seg_head2 = mast3r_head_factory(
            head_type, output_mode, self.mast3r, has_conf=True
        )

        # Load SegMASt3R weights into seg heads
        seg_ckpt = torch.load(segmast3r_ckpt, map_location="cpu", weights_only=False)
        seg_state = seg_ckpt["state_dict"]

        def extract(prefix):
            matched = {
                k[len(prefix):]: v
                for k, v in seg_state.items()
                if k.startswith(prefix)
            }
            print(f"[UnifiedMASt3RInfer] extracted {len(matched)} keys for '{prefix}'")
            return matched

        self.seg_head1.load_state_dict(extract("encoder.downstream_head1."))
        self.seg_head2.load_state_dict(extract("encoder.downstream_head2."))


        # ── Feature matcher ───────────────────────────────────────
        self.feature_matcher = featureMatcher(feature_matcher_cfg)

        matcher_state = {
            k[len("feature_matcher."):]: v
            for k, v in seg_state.items()
            if k.startswith("feature_matcher.")
        }
        if matcher_state:
            self.feature_matcher.load_state_dict(matcher_state)
            print(f"[UnifiedMASt3RInfer] loaded feature matcher state, "
                  f"dustbin={self.feature_matcher.matching_mat.dustbin_score.item():.4f}")

        self._configure_grad()

    def _configure_grad(self):
        self.mast3r.requires_grad_(False)
        self.seg_head1.requires_grad_(False)
        self.seg_head2.requires_grad_(False)
        self.feature_matcher.requires_grad_(False)

    def prepare(self, device):
        self.device = device
        self.eval()
        self.to(device)

    # ------------------------------------------------------------------
    # Encoder (cached per frame)
    # ------------------------------------------------------------------

    def _encode_frame(self, frame):
        if frame.feat is None:
            frame.feat, frame.pos, _ = self.mast3r._encode_image(
                frame.img, frame.img_true_shape
            )

    # ------------------------------------------------------------------
    # Decoder (shared trunk)
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def _decoder(self, feat1, pos1, feat2, pos2):
        return self.mast3r._decoder(feat1, pos1, feat2, pos2)

    # ------------------------------------------------------------------
    # SLAM heads — original MASt3R weights
    # pts3d and conf for geometry; desc used by tracker's matcher
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # SegMASt3R heads — fine-tuned weights
    # desc used for segment matching; pts3d/conf ignored
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def _segmast3r_head(self, dec1, dec2, shape1, shape2):
        shape1_sq = shape1.squeeze(0)
        shape2_sq = shape2.squeeze(0)
        with torch.amp.autocast(enabled=False, device_type="cuda"):
            res1 = self.seg_head1([tok.float() for tok in dec1], shape1_sq)
            res2 = self.seg_head2([tok.float() for tok in dec2], shape2_sq)
        return res1, res2

    # ------------------------------------------------------------------
    # Downsampling (SLAM branch only)
    # ------------------------------------------------------------------

    def _downsample(self, X, C, D, Q):
        factor = config["dataset"]["img_downsample"]
        if factor > 1:
            X = X[..., ::factor, ::factor, :].contiguous()
            C = C[..., ::factor, ::factor].contiguous()
            D = D[..., ::factor, ::factor, :].contiguous()
            Q = Q[..., ::factor, ::factor].contiguous()
        return X, C, D, Q

    # ------------------------------------------------------------------
    # Shared forward: encoder + symmetric decoder, returns raw tokens
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def forward(self, frame_i, frame_j):
        self._encode_frame(frame_i)
        self._encode_frame(frame_j)

        feat_i, pos_i = frame_i.feat, frame_i.pos
        feat_j, pos_j = frame_j.feat, frame_j.pos

        dec11, dec21 = self._decoder(feat_i, pos_i, feat_j, pos_j)
        dec22, dec12 = self._decoder(feat_j, pos_j, feat_i, pos_i)

        return dec11, dec21, dec22, dec12

    # ------------------------------------------------------------------
    # Unified inference
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def infer_unified(self, frame_i, frame_j, masks_i, masks_j, debug=False):
        assert hasattr(self, "device"), "Call model.prepare(device) before infer_unified"

        dec11, dec21, dec22, dec12 = self.forward(frame_i, frame_j)
        shape1, shape2 = frame_i.img_true_shape, frame_j.img_true_shape

        # -- SLAM branch: original heads --
        res11, res21 = self._mast3r_head(dec11, dec21, shape1, shape2)
        res22, res12 = self._mast3r_head(dec22, dec12, shape2, shape1)

        X, C, D, Q = zip(*[
            (r["pts3d"][0], r["conf"][0], r["desc"][0], r["desc_conf"][0])
            for r in [res11, res21, res22, res12]
        ])
        X, C, D, Q = torch.stack(X), torch.stack(C), torch.stack(D), torch.stack(Q)
        X, C, D, Q = self._downsample(X, C, D, Q)

        # -- SegMASt3R branch: fine-tuned heads --
        # Only need the i->j direction for descriptors
        seg11, seg21 = self._segmast3r_head(dec11, dec21, shape1, shape2)

        desc_i = seg11["desc"].permute(0, 3, 1, 2)  # (B, 24, H, W)
        desc_j = seg21["desc"].permute(0, 3, 1, 2)

        if debug:
            with torch.no_grad():
                raw_norm_i = desc_i.norm(dim=1).mean().item()
                raw_norm_j = desc_j.norm(dim=1).mean().item()
                print(f"[seg-head] desc_i raw norm={raw_norm_i:.4f} "
                      f"desc_j raw norm={raw_norm_j:.4f}")

        agg_desc_i = masked_average_pooling(desc_i, masks_i)  # (B, 24, M)
        agg_desc_j = masked_average_pooling(desc_j, masks_j)  # (B, 24, N)

        if debug:
            with torch.no_grad():
                norm_i = agg_desc_i.norm(dim=1).mean().item()
                norm_j = agg_desc_j.norm(dim=1).mean().item()
                desc_i_n = torch.nn.functional.normalize(agg_desc_i, dim=1)
                desc_j_n = torch.nn.functional.normalize(agg_desc_j, dim=1)
                cos_i = torch.bmm(desc_i_n.transpose(1,2), desc_i_n)
                cos_j = torch.bmm(desc_j_n.transpose(1,2), desc_j_n)
                eye_i = ~torch.eye(cos_i.shape[1], device=cos_i.device, dtype=torch.bool).unsqueeze(0)
                eye_j = ~torch.eye(cos_j.shape[1], device=cos_j.device, dtype=torch.bool).unsqueeze(0)
                print(
                    "[descriptor-quality]",
                    f"agg_norm_i={norm_i:.4f}",
                    f"agg_norm_j={norm_j:.4f}",
                    f"offdiag_cos_i={cos_i[eye_i].mean().item():.4f}",
                    f"offdiag_cos_j={cos_j[eye_j].mean().item():.4f}",
                )

        log_P_dustb = self.feature_matcher(agg_desc_i, agg_desc_j)
        scores = torch.exp(log_P_dustb)

        if debug:
            with torch.no_grad():
                nd = scores[:, :-1, :-1]
                print(
                    "[unified-debug]",
                    f"scores={tuple(scores.shape)}",
                    f"non_dustbin_max={nd.max().item():.4f}",
                    f"non_dustbin_mean={nd.mean().item():.4f}",
                    f"dustbin_row_mean={scores[:, :-1, -1].mean().item():.4f}",
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

        match_result = torch.full((B, M), -1, dtype=torch.int64, device=self.device)
        valid = is_not_dustbin & is_mutual
        match_result[valid] = ref_to_target[valid]

        return (X, C, D, Q), match_result, agg_desc_i, agg_desc_j
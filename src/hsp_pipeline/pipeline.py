import argparse
import datetime
import os
import pathlib
import sys
import time
import cv2
import numpy as np
import lietorch
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import tqdm
import yaml

from pathlib import Path

from mast3r_slam.global_opt import FactorGraph
from mast3r_slam.config import load_config, config, set_global_config
from mast3r_slam.dataloader import Intrinsics, load_dataset
import mast3r_slam.evaluate as eval
from mast3r_slam.frame import Mode, SharedKeyframes, SharedStates, create_frame
from mast3r_slam.mast3r_utils import (
    load_mast3r,
    load_retriever,
    mast3r_inference_mono,
)
from mast3r_slam.multiprocess_utils import new_queue, try_get_msg
from mast3r_slam.tracker import FrameTracker
from mast3r_slam.visualization import WindowMsg, run_visualization

from .unified_inference import UnifiedMASt3RInfer
from .instance_tracker_new import (
    InstanceTracker,
    SegmentationStore,
    build_segment_records,
    MergeWeights,
    SegmentRecord,
    instance_rgba,
)
from .segmentor import SegmentationPipeline
from .config import load_pipeline_config, overrides_from_args

def relocalization(frame, keyframes, factor_graph, retrieval_database):
    # we are adding and then removing from the keyframe, so we need to be careful.
    # The lock slows viz down but safer this way...
    with keyframes.lock:
        kf_idx = []
        retrieval_inds = retrieval_database.update(
            frame,
            add_after_query=False,
            k=config["retrieval"]["k"],
            min_thresh=config["retrieval"]["min_thresh"],
        )
        kf_idx += retrieval_inds
        successful_loop_closure = False
        if kf_idx:
            keyframes.append(frame)
            n_kf = len(keyframes)
            kf_idx = list(kf_idx)  # convert to list
            frame_idx = [n_kf - 1] * len(kf_idx)
            print("RELOCALIZING against kf ", n_kf - 1, " and ", kf_idx)
            if factor_graph.add_factors(
                frame_idx,
                kf_idx,
                config["reloc"]["min_match_frac"],
                is_reloc=config["reloc"]["strict"],
            ):
                retrieval_database.update(
                    frame,
                    add_after_query=True,
                    k=config["retrieval"]["k"],
                    min_thresh=config["retrieval"]["min_thresh"],
                )
                print("Success! Relocalized")
                successful_loop_closure = True
                keyframes.T_WC[n_kf - 1] = keyframes.T_WC[kf_idx[0]].clone()
            else:
                keyframes.pop_last()
                print("Failed to relocalize")

        if successful_loop_closure:
            if config["use_calib"]:
                factor_graph.solve_GN_calib()
            else:
                factor_graph.solve_GN_rays()
        return successful_loop_closure


def build_instance_color_image(segments, masks, height: int, width: int) -> np.ndarray:
    color_image = np.zeros((height, width, 3), dtype=np.float32)
    masks_np = masks[0].detach().cpu().numpy() if hasattr(masks, "detach") else np.asarray(masks)
    for segment in segments:
        if segment.instance_id is None or not getattr(segment, "is_matched", False):
            continue
        if segment.mask_id < 0 or segment.mask_id >= masks_np.shape[0]:
            continue
        mask = masks_np[segment.mask_id] > 0
        if not np.any(mask):
            continue
        color_image[mask] = instance_rgba(segment.instance_id, alpha=1.0)[:3]
    return np.ascontiguousarray(color_image)


def run_backend(cfg, model, states, keyframes, K, retrieval_path):
    import torch.serialization
    torch.serialization.add_safe_globals([argparse.Namespace])
    
    set_global_config(cfg)

    device = keyframes.device
    factor_graph = FactorGraph(model, keyframes, K, device)
    retrieval_database = load_retriever(model, retriever_path=retrieval_path)

    mode = states.get_mode()
    while mode is not Mode.TERMINATED:
        mode = states.get_mode()
        if mode == Mode.INIT or states.is_paused():
            time.sleep(0.01)
            continue
        if mode == Mode.RELOC:
            frame = states.get_frame()
            success = relocalization(frame, keyframes, factor_graph, retrieval_database)
            if success:
                states.set_mode(Mode.TRACKING)
            states.dequeue_reloc()
            continue
        idx = -1
        with states.lock:
            if len(states.global_optimizer_tasks) > 0:
                idx = states.global_optimizer_tasks[0]
        if idx == -1:
            time.sleep(0.01)
            continue

        # Graph Construction
        kf_idx = []
        # k to previous consecutive keyframes
        n_consec = 1
        for j in range(min(n_consec, idx)):
            kf_idx.append(idx - 1 - j)
        frame = keyframes[idx]
        retrieval_inds = retrieval_database.update(
            frame,
            add_after_query=True,
            k=config["retrieval"]["k"],
            min_thresh=config["retrieval"]["min_thresh"],
        )
        kf_idx += retrieval_inds

        lc_inds = set(retrieval_inds)
        lc_inds.discard(idx - 1)
        if len(lc_inds) > 0:
            print("Database retrieval", idx, ": ", lc_inds)

        kf_idx = set(kf_idx)  # Remove duplicates by using set
        kf_idx.discard(idx)  # Remove current kf idx if included
        kf_idx = list(kf_idx)  # convert to list
        frame_idx = [idx] * len(kf_idx)
        if kf_idx:
            factor_graph.add_factors(
                kf_idx, frame_idx, config["local_opt"]["min_match_frac"]
            )

        with states.lock:
            states.edges_ii[:] = factor_graph.ii.cpu().tolist()
            states.edges_jj[:] = factor_graph.jj.cpu().tolist()

        if config["use_calib"]:
            factor_graph.solve_GN_calib()
        else:
            factor_graph.solve_GN_rays()

        with states.lock:
            if len(states.global_optimizer_tasks) > 0:
                idx = states.global_optimizer_tasks.pop(0)


def run_pipeline(args):
    mp.set_start_method("spawn")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)
    device = "cuda:0"
    save_frames = False
    datetime_now = str(datetime.datetime.now()).replace(" ", "_")

    
    app_config = load_pipeline_config(
        args.config,
        preset=getattr(args, "preset", None),
        overrides=overrides_from_args(args),
    )
    print(app_config)

    load_config(app_config["slam"]["config"])   # MASt3R-SLAM uses global variable named "config"
    print(config)

    no_viz    = app_config["slam"]["no_viz"]

    manager = mp.Manager()
    main2viz = new_queue(manager, no_viz)
    viz2main = new_queue(manager, no_viz)

    dataset   = load_dataset(app_config["dataset"]["path"])
    dataset.subsample(config["dataset"]["subsample"])
    h, w = dataset.get_img_shape()[0]

    if args.calib:
        with open(args.calib, "r") as f:
            intrinsics = yaml.load(f, Loader=yaml.SafeLoader)
        config["use_calib"] = True
        dataset.use_calibration = True
        dataset.camera_intrinsics = Intrinsics.from_calib(
            dataset.img_size,
            intrinsics["width"],
            intrinsics["height"],
            intrinsics["calibration"],
        )

    keyframes = SharedKeyframes(manager, h, w)
    states = SharedStates(manager, h, w)

    if not no_viz:
        viz = mp.Process(
            target=run_visualization,
            args=(config, states, keyframes, main2viz, viz2main),
        )
        viz.start()

    # ---------------------------------------------------------------
    # Set calibration
    # ---------------------------------------------------------------
    has_calib = dataset.has_calib()
    use_calib = config["use_calib"]

    if use_calib and not has_calib:
        print("[Warning] No calibration provided for this dataset!")
        sys.exit(0)
    K = None
    if use_calib:
        K = torch.from_numpy(dataset.camera_intrinsics.K_frame).to(
            device, dtype=torch.float32
        )
        keyframes.set_intrinsics(K)

    # ---------------------------------------------------------------
    # Remove previous trajectory
    # ---------------------------------------------------------------
    # remove the trajectory from the previous run
    if dataset.save_results:
        args.save_as = app_config["dataset"]["save_as"]
        save_dir, seq_name = eval.prepare_savedir(args, dataset)
        traj_file = save_dir / f"{seq_name}.txt"
        recon_file = save_dir / f"{seq_name}.ply"
        if traj_file.exists():
            traj_file.unlink()
        if recon_file.exists():
            recon_file.unlink()
    
    # ---------------------------------------------------------------
    # Initialize models
    # ---------------------------------------------------------------

    # Unified MASt3R Infer
    model = UnifiedMASt3RInfer(
        mast3r_ckpt=app_config["paths"]["checkpoints"]["mast3r_original"],
        segmast3r_ckpt=app_config["paths"]["checkpoints"]["segmast3r_head"],
    )
    model.prepare(device)
    model.share_memory()

    # FastSAM
    segmentor = SegmentationPipeline(
        app_config["paths"]["checkpoints"]["fastsam"],
        seg_config=app_config.get("segmentation")
    )

    # ---------------------------------------------------------------
    # FrameTracker, SegmentationStore, InstanceTracker
    # ---------------------------------------------------------------

    # FrameTracker
    tracker = FrameTracker(model.mast3r, keyframes, device)
    last_msg = WindowMsg()

    # SegmentationStore
    seg_store = SegmentationStore()

    # InstanceTracker
    tracker_config = app_config["instance_tracker"].copy()
    tracker_config["merge_weights"] = MergeWeights(**tracker_config["merge_weights"])
    instance_tracker = InstanceTracker(**tracker_config)

    # ---------------------------------------------------------------
    # Start MASt3R-SLAM backend
    # ---------------------------------------------------------------
    
    torch.serialization.add_safe_globals([argparse.Namespace])
    
    retrieval_path = str(Path(app_config["paths"]["checkpoints"]["mast3r_retrieval"]).resolve())
    backend = mp.Process(target=run_backend, args=(config, model.mast3r, states, keyframes, K, retrieval_path))
    backend.start()

    # ---------------------------------------------------------------
    # Seed frame 0 before loop
    # ---------------------------------------------------------------

    _, img0 = dataset[0]
    prev_img_np = (img0 * 255).clip(0, 255).astype(np.uint8)
    prev_seg_result = segmentor.segment(prev_img_np)
    prev_masks_raw = (
        prev_seg_result.masks.data
        if prev_seg_result.masks is not None
        else torch.zeros((0, prev_img_np.shape[0], prev_img_np.shape[1]),
                        dtype=torch.uint8, device=device)
    )
    prev_frame_obj = None   # set after INIT fires on i=0

    # ---------------------------------------------------------------
    # MASt3R-SLAM pipeline
    # ---------------------------------------------------------------

    i = 0
    fps_timer = time.time()

    frames = []

    while True:
        mode = states.get_mode()
        print(f"Mode:{mode}")
        msg = try_get_msg(viz2main)
        last_msg = msg if msg is not None else last_msg
        if last_msg.is_terminated:
            states.set_mode(Mode.TERMINATED)
            break

        if last_msg.is_paused and not last_msg.next:
            states.pause()
            time.sleep(0.01)
            continue

        if not last_msg.is_paused:
            states.unpause()

        if i == len(dataset):
            states.set_mode(Mode.TERMINATED)
            break

        timestamp, img = dataset[i]
        if save_frames:
            frames.append(img)

        instance_color_image = None

        # get frames last camera pose
        T_WC = (
            lietorch.Sim3.Identity(1, device=device)
            if i == 0
            else states.get_frame().T_WC
        )
        frame = create_frame(i, img, T_WC, img_size=dataset.img_size, device=device)

        if mode == Mode.INIT:
            # Initialize via mono inference, and encoded features neeed for database
            X_init, C_init = mast3r_inference_mono(model.mast3r, frame)
            frame.update_pointmap(X_init, C_init)
            keyframes.append(frame)
            states.queue_global_optimization(len(keyframes) - 1)
            states.set_mode(Mode.TRACKING)
            states.set_frame(frame)

            prev_frame_obj = frame
            i += 1
            continue

        if mode == Mode.TRACKING:
            # -- Segment current frame --
            curr_img_np = (img * 255).clip(0, 255).astype(np.uint8) # img is in np.float32
            curr_seg_result = segmentor.segment(curr_img_np)
            curr_masks_raw = (
                curr_seg_result.masks.data
                if curr_seg_result.masks is not None
                else torch.zeros((0, curr_img_np.shape[0], curr_img_np.shape[1]),
                                dtype=torch.uint8, device=device)
            )

            # -- Resize masks to MASt3R feature map resolution --
            _, _, H_feat, W_feat = prev_frame_obj.img.shape

            def resize_masks(masks, h, w):
                if masks.shape[0] == 0:
                    return torch.zeros((0, h, w), dtype=masks.dtype, device=masks.device)
                return F.interpolate(
                    masks.unsqueeze(1).float(), size=(h, w), mode="nearest"
                ).squeeze(1)

            masks_i = resize_masks(prev_masks_raw, H_feat, W_feat).unsqueeze(0)  # (1, M, H, W)
            masks_j = resize_masks(curr_masks_raw, H_feat, W_feat).unsqueeze(0)  # (1, N, H, W)

            # -- Unified inference --
            if masks_i.shape[1] > 0 and masks_j.shape[1] > 0:
                (X, C, D, Q), match_result, agg_desc_i, agg_desc_j = model.infer_unified(
                    prev_frame_obj,
                    frame,
                    masks_i,
                    masks_j,
                    debug=os.environ.get("HSP_DEBUG_UNIFIED", "0") == "1",
                )

                # -- Build SegmentRecords for current frame --
                # X[0] is res11 pts3d: current frame as reference, (H, W, 3)
                # C[0] is res11 conf:  (H, W)
                # agg_desc_j is pooled descriptors for current frame: (1, 24, N)
                # pose_world: use frame.T_WC to transform points into world frame
                pose_world = frame.T_WC.matrix().squeeze(0).cpu().numpy()  # (4, 4)

                segment_records = build_segment_records(
                    frame_index=i,
                    masks=masks_j,                          # (1, N, H, W)
                    points_3d=X[0],                         # (H, W, 3)
                    conf=C[0],                              # (H, W)
                    descriptors=agg_desc_j,                 # (1, 24, N)
                    pose_world=pose_world,
                    frame_ts=timestamp,
                    label=None,                             # FastSAM is class-agnostic
                    score=1.0,                              # placeholder
                )

                # -- Store results --
                match_result_cpu = match_result[0].detach().cpu().numpy()
                seg_store.add_segments(i, segment_records)
                seg_store.add_match_result(i - 1, i, match_result_cpu)
                print(match_result_cpu)

                # -- Update instance tracker --
                prev_match_result = seg_store.get_match_result(i - 1, i)
                instance_tracker.update_frame(
                    segments=segment_records,
                    frame_index=i,
                    match_result=prev_match_result,
                )

                instance_color_image = build_instance_color_image(
                    segment_records,
                    masks_j,
                    H_feat,
                    W_feat,
                )
                main2viz.put(
                    {
                        "frame_index": i,
                        "instances": instance_tracker.summaries(),
                        "point_colors": instance_color_image,
                    }
                )

                print(
                    f"[seg] {i-1}->{i} | "
                    f"masks: {masks_i.shape[1]}/{masks_j.shape[1]} | "
                    f"matches: {np.sum(match_result_cpu >= 0)} | "
                    f"instances: {len(instance_tracker.valid_instances)} | "
                    f"candidates: {len(instance_tracker.candidate_instances)}"
                )

            else:
                # No masks on one or both sides — SLAM only, no seg update
                print(f"[seg] {i-1}->{i} | skipped (masks: {masks_i.shape[1]}/{masks_j.shape[1]})")

            # -- SLAM tracking (always runs) --
            # Note: tracker internally re-decodes via mast3r_match_asymmetric.
            # Encoder cache on frame means _encode_image is skipped.
            add_new_kf, match_info, try_reloc = tracker.track(frame)
            if try_reloc:
                states.set_mode(Mode.RELOC)
            states.set_frame(frame)

            # -- Roll forward --
            prev_frame_obj = frame
            prev_masks_raw = curr_masks_raw

        elif mode == Mode.RELOC:
            X, C = mast3r_inference_mono(model.mast3r, frame)
            frame.update_pointmap(X, C)
            states.set_frame(frame)
            states.queue_reloc()
            # In single threaded mode, make sure relocalization happen for every frame
            while config["single_thread"]:
                with states.lock:
                    if states.reloc_sem.value == 0:
                        break
                time.sleep(0.01)

        else:
            raise Exception("Invalid mode")

        if add_new_kf:
            keyframes.append(frame)
            states.queue_global_optimization(len(keyframes) - 1)
            if instance_color_image is not None:
                main2viz.put(
                    {
                        "frame_index": i,
                        "keyframe_frame_id": int(frame.frame_id),
                        "keyframe_point_colors": instance_color_image,
                    }
                )
            # In single threaded mode, wait for the backend to finish
            while config["single_thread"]:
                with states.lock:
                    if len(states.global_optimizer_tasks) == 0:
                        break
                time.sleep(0.01)
        # log time
        if i % 30 == 0:
            FPS = i / (time.time() - fps_timer)
            print(f"FPS: {FPS}")
        i += 1

    if dataset.save_results:
        save_dir, seq_name = eval.prepare_savedir(args, dataset)
        eval.save_traj(save_dir, f"{seq_name}.txt", dataset.timestamps, keyframes)
        eval.save_reconstruction(
            save_dir,
            f"{seq_name}.ply",
            keyframes,
            last_msg.C_conf_threshold,
        )
        eval.save_keyframes(
            save_dir / "keyframes" / seq_name, dataset.timestamps, keyframes
        )
    if save_frames:
        savedir = pathlib.Path(f"logs/frames/{datetime_now}")
        savedir.mkdir(exist_ok=True, parents=True)
        for i, frame in tqdm.tqdm(enumerate(frames), total=len(frames)):
            frame = (frame * 255).clip(0, 255)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"{savedir}/{i}.png", frame)

    print("done")
    backend.join()
    if not no_viz:
        viz.join()
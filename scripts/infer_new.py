#!/usr/bin/env python3

"""Infers pose from objects."""

import datetime

import json
import os
from pathlib import Path
from PIL import Image
import gc
import time
import struct

from typing import List, NamedTuple, Optional, Tuple

import cv2
import imageio

import numpy as np

import torch
import roma

from bop_toolkit_lib import inout

from utils.misc import array_to_tensor, tensor_to_array, tensors_to_arrays


from utils import (
    corresp_util,
    config_util,
    eval_errors,
    eval_util,
    feature_util,
    infer_pose_util,
    knn_util,
    misc as misc_util,
    pnp_util,
    projector_util,
    repre_util,
    vis_util,
    data_util,
    renderer_builder,
    json_util, 
    logging,
    misc,
    structs,
)

from utils.structs import AlignedBox2f, PinholePlaneCameraModel, CameraModel
from utils.misc import warp_depth_image, warp_image


logger: logging.Logger = logging.get_logger()


class InferOpts(NamedTuple):
    """Options that can be specified via the command line."""

    dataset_path: str
    mask_path: str
    object_path: str
    output_path: str
    cam_json_path: str
    use_meters: bool
    max_sym_disc_step: float = 0.01

    # Cropping options.
    crop: bool = True
    crop_rel_pad: float = 0.2
    crop_size: Tuple[int, int] = (420, 420)

    # Object instance options.
    use_detections: bool = True
    num_preds_factor: float = 1.0
    min_visibility: float = 0.1

    # Feature extraction options.
    extractor_name: str = "dinov2_vitl14"
    grid_cell_size: float = 1.0
    max_num_queries: int = 1000000

    # Feature matching options.
    match_template_type: str = "tfidf"
    match_top_n_templates: int = 5
    match_feat_matching_type: str = "cyclic_buddies"
    match_top_k_buddies: int = 300

    # PnP options.
    pnp_type: str = "opencv"
    pnp_ransac_iter: int = 1000
    pnp_required_ransac_conf: float = 0.99
    pnp_inlier_thresh: float = 10.0
    pnp_refine_lm: bool = True

    final_pose_type: str = "best_coarse"

    # Other options.
    save_estimates: bool = True
    vis_results: bool = True
    vis_corresp_top_n: int = 100
    vis_feat_map: bool = True
    vis_for_paper: bool = True
    debug: bool = True


def robustlosstorch(x: torch.Tensor, a=-5, c=0.5):
    """
    Args:
        x (torch.Tensor): Nxd tensor
    
    Returns:
        loss (torch.Tensor): N tensor
    """
    return torch.sum((abs(a - 2) / a) * (((x / c) ** 2 / abs(a - 2) + 1) ** (a / 2) - 1), dim=-1)


def descriptor_from_pose(
    q: torch.Tensor, t: torch.Tensor, camera: CameraModel, patchvtxs: torch.Tensor,
    featmap: torch.Tensor, patchsize: int, device=None
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    T = roma.RigidUnitQuat(q, t)
    camf = torch.tensor(camera.f).float().to(device)
    camc = torch.tensor(camera.c).float().to(device)
    projected = camera.project(T[None].apply(patchvtxs)) * camf + camc
    patchproj = projected / patchsize
    projfeatures = feature_util.sample_feature_map_at_points(featmap, patchproj, featmap.shape[1:])
    return projfeatures


def featuremetric_loss(
    q: torch.Tensor, t: torch.Tensor, camera: CameraModel, patchdescs: torch.Tensor,
    patchvtxs: torch.Tensor, featmap: torch.Tensor, patchsize: int, a: float=-5, c: float=0.5,
    device=None
):
    """
    Args:
        q: quaternion input
        t: translation input
    """
    # need: projected features from every patch in query rgb image
    # template descriptor + 3d point from patches inside mask
    # coarse R and t, need to parametrize R as quaternion or axangle
    # tfm is 6d [*axangle, *t]
    projfeatures = descriptor_from_pose(q, t, camera, patchvtxs, featmap, patchsize, device=device)
    featurediff = patchdescs - projfeatures
    return robustlosstorch(featurediff, a=a, c=c)


def featuremetric_cost(
    q: torch.Tensor, t: torch.Tensor, camera: CameraModel, patchdescs: torch.Tensor,
    patchvtxs: torch.Tensor, featmap: torch.Tensor, patchsize: int, a: float=-5, c: float=0.5,
    device=None
):
    """
    Args:
        q: quaternion input
        t: translation input
    """
    return torch.sum(featuremetric_loss(q, t, camera, patchdescs, patchvtxs, featmap, patchsize, a=a, c=c, device=device))


def infer(opts: InferOpts) -> None:
    dataset_path = Path(opts.dataset_path)
    mask_path = Path(opts.mask_path)

    # Prepare a logger and a timer.
    logger = logging.get_logger(level=logging.INFO if opts.debug else logging.WARNING)
    timer = misc_util.Timer(enabled=opts.debug)
    timer.start()

    # Prepare feature extractor.
    extractor = feature_util.make_feature_extractor(opts.extractor_name)
    # Prepare a device.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor.to(device)

    # Create a renderer.
    renderer_type = renderer_builder.RendererType.PYRENDER_RASTERIZER
    renderer = renderer_builder.build(renderer_type=renderer_type, model_path=opts.object_path)

    timer.elapsed("Time for setting up the stage")

    timer.start()

    # The output folder is named with slugified dataset path.
    output_dir = Path(opts.output_path, "inference")
    os.makedirs(output_dir, exist_ok=True)

    # Save parameters to a file.
    config_path = os.path.join(output_dir, "config.json")
    json_util.save_json(config_path, opts)

    # Create a pose evaluator.
    pose_evaluator = eval_util.EvaluatorPose([0])

    # Load the object representation.
    # logger.info(
    #     f"Loading representation for object {0} from dataset {opts.object_dataset}..."
    # )
    base_repre_dir = Path(opts.output_path, "object_repre")
    repre_dir = base_repre_dir
    repre = repre_util.load_object_repre(
        repre_dir=repre_dir,
        tensor_device=device,
    )

    logger.info("Object representation loaded.")
    repre_np = repre_util.convert_object_repre_to_numpy(repre)

    # Build a kNN index from object feature vectors.
    visual_words_knn_index = None
    if opts.match_template_type == "tfidf":
        visual_words_knn_index = knn_util.KNN(
            k=repre.template_desc_opts.tfidf_knn_k,
            metric=repre.template_desc_opts.tfidf_knn_metric
        )
        visual_words_knn_index.fit(repre.feat_cluster_centroids)

    # Build per-template KNN index with features from that template.
    template_knn_indices = []
    if opts.match_feat_matching_type == "cyclic_buddies":
        logger.info("Building per-template KNN indices...")
        for template_id in range(len(repre.template_cameras_cam_from_model)):
            logger.info(f"Building KNN index for template {template_id}...")
            tpl_feat_mask = repre.feat_to_template_ids == template_id
            tpl_feat_ids = torch.nonzero(tpl_feat_mask).flatten()

            template_feats = repre.feat_vectors[tpl_feat_ids]

            # Build knn index for object features.
            template_knn_index = knn_util.KNN(k=1, metric="l2")
            template_knn_index.fit(template_feats.cpu())
            template_knn_indices.append(template_knn_index)
        logger.info("Per-template KNN indices built.")

    logging.log_heading(
        logger,
        f"Object: {0}, vertices: {len(repre.vertices)}",
        style=logging.WHITE_BOLD,
    )

    # Get the object mesh and meta information.
    model_path = opts.object_path
    object_mesh = inout.load_ply(model_path)

    max_vertices = 1000
    subsampled_vertices = np.random.permutation(object_mesh["pts"])[:max_vertices]

    timer.elapsed("Time for preparing object data")
    
    imgpaths = sorted(list(dataset_path.glob("*.jpg")) + list(dataset_path.glob("*.png")))
    maskpaths = sorted(list(mask_path.glob("*.png")))

    # Perform inference on each selected image.
    for i, imgpath in enumerate(imgpaths):
        timer.start()

        # Camera parameters.
        # transform is from GT, can we just leave as identity?
        with open(Path(opts.cam_json_path), "r") as f:
            camjson = json.load(f)
        orig_camera_c2w = PinholePlaneCameraModel(
            camjson["width"], camjson["height"],
            (camjson["fx"], camjson["fy"]), (camjson["cx"], camjson["cy"])
        )
        orig_image_size = (
            orig_camera_c2w.width,
            orig_camera_c2w.height,
        )

        # Generate grid points at which to sample the feature vectors.
        if opts.crop:
            grid_size = opts.crop_size
        else:
            grid_size = orig_image_size
        grid_points = feature_util.generate_grid_points(
            grid_size=grid_size,
            cell_size=opts.grid_cell_size,
        )
        grid_points = grid_points.to(device)

        timer.elapsed("Time for preparing image data")

        # Estimate pose for each object instance.
        times = {}

        # Get the input image.
        orig_image_np_hwc = np.array(Image.open(imgpath)) / 255.0

        # Get the modal mask and amodal bounding box of the instance.
        # binary mask
        orig_mask_modal = np.array(Image.open(maskpaths[i]).convert("L")) / 255.0
        sumvert = np.sum(orig_mask_modal, axis=0)
        left = np.where(sumvert > 0)[0][0]
        right = np.where(sumvert > 0)[0][-1]
        sumhor = np.sum(orig_mask_modal, axis=1)
        bottom = np.where(sumhor > 0)[0][0]
        top = np.where(sumhor > 0)[0][-1]
        # bounding box of mask
        orig_box_amodal = AlignedBox2f(
            left=left,
            top=top,
            right=right,
            bottom=bottom,
        )
        timer.start()

        # Optional cropping.
        if not opts.crop:
            camera_c2w = orig_camera_c2w
            image_np_hwc = orig_image_np_hwc
            mask_modal = orig_mask_modal
            box_amodal = orig_box_amodal
        else:
            # Get box for cropping.
            crop_box = misc_util.calc_crop_box(
                box=orig_box_amodal,
                make_square=True,
            )

            # Construct a virtual camera focused on the crop.
            crop_camera_model_c2w = misc_util.construct_crop_camera(
                box=crop_box,
                camera_model_c2w=orig_camera_c2w,
                viewport_size=opts.crop_size,
                viewport_rel_pad=opts.crop_rel_pad,
            )

            # Map images to the virtual camera.
            interpolation = (
                cv2.INTER_AREA
                if crop_box.width >= crop_camera_model_c2w.width
                else cv2.INTER_LINEAR
            )
            image_np_hwc = warp_image(
                src_camera=orig_camera_c2w,
                dst_camera=crop_camera_model_c2w,
                src_image=orig_image_np_hwc,
                interpolation=interpolation,
            )
            mask_modal = warp_image(
                src_camera=orig_camera_c2w,
                dst_camera=crop_camera_model_c2w,
                src_image=orig_mask_modal,
                interpolation=cv2.INTER_NEAREST,
            )

            # Recalculate the object bounding box (it changed if we constructed the virtual camera).
            ys, xs = mask_modal.nonzero()
            box = np.array(misc_util.calc_2d_box(xs, ys))
            box_amodal = AlignedBox2f(
                left=box[0],
                top=box[1],
                right=box[2],
                bottom=box[3],
            )

            # The virtual camera is becoming the main camera.
            camera_c2w = crop_camera_model_c2w

        times["prep"] = timer.elapsed("Time for preparation")
        timer.start()

        # Extract feature map from the crop.
        image_tensor_chw = array_to_tensor(image_np_hwc).to(torch.float32).permute(2, 0, 1).to(device)
        image_tensor_bchw = image_tensor_chw.unsqueeze(0)
        # BxDxHxW
        extractor_output = extractor(image_tensor_bchw)
        feature_map_chw = extractor_output["feature_maps"][0]

        times["feat_extract"] = timer.elapsed("Time for feature extraction")
        timer.start()

        # Keep only points inside the object mask.
        mask_modal_tensor = array_to_tensor(mask_modal).to(device)
        query_points = feature_util.filter_points_by_mask(
            grid_points, mask_modal_tensor
        )

        # Subsample query points if we have too many.
        if query_points.shape[0] > opts.max_num_queries:
            perm = torch.randperm(query_points.shape[0])
            query_points = query_points[perm[: opts.max_num_queries]]
            msg = (
                "Randomly sumbsampled queries "
                f"({perm.shape[0]} -> {query_points.shape[0]}))"
            )
            logging.log_heading(logger, msg, style=logging.RED_BOLD)

        # Extract features at the selected points, of shape (num_points, feat_dims).
        timer.start()
        query_features = feature_util.sample_feature_map_at_points(
            feature_map_chw=feature_map_chw,
            points=query_points,
            image_size=(image_np_hwc.shape[1], image_np_hwc.shape[0]),
        ).contiguous()

        times["grid_sample"] = timer.elapsed("Time for grid sample")
        timer.start()
        # Potentially project features to a PCA space.
        if (
            query_features.shape[1] != repre.feat_vectors.shape[1]
            and len(repre.feat_raw_projectors) != 0
        ):
            query_features_proj = projector_util.project_features(
                feat_vectors=query_features,
                projectors=repre.feat_raw_projectors,
            ).contiguous()

            _c, _h, _w = feature_map_chw.shape
            feature_map_chw_proj = (
                projector_util.project_features(
                    feat_vectors=feature_map_chw.permute(1, 2, 0).view(-1, _c),
                    projectors=repre.feat_raw_projectors,
                )
                .view(_h, _w, -1)
                .permute(2, 0, 1)
            )
        else:
            query_features_proj = query_features
            feature_map_chw_proj = feature_map_chw

        times["proj"] = timer.elapsed("Time for projection")
        timer.start()

        # Establish 2D-3D correspondences.
        corresp = []
        if len(query_points) != 0:
            corresp = corresp_util.establish_correspondences(
                query_points=query_points,
                query_features=query_features_proj,
                object_repre=repre,
                template_matching_type=opts.match_template_type,
                template_knn_indices=template_knn_indices,
                feat_matching_type=opts.match_feat_matching_type,
                top_n_templates=opts.match_top_n_templates,
                top_k_buddies=opts.match_top_k_buddies,
                visual_words_knn_index=visual_words_knn_index,
                debug=opts.debug,
            )

        times["corresp"] = timer.elapsed("Time for corresp")
        timer.start()

        logger.info(
            f"Number of corresp: {[len(c['coord_2d']) for c in corresp]}"
        )

        # Estimate coarse poses from corespondences.
        coarse_poses = []
        for corresp_id, corresp_curr in enumerate(corresp):
            # We need at least 3 correspondences for P3P.
            num_corresp = len(corresp_curr["coord_2d"])
            if num_corresp < 6:
                logger.info(f"Only {num_corresp} correspondences, skipping.")
                continue

            (
                coarse_pose_success,
                R_m2c_coarse,
                t_m2c_coarse,
                inliers_coarse,
                quality_coarse,
            ) = pnp_util.estimate_pose(
                corresp=corresp_curr,
                camera_c2w=camera_c2w,
                pnp_type=opts.pnp_type,
                pnp_ransac_iter=opts.pnp_ransac_iter,
                pnp_inlier_thresh=opts.pnp_inlier_thresh,
                pnp_required_ransac_conf=opts.pnp_required_ransac_conf,
                pnp_refine_lm=opts.pnp_refine_lm,
            )

            logger.info(
                f"Quality of coarse pose {corresp_id}: {quality_coarse}"
            )

            if coarse_pose_success:
                coarse_poses.append(
                    {
                        "type": "coarse",
                        "R_m2c": R_m2c_coarse,
                        "t_m2c": t_m2c_coarse,
                        "corresp_id": corresp_id,
                        "quality": quality_coarse,
                        "inliers": inliers_coarse,
                        # lol
                        "template_id": int(corresp_curr["template_id"].cpu().numpy()),
                    }
                )

        # Find the best coarse pose.
        best_coarse_quality = None
        best_coarse_pose_id = 0
        qualities = [p["quality"] for p in coarse_poses]
        sortidxs = np.argsort(qualities)[::-1]

        for coarse_pose_id, pose in enumerate(coarse_poses):
            if best_coarse_quality is None or pose["quality"] > best_coarse_quality:
                best_coarse_pose_id = coarse_pose_id
                best_coarse_quality = pose["quality"]

        times["pose_coarse"] = timer.elapsed("Time for coarse pose")

        timer.start()

        # Select the final pose estimate.
        final_poses = []
        
        if opts.final_pose_type in ["best_coarse",]:
            # If no successful coarse pose, continue.
            if len(coarse_poses) == 0:
                continue

            # Select the refined pose corresponding to the best coarse pose as the final pose.
            final_pose = None

            final_pose = coarse_poses[best_coarse_pose_id]

            if final_pose is not None:
                final_poses.append(final_pose)
        elif opts.final_pose_type in ["top5_coarse",]:
            if len(coarse_poses) == 0:
                continue
            topidxs = sortidxs[:5]
            for idx in topidxs:
                final_poses.append(coarse_poses[idx])
        else:
            raise ValueError(f"Unknown final pose type {opts.final_pose_type}")

        times["final_select"] = timer.elapsed("Time for selecting final pose")

        # Print summary.
        if len(final_poses) > 0:
            # Divide to the number of hypothesis because this is the real time needed per hypothesis.
            time_per_hypothesis = sum(times.values()) / len(final_poses)

            # pose_found = final_pose is not None
            logging.log_heading(
                logger,
                f"SUMMARY - success: {len(final_poses)}, time: {time_per_hypothesis:.4f}s",
                style=logging.WHITE_BOLD,
            )

        # Iterate over the final poses to collect visuals.
        for hypothesis_id, final_pose in enumerate(final_poses):
            # Visualizations and saving of results.
            vis_tiles = []

            # Increment hypothesis id by one for each found pose hypothesis.
            pose_m2w = None
            pose_m2w_coarse = None

            # Express the estimated pose as an m2w transformation.
            pose_est_m2c = structs.ObjectPose(
                R=final_pose["R_m2c"], t=final_pose["t_m2c"]
            )
            trans_c2w = camera_c2w.T_world_from_eye

            trans_m2w = trans_c2w.dot(misc.get_rigid_matrix(pose_est_m2c))
            pose_m2w_coarse = structs.ObjectPose(
                R=trans_m2w[:3, :3], t=trans_m2w[:3, 3:]
            )

            # featuremetric refinement
            template_id = final_pose["template_id"]
            feattempmask = repre.feat_to_template_ids == template_id
            # all p_i values
            allpi = repre.feat_vectors[feattempmask]
            # all x_i values
            allxi = repre.vertices[feattempmask]
            Fq = feature_map_chw_proj
            qtorch = roma.rotmat_to_unitquat(torch.tensor(pose_m2w_coarse.R)).float().to(device).requires_grad_()
            ttorch = torch.tensor(pose_m2w_coarse.t.reshape(-1)).float().to(device).requires_grad_()
            optimizer = torch.optim.Adam([qtorch, ttorch], lr=0.01)

            for _ in range(100):
                loss = featuremetric_cost(qtorch, ttorch, orig_camera_c2w, allpi, allxi, Fq, opts.grid_cell_size, device=device)
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    qtorch = roma.quat_normalize(qtorch)

            pose_m2w = structs.ObjectPose(
                R=roma.unitquat_to_rotmat(qtorch).detach().cpu().numpy(), t=ttorch.detach().cpu().numpy().reshape(3, 1)
            )

            # Get image for visualization.
            vis_base_image = (255 * image_np_hwc).astype(np.uint8)

            # Convert correspondences from tensors to numpy arrays.
            best_corresp_np = tensors_to_arrays(
                corresp[final_pose["corresp_id"]]
            )

            # IDs and scores of the matched templates.
            matched_template_ids = [c["template_id"] for c in corresp]
            matched_template_scores = [c["template_score"] for c in corresp]

            # Skip evaluation if there is no ground truth available, and only keep
            # the estimated poses.
            pose_eval_dict = None
            pose_eval_dict_coarse = None
            pose_eval_dict = pose_evaluator.update_without_anno(
                scene_id=0,
                im_id=i,
                inst_id=0,
                hypothesis_id=hypothesis_id,
                object_repre_vertices=tensor_to_array(repre.vertices),
                obj_lid=0,
                object_pose_m2w=pose_m2w,
                orig_camera_c2w=orig_camera_c2w,
                camera_c2w=orig_camera_c2w,
                time_per_inst=times,
                corresp=best_corresp_np,
                inlier_radius=(opts.pnp_inlier_thresh),
                img_path=imgpath,
                template_id=final_pose["template_id"],
                object_pose_m2w_coarse=pose_m2w_coarse,
            )

            # Optionally visualize the results.
            if opts.vis_results:
                # IDs and scores of the matched templates.
                matched_template_ids = [c["template_id"] for c in corresp]
                matched_template_scores = [c["template_score"] for c in corresp]

                timer.start()
                vis_tiles += vis_util.vis_inference_results(
                    base_image=vis_base_image,
                    object_repre=repre_np,
                    object_lid=0,
                    object_pose_m2w=pose_m2w, # pose_m2w,
                    object_pose_m2w_gt=None,
                    feature_map_chw=feature_map_chw,
                    feature_map_chw_proj=feature_map_chw_proj,
                    vis_feat_map=opts.vis_feat_map,
                    object_box=box_amodal.array_ltrb(),
                    object_mask=mask_modal,
                    camera_c2w=camera_c2w,
                    corresp=best_corresp_np,
                    matched_template_ids=matched_template_ids,
                    matched_template_scores=matched_template_scores,
                    best_template_ind=final_pose["corresp_id"],
                    renderer=renderer,
                    pose_eval_dict=pose_eval_dict,
                    corresp_top_n=opts.vis_corresp_top_n,
                    inlier_thresh=(opts.pnp_inlier_thresh),
                    object_pose_m2w_coarse=pose_m2w_coarse,
                    pose_eval_dict_coarse=pose_eval_dict_coarse,
                    obj_in_meters=opts.use_meters,
                    # For paper visualizations:
                    vis_for_paper=opts.vis_for_paper,
                    extractor=extractor,
                )
                timer.elapsed("Time for visualization")

            # Assemble visualization tiles to a grid and save it.
            if len(vis_tiles):
                if repre.feat_vis_projectors[0].pca.n_components == 12:
                    pca_tiles = np.vstack(vis_tiles[1:5])
                    vis_tiles = np.vstack([vis_tiles[0]] + vis_tiles[5:])
                    vis_grid = np.hstack([vis_tiles, pca_tiles])
                else:
                    vis_grid = np.vstack(vis_tiles)
                ext = ".png" if opts.vis_for_paper else ".jpg"
                vis_path = os.path.join(
                    output_dir,
                    f"image_{i}_{hypothesis_id}{ext}",
                )
                visgrid_img = Image.fromarray(vis_grid)
                visgrid_img.save(vis_path)
                # inout.save_im(vis_path, vis_grid)
                logger.info(f"Visualization saved to {vis_path}")

                # if opts.debug:
                #     pts_path = os.path.join(
                #         output_dir,
                #         f"{bop_chunk_id}_{bop_im_id}_{object_lid}_{inst_j}_{hypothesis_id}_vertice_error.ply",
                #     )
                #     vis_util.vis_pointcloud_error(
                #         repre_np,
                #         pose_m2w,
                #         object_pose_m2w_gt,
                #         camera_c2w,
                #         0,
                #         pts_path,
                #     )

    # Empty unused GPU cache variables.
    if device == "cuda":
        time_start = time.time()
        torch.cuda.empty_cache()
        gc.collect()
        time_end = time.time()
        logger.info(f"Garbage collection took {time_end - time_start} seconds.")

    # Save the pose estimates.
    if opts.save_estimates:
        results_path = os.path.join(output_dir, "estimated-poses.json")
        logger.info("Saving estimated poses to: {}".format(results_path))
        pose_evaluator.save_results_json(results_path)


def main() -> None:
    opts = config_util.load_opts_from_json_or_command_line(
        InferOpts
    )[0]
    infer(opts)


if __name__ == "__main__":
    main()

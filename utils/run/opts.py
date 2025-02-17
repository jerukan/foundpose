from typing import NamedTuple, Optional, Tuple

from utils import repre_util


class CommonOpts(NamedTuple):
    object_path: str
    output_path: str
    cam_json_path: str
    extractor_name: str = "dinov2_version=vitl14-reg_stride=14_facet=token_layer=18_logbin=0_norm=1"
    grid_cell_size: float = 14
    units: str = "m"
    device: str = "cuda"

    # Cropping options.
    crop: bool = True
    crop_rel_pad: float = 0.2
    crop_size: Tuple[int, int] = (420, 420)

    debug: bool = True


class GenTemplatesOpts(NamedTuple):
    light_intensity: float = 10.0

    # Viewpoint options.
    num_viewspheres: int = 1
    min_num_viewpoints: int = 57
    num_inplane_rotations: int = 14
    depth_range: Tuple[float, float] = (1.0, 3.0)
    images_per_view: int = 1

    # Mesh pre-processing options.
    max_num_triangles: int = 20000
    back_face_culling: bool = False
    texture_size: Tuple[int, int] = (1024, 1024)

    # Rendering options.
    ssaa_factor: float = 4.0
    background_type: str = "black"
    light_type: str = "multi_directional"

    # Other options.
    features_patch_size: int = 14
    save_templates: bool = True
    overwrite: bool = True


class GenRepreOpts(NamedTuple):
    # Feature PCA options.
    apply_pca: bool = True
    pca_components: int = 256
    pca_whiten: bool = False
    pca_max_samples_for_fitting: int = 100000

    # Feature clustering options.
    cluster_features: bool = True
    cluster_num: int = 2048

    # Template descriptor options.
    template_desc_opts: Optional[repre_util.TemplateDescOpts] = None

    # Other options.
    overwrite: bool = True


class InferOpts(NamedTuple):
    dataset_path: str
    mask_path: str
    max_sym_disc_step: float = 0.01

    # Object instance options.
    use_detections: bool = True
    num_preds_factor: float = 1.0
    min_visibility: float = 0.1

    # Feature extraction options.
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

    # Refinement options
    max_iters_refinement: int = 2000
    lr_refinement: float = 0.01

    # Other options.
    save_estimates: bool = True
    vis_results: bool = True
    vis_corresp_top_n: int = 100
    vis_feat_map: bool = True
    vis_for_paper: bool = True

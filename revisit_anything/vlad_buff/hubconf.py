dependencies = ["torch"]

import torch
from vpr_model import VPRModel
from models.backbones.dinov2 import DINOV2_ARCHS
import argparse
from collections import OrderedDict

VAL_DATASETS = [
    "MSLS",
    "MSLS_Test",
    "pitts30k_test",
    "pitts250k_test",
    "Nordland",
    "SPED",
    "pitts30k_val",
    "st_lucia",
    "tokyo247",
    "amstertime",
    "baidu",
    "sfsm",
]


def parse_args(antiburst=False, nv_pca=None, wpca=False, num_pcs=8192):
    parser = argparse.ArgumentParser(
        description="Eval VPR model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset_name", type=str, default="gsv_cities", help="Dataset"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./logs/lightning_logs/",
        help="name of directory on which to save the logs, under logs/save_dir",
    )
    parser.add_argument(
        "--expName", default="0", help="Unique string for an experiment"
    )
    parser.add_argument(
        "--batch_size", type=int, default=200, help="Batch size for the data module"
    )
    parser.add_argument("--img_per_place", type=int, default=4, help="Images per place")
    parser.add_argument(
        "--min_img_per_place", type=int, default=4, help="min_img_per_place"
    )
    parser.add_argument(
        "--shuffle_all",
        type=bool,
        default=False,
        help="Shuffle all images or keep shuffling in-city only",
    )
    parser.add_argument(
        "--random_sample_from_each_place",
        type=bool,
        default=True,
        help="Random sample from each place",
    )
    parser.add_argument(
        "--resize",
        type=int,
        nargs=2,
        default=[322, 322],
        help="Resizing shape for images (HxW).",
    )

    parser.add_argument("--num_workers", type=int, default=20, help="Number of workers")
    parser.add_argument(
        "--show_data_stats", type=bool, default=True, help="Show data statistics"
    )

    parser.add_argument(
        "--backbone",
        type=str,
        default="dinov2_vitb14",
        choices=["dinov2_vitb14", "resnet"],
        help="Backbone architecture",
    )
    parser.add_argument(
        "--num_trainable_blocks", type=int, default=4, help="Trainable blocks"
    )
    parser.add_argument("--norm_layer", type=bool, default=True, help="Use norm layer")

    # Arguments for helper.py (aggregation configuration)
    parser.add_argument(
        "--aggregation",
        type=str,
        default="NETVLAD",
        choices=["SALAD", "cosplace", "gem", "convap", "mixvpr", "NETVLAD"],
    )
    # Cosplace
    parser.add_argument("--in_dim", type=int, default=2048, help="In dim for cosplace")
    parser.add_argument("--out_dim", type=int, default=512, help="In dim for cosplace")
    # gem
    parser.add_argument("--p", type=int, default=3, help="power for gem")
    # convap
    parser.add_argument(
        "--in_channels", type=int, default=2048, help="in_channels for convap"
    )
    # mixvpr
    parser.add_argument(
        "--out_channels", type=int, default=512, help="out_channels for mixvpr"
    )
    parser.add_argument("--in_h", type=int, default=20, help="in_h for mixvpr")
    parser.add_argument("--in_w", type=int, default=20, help="in_w for mixvpr")
    parser.add_argument("--mix_depth", type=int, default=1, help="mix depth for mixvpr")
    # salad
    parser.add_argument(
        "--storeSOTL",
        action="store_true",
        help="Store the soft_assign (optimal transport layer) and vlad",
    )
    parser.add_argument(
        "--num_channels", type=int, default=768, help="num channels for salad"
    )
    parser.add_argument(
        "--num_clusters", type=int, default=64, help="num clusters for salad"
    )
    parser.add_argument(
        "--cluster_dim", type=int, default=128, help="cluster_dim for salad"
    )
    parser.add_argument(
        "--token_dim", type=int, default=256, help="token_dim for salad"
    )
    parser.add_argument(
        "--reduce_feature_dims",
        type=bool,
        default=True,
        help="Perform dimensionlity reduction for feature",
    )

    parser.add_argument(
        "--reduce_token_dims",
        type=bool,
        default=True,
        help="Perform dimensionlity reduction for token",
    )
    # netvlad
    parser.add_argument(
        "--l2",
        type=str,
        default="none",
        choices=["before_pool", "after_pool", "onlyFlatten", "none"],
        help="When (and if) to apply the l2 norm with shallow aggregation layers",
    )
    parser.add_argument(
        "--forLoopAlt",
        type=bool,
        default=True,
        help="if True, it will not use For loop to calculate VLAD ",
    )
    parser.add_argument(
        "--fc_output_dim",
        type=int,
        default=512,
        help="Output dimension of final fully connected layer",
    )
    parser.add_argument("--dim", type=int, default=768, help="dim for netvlad")
    parser.add_argument(
        "--clusters_num", type=int, default=64, help="clusters_num for netvlad"
    )
    parser.add_argument(
        "--initialize_clusters",
        type=bool,
        default=False,
        help="Initialize the cluster for VLAD layer",
    )
    parser.add_argument(
        "--useFC",
        action="store_true",
        help="Add fully connected layer after VLAD layer",
    )
    parser.add_argument(
        "--nv_pca",
        type=int,
        default=nv_pca,
        help="Use PCA before clustering and nv aggregation.",
    )
    parser.add_argument(
        "--nv_pca_randinit",
        action="store_true",
        help="Initialize randomly instead of pca",
    )
    parser.add_argument(
        "--nv_pca_alt", action="store_true", help="use fc layer instead of pca"
    )
    parser.add_argument(
        "--nv_pca_alt_mlp",
        action="store_true",
        help="use 2-fc layer mlp layer instead of pca / pca_alt",
    )

    # ab params
    parser.add_argument(
        "--infer_batch_size",
        type=int,
        default=16,
        help="Batch size for inference (validating and testing)",
    )
    parser.add_argument(
        "--storeSAB",
        action="store_true",
        help="Store the soft_assign, selfDis, w_burst, new soft_assign, vlad",
    )
    parser.add_argument(
        "--antiburst",
        action="store_true",
        default=antiburst,
        help="use self sim + sigmoid to remove burstiness",
    )
    parser.add_argument("--ab_w", type=float, default=8.0, help="")
    parser.add_argument("--ab_b", type=float, default=7.0, help="")
    parser.add_argument("--ab_p", type=float, default=1.0, help="")
    parser.add_argument(
        "--ab_gen", type=int, help="generates thresholds from soft_assign"
    )
    parser.add_argument("--ab_relu", action="store_true", help="")
    parser.add_argument(
        "--ab_soft",
        action="store_true",
        help="softmax instead of sigmoid before summing",
    )
    parser.add_argument("--ab_inv", action="store_true", help="")
    parser.add_argument("--ab_t", type=float, help="thresh for relu")
    parser.add_argument("--ab_testOnly", action="store_true", help="")
    parser.add_argument("--ab_allFreezeButAb", action="store_true", help="")
    parser.add_argument(
        "--ab_fixed",
        action="store_true",
        help="ab params are init but arent nn.Parameter",
    )
    parser.add_argument(
        "--ab_kp", type=int, help="num middle dim for fc-relu-fc weight per pixel"
    )
    parser.add_argument(
        "--ab_wOnly", action="store_true", help="train w, freeze b and p as init"
    )

    parser.add_argument(
        "--device", type=str, default="cuda", choices=["cuda", "cpu"], help="_"
    )
    parser.add_argument(
        "--resume_train",
        type=str,
        default=None,
        help="path to checkpoint to resume, e.g. logs/.../last_checkpoint.pth",
    )
    parser.add_argument(
        "--ckpt_state_dict",
        type=bool,
        default=True,
        help="Use checkpoint state dictionary",
    )
    # Datasets parameters
    parser.add_argument(
        "--val_datasets",
        nargs="+",
        default=VAL_DATASETS,
        help="Validation datasets to use",
        choices=VAL_DATASETS,
    )
    parser.add_argument(
        "--pl_seed",
        type=bool,
        default=True,
        help="Use pytorch_lightning (pl) seed",
    )
    parser.add_argument(
        "--wpca",
        action="store_true",
        default=wpca,
        help="Use post pool WPCA layer",
    )
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument(
        "--num_pcs", type=int, default=num_pcs, help="Use post pool PCA."
    )
    parser.add_argument(
        "--no_wandb",
        type=bool,
        default=True,
        help="Dont use wandb",
    )
    parser.add_argument(
        "--store_eval_output", action="store_true", help="store eval output"
    )

    args = parser.parse_args()

    # Parse image size
    if args.resize:
        if len(args.resize) == 1:
            args.resize = (args.resize[0], args.resize[0])
        elif len(args.resize) == 2:
            args.resize = tuple(args.resize)
        else:
            raise ValueError("Invalid image size, must be int, tuple or None")

        args.resize = tuple(map(int, args.resize))

    return args


def replace_key(k, num):
    for old, new in {"WPCA_" + str(num) + ".": "WPCA."}.items():
        if old in k:
            k = k.replace(old, new)
    return k


def vlad_buff(
    pretrained=True, antiburst=True, nv_pca=None, wpca=True, num_pcs=8192
) -> torch.nn.Module:
    """Return a VLAD-BuFF model.
    Args:
        pretrained (bool): Whether to load pretrained weights.
        antiburst (bool): Whether to apply antiburst mechanism.
        nv_pca (int): Use PCA before clustering and nv aggregation.
        wpca (bool): Whether to Use post pool WPCA layer.
        num_pcs (int): Use post pool PCA.
    Return:
        model (torch.nn.Module): the model.
    """
    args = parse_args(antiburst=antiburst, nv_pca=nv_pca, wpca=wpca, num_pcs=num_pcs)

    assert args.aggregation == "NETVLAD"
    assert args.antiburst == True
    assert args.wpca == True

    agg_args = args

    assert (
        args.backbone in DINOV2_ARCHS.keys()
    ), f"Parameter `backbone` is set to {args.backbone} but it must be one of {list(DINOV2_ARCHS.keys())}"
    assert (
        not pretrained or args.backbone == "dinov2_vitb14"
    ), f"Parameter `pretrained` can only be set to True if backbone is 'dinov2_vitb14', but it is set to {args.backbone}"

    backbone_config = {
        "num_trainable_blocks": args.num_trainable_blocks,
        "return_token": False,
        "norm_layer": args.norm_layer,
    }

    model = VPRModel(
        backbone_arch=args.backbone,
        backbone_config=backbone_config,
        agg_arch=args.aggregation,
        agg_config=agg_args,
        args=args,
    )

    if args.nv_pca is None:
        assert args.num_pcs == 8192
        checkpoint = torch.hub.load_state_dict_from_url(
            f"https://github.com/ahmedest61/VLAD-BuFF/releases/download/v1.0.0/dnv2_NV_AB_wpca{args.num_pcs}_last.ckpt"
        )
    else:
        assert args.num_pcs == 4096
        checkpoint = torch.hub.load_state_dict_from_url(
            f"https://github.com/ahmedest61/VLAD-BuFF/releases/download/v1.0.0/dnv2_NV_{args.nv_pca}PCA_AB_wpca{args.num_pcs}_last.ckpt"
        )

    checkpoint["state_dict"] = OrderedDict(
        {replace_key(k, args.num_pcs): v for k, v in checkpoint["state_dict"].items()}
    )
    model_state_dict = model.state_dict()
    # Filter out keys that are not in the model's current state dict
    checkpoint["state_dict"] = {
        k: v for k, v in checkpoint["state_dict"].items() if k in model_state_dict
    }
    model.load_state_dict(checkpoint["state_dict"])
    
    return model

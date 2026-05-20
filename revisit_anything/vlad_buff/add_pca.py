import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
import torchvision.transforms as T
from tqdm import tqdm
import argparse
from os.path import join, isfile
from vpr_model import VPRModel
from utils.validation import get_validation_recalls
from scipy.sparse.linalg import eigs
import torch.nn as nn
import time
from sklearn.decomposition import PCA
import torch.nn.functional as F

# Dataloader
from dataloaders.GSVCitiesDataset import GSVCitiesDataset

IMAGENET_MEAN_STD = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}

TRAIN_CITIES = [
    "Bangkok",
    "BuenosAires",
    "LosAngeles",
    "MexicoCity",
    "OSL",
    "Rome",
    "Barcelona",
    "Chicago",
    "Madrid",
    "Miami",
    "Phoenix",
    "TRT",
    "Boston",
    "Lisbon",
    "Medellin",
    "Minneapolis",
    "PRG",
    "WashingtonDC",
    "Brussels",
    "London",
    "Melbourne",
    "Osaka",
    "PRS",
]


class Flatten(nn.Module):
    def forward(self, input_data):
        return input_data.view(input_data.size(0), -1)


class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, input_data):
        return F.normalize(input_data, p=2, dim=self.dim)


def pca(x: np.ndarray, num_pcs=None, subtract_mean=True):
    # translated from MATLAB:
    # - https://github.com/Relja/relja_matlab/blob/master/relja_PCA.m
    # - https://github.com/Relja/netvlad/blob/master/addPCA.m

    # assumes x = nvectors x ndims
    x = x.T  # matlab code is ndims x nvectors, so transpose

    n_points = x.shape[1]
    n_dims = x.shape[0]

    if num_pcs is None:
        num_pcs = n_dims

    print(
        "PCA for {} points of dimension {} to PCA dimension {}".format(
            n_points, n_dims, num_pcs
        )
    )

    if subtract_mean:
        # Subtract mean
        mu = np.mean(x, axis=1)
        x = (x.T - mu).T
    else:
        mu = np.zeros(n_dims)

    assert num_pcs < n_dims

    if n_dims <= n_points:
        do_dual = False
        # x2 = dims * dims
        x2 = np.matmul(x, x.T) / (n_points - 1)
    else:
        do_dual = True
        # x2 = vectors * vectors
        x2 = np.matmul(x.T, x) / (n_points - 1)

    if num_pcs < x2.shape[0]:
        print("Compute {} eigenvectors".format(num_pcs))
        lams, u = eigs(x2, num_pcs)
    else:
        print("Compute eigenvectors")
        lams, u = np.linalg.eig(x2)

    assert np.all(np.isreal(lams)) and np.all(np.isreal(u))
    lams = np.real(lams)
    u = np.real(u)

    sort_indices = np.argsort(lams)[::-1]
    lams = lams[sort_indices]
    u = u[:, sort_indices]

    if do_dual:
        # U = x * ( U * diag(1./sqrt(max(lams,1e-9))) / sqrt(nPoints-1) );
        diag = np.diag(1.0 / np.sqrt(np.maximum(lams, 1e-9)))
        utimesdiag = np.matmul(u, diag)
        u = np.matmul(x, utimesdiag / np.sqrt(n_points - 1))

    return u, lams, mu


def parse_args():
    parser = argparse.ArgumentParser(
        description="Eval VPR model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset_name", type=str, default="gsv_cities", help="Dataset"
    )
    parser.add_argument(
        "--expName", default="0", help="Unique string for an experiment"
    )
    parser.add_argument(
        "--batch_size", type=int, default=250, help="Batch size for the data module"
    )
    parser.add_argument("--img_per_place", type=int, default=1, help="Images per place")
    parser.add_argument(
        "--min_img_per_place", type=int, default=1, help="min_img_per_place"
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
    #    parser.add_argument("--image_size", type=int, nargs=2, default=[224, 224], help="Image size (width, height)")
    parser.add_argument(
        "--resize",
        type=int,
        nargs=2,
        default=[224, 224],
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
    parser.add_argument(
        "--aggregation",
        type=str,
        default="SALAD",
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
        action="store_true",
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
        "--nv_pca", type=int, help="Use PCA before clustering and nv aggregation."
    )
    parser.add_argument(
        "--wpca",
        action="store_true",
        help="Use post pool WPCA layer / cannot be used during training",
    )
    parser.add_argument(
        "--num_pcs",
        type=int,
        nargs="+",  # '+' allows for one or more inputs
        default=[8192],
        help="Use post pool PCA. Can specify one, two, or three values.",
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
        help="use args sim + sigmoid to remove burstiness",
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
    #    parser.add_argument("--ckpt_path", type=str, required=True, default=None, help="Path to the checkpoint")
    parser.add_argument(
        "--resume_train",
        type=str,
        required=True,
        default=None,
        help="path to checkpoint to resume, e.g. logs/.../last_checkpoint.pth",
    )
    parser.add_argument(
        "--ckpt_state_dict", action="store_true", help="Use checkpoint state dictionary"
    )

    parser.add_argument(
        "--pl_seed",
        type=bool,
        default=True,
        help="Use pytorch_lightning (pl) seed",
    )
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument(
        "--cities",
        nargs="+",
        default=TRAIN_CITIES,
        help="Cities to use for PCA training",
        choices=TRAIN_CITIES,
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


if __name__ == "__main__":
    args = parse_args()

    if args.pl_seed:
        pl.seed_everything(seed=int(args.seed), workers=True)

    if "netvlad" in args.aggregation.lower():
        agg_config = args
        useToken = False
    elif "salad" in args.aggregation.lower():
        agg_config = {
            "num_channels": args.num_channels,
            "num_clusters": args.num_clusters,
            "cluster_dim": args.cluster_dim,
            "token_dim": args.token_dim,
            "expName": args.expName,
            "reduce_feature_dims": args.reduce_feature_dims,
            "reduce_token_dims": args.reduce_token_dims,
        }
        useToken = True

    model = VPRModel(
        backbone_arch=args.backbone,
        backbone_config={
            "num_trainable_blocks": args.num_trainable_blocks,
            "return_token": useToken,
            "norm_layer": args.norm_layer,
        },
        agg_arch=args.aggregation,
        agg_config=agg_config,
        args=args,
    )

    print(f"<=======backbone_arch: {args.backbone}========>")
    print(f"<=======aggregation: {args.aggregation}========>")
    print(f"<=======args: {args}========>")
    print(f"<=======model: {model}========>")

    print("===> Building model")

    if "netvlad" in args.aggregation.lower():
        encoder_dim = args.dim

    if args.resume_train:  # must resume for PCA
        if isfile(args.resume_train):
            print("=> loading checkpoint '{}'".format(args.resume_train))
            checkpoint = torch.load(
                args.resume_train, map_location=lambda storage, loc: storage
            )
            if "netvlad" in args.aggregation.lower():
                assert args.clusters_num == int(
                    checkpoint["state_dict"]["aggregator.centroids"].shape[0]
                )

            if args.ckpt_state_dict:
                model.load_state_dict(checkpoint["state_dict"])
                args.start_epoch = checkpoint["epoch"]
            else:
                model.load_state_dict(checkpoint)
            print(
                "=> loaded checkpoint '{}'".format(
                    args.resume_train,
                )
            )
        else:
            raise FileNotFoundError(
                "=> no checkpoint found at '{}'".format(args.resume_train)
            )
    else:
        raise ValueError("Need an existing checkpoint in order to run PCA")

    model = model.to(args.device)
    if "netvlad" in args.aggregation.lower():
        pool_size = encoder_dim
        # check preNV pca is used
        if args.nv_pca is not None:
            pool_size = args.nv_pca

        pool_size *= args.clusters_num
    elif "salad" in args.aggregation.lower():
        pool_size = args.num_clusters * args.cluster_dim + args.token_dim

    train_transform = T.Compose(
        [
            T.Resize(args.resize, interpolation=T.InterpolationMode.BILINEAR),
            T.RandAugment(num_ops=3, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN_STD["mean"], std=IMAGENET_MEAN_STD["std"]),
        ]
    )

    train_dataset = GSVCitiesDataset(
        cities=args.cities,
        img_per_place=args.img_per_place,
        min_img_per_place=args.min_img_per_place,
        random_sample_from_each_place=args.random_sample_from_each_place,
        transform=train_transform,
    )

    total_cluster_ds_images = train_dataset.total_nb_images
    num_pca_train_images = len(train_dataset)

    print("===> Loading PCA dataset(s)")

    nFeatures = 10000

    if nFeatures > num_pca_train_images:
        nFeatures = num_pca_train_images

    sampler = SubsetRandomSampler(
        np.random.choice(num_pca_train_images, nFeatures, replace=False)
    )

    train_loader_config = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "drop_last": False,
        "pin_memory": True,
        "shuffle": args.shuffle_all,
        "sampler": sampler,
    }

    data_loader = DataLoader(dataset=train_dataset, **train_loader_config)

    print("===> Do inference to extract features and save them.")
    model.eval()
    with torch.no_grad():
        tqdm.write("====> Extracting Features")

        dbFeat = np.empty((len(data_loader.sampler), pool_size))
        print("Compute", len(dbFeat), "features")
        total_input = 0
        for iteration, (input_data, indices) in enumerate(tqdm(data_loader)):
            # In GSVCities, we return a place, which is a Tesor of K images (K=self.img_per_place)
            # this will return a Tensor of shape [K, channels, height, width]. This needs to be taken into account
            # in the Dataloader (which will yield batches of shape [BS, K, channels, height, width])
            input_data = input_data.view(
                -1, *input_data.shape[2:]
            )  # Reshape input [BS*K, channels, height, width]
            input_data = input_data.to(args.device)
            image_encoding = model.backbone(input_data)
            if args.useFC:
                vlad_encoding = model.aggLayer(image_encoding)
            else:
                vlad_encoding = model.aggregator(image_encoding)
            out_vectors = vlad_encoding.detach().cpu().numpy()
            # this allows for randomly shuffled inputs
            for idx, out_vector in enumerate(out_vectors):
                dbFeat[iteration * data_loader.batch_size + idx, :] = out_vector

            del input_data, image_encoding, vlad_encoding

    print(args.expName)
    print(f"===> Compute PCA for {args.num_pcs}, takes a while")
    # Ensure args.num_pcs is sorted to efficiently use PCA components
    args.num_pcs.sort()
    # Compute PCA for the maximum number of components specified
    max_num_pcs = max(args.num_pcs)

    # Start the timer before PCA computation
    start_time = time.time()

    print("NOTE: This process may take some time depending on the size of the data.")
    u, lams, mu = pca(dbFeat, num_pcs=max_num_pcs)

    # End the timer after PCA computation
    end_time = time.time()
    print(f"PCA computation took {end_time - start_time:.2f} seconds.")

    for num_pc in args.num_pcs:
        start_time = time.time()
        print(f"===> Processing PCA with {num_pc} components")
        save_path = args.resume_train.replace("last.ckpt", f"wpca{num_pc}_last.ckpt")
        # Slice the PCA components and explained variances for the current num_pc
        current_u = u[:, :num_pc]
        current_lams = lams[:num_pc]

        # Perform the whitening manually
        current_u_whitened = np.matmul(
            current_u, np.diag(1.0 / np.sqrt(current_lams + 1e-9))
        )

        # Compute the bias term (mean adjustment for whitening)
        utmu = np.matmul(current_u_whitened.T, mu)

        # Create the Conv2D layer with the PCA components as weights
        pca_conv = nn.Conv2d(pool_size, num_pc, kernel_size=(1, 1), stride=1, padding=0)
        pca_conv.weight = nn.Parameter(
            torch.from_numpy(
                np.expand_dims(np.expand_dims(current_u_whitened.T, -1), -1)
            )
        )
        pca_conv.bias = nn.Parameter(torch.from_numpy(-utmu))
        end_time = time.time()
        print(f"{num_pc} PCA computation took {end_time - start_time:.2f} seconds.")

        # Add the PCA layer to the model
        model.add_module(
            f"WPCA_{num_pc}", nn.Sequential(*[pca_conv, Flatten(), L2Norm(dim=-1)])
        )

        # Save the model with the new PCA layer
        torch.save({"num_pcs": num_pc, "state_dict": model.state_dict()}, save_path)

        # Clear GPU memory
        torch.cuda.empty_cache()
        print(save_path)
        print("Done")

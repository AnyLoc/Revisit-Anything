import pytorch_lightning as pl
from vpr_model import VPRModel
from dataloaders.GSVCitiesDataloader import GSVCitiesDataModule
import argparse
import wandb

VAL_DATASETS = ["pitts30k_val", "pitts30k_test", "msls_val"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train VPR model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset_name", type=str, default="gsv_cities", help="Dataset"
    )
    parser.add_argument(
        "--expName", default="0", help="Unique string for an experiment"
    )
    parser.add_argument(
        "--batch_size", type=int, default=60, help="Batch size for the data module"
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
    parser.add_argument("--epochs", type=int, default=4, help="number of epochs")

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
        default=True,
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
        "--val_set_names",
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

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--wpca",
        action="store_true",
        help="Use post pool WPCA layer / cannot be used during training",
    )
    parser.add_argument("--num_pcs", type=int, default=8192, help="Use post pool PCA.")
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Dont use wandb",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="16-mixed",
        choices=["32-true", "16-mixed"],
        help="_",
    )
    parser.add_argument(
        "--loss_name",
        type=str,
        default="MultiSimilarityLoss",
        choices=[
            "ContrastiveLoss",
            "TripletMarginLoss",
            "MultiSimilarityLoss",
            "FastAPLoss",
            "CircleLoss",
            "SupConLoss",
        ],
        help="_",
    )
    parser.add_argument(
        "--miner_name",
        type=str,
        default="MultiSimilarityMiner",
        choices=["TripletMarginMiner", "MultiSimilarityMiner", "PairMarginMiner"],
        help="_",
    )
    parser.add_argument("--miner_margin", type=float, default=0.1, help="_")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./logs/",
        help="path to store the models, e.g. ./logs/",
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

    # no wpca during training
    assert args.wpca == False

    if not args.no_wandb:
        dataset_name = args.dataset_name.lower()[:4]
        wandb_dataStr = dataset_name
        args.expName = "train-" + wandb_dataStr + args.expName

        wandb.init(project="vlad_buff", config=args)
        # update runName
        runName = wandb.run.name
        if (
            args.expName != "" and runName is not None
        ):  # runName is None when running wandb offline
            wandb.run.name = args.expName + "-" + runName.split("-")[-1]
            wandb.run.save()
        else:
            args.expName = runName
        # args = wandb.config

    if args.pl_seed:
        pl.seed_everything(seed=int(args.seed), workers=True)

    datamodule = GSVCitiesDataModule(
        batch_size=args.batch_size,
        img_per_place=args.img_per_place,
        min_img_per_place=args.min_img_per_place,
        shuffle_all=args.shuffle_all,  # shuffle all images or keep shuffling in-city only
        random_sample_from_each_place=args.random_sample_from_each_place,
        image_size=args.resize,
        num_workers=args.num_workers,
        show_data_stats=args.show_data_stats,
        val_set_names=args.val_set_names,  # pitts30k_val, pitts30k_test, msls_val
    )

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
            "args": args,
        }
        useToken = True

    model = VPRModel(
        # ---- Encoder
        backbone_arch=args.backbone,
        backbone_config={
            "num_trainable_blocks": args.num_trainable_blocks,
            "return_token": useToken,
            "norm_layer": args.norm_layer,
        },
        agg_arch=args.aggregation,
        agg_config=agg_config,
        lr=6e-5,
        optimizer="adamw",
        weight_decay=9.5e-9,  # 0.001 for sgd and 0 for adam,
        momentum=0.9,
        lr_sched="linear",
        lr_sched_args={
            "start_factor": 1,
            "end_factor": 0.2,
            "total_iters": 4000,
        },
        # ----- Loss functions
        # example: ContrastiveLoss, TripletMarginLoss, MultiSimilarityLoss,
        # FastAPLoss, CircleLoss, SupConLoss,
        loss_name=args.loss_name,  # "MultiSimilarityLoss",
        miner_name=args.miner_name,  # "MultiSimilarityMiner",  # example: TripletMarginMiner, MultiSimilarityMiner, PairMarginMiner
        miner_margin=args.miner_margin,  # 0.1,
        faiss_gpu=True,
        args=args,
    )
    print(model)
    # model params saving using Pytorch Lightning
    # we save the best 3 models accoring to Recall@1 on pittsburg val
    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        monitor="pitts30k_val/R1",
        filename=f"{model.encoder_arch}"
        + "_({epoch:02d})_R1[{pitts30k_val/R1:.4f}]_R5[{pitts30k_val/R5:.4f}]",
        auto_insert_metric_name=False,
        save_weights_only=True,
        save_top_k=args.epochs,
        save_last=True,
        mode="max",
    )

    # ------------------
    # we instanciate a trainer
    trainer_params = {
        "accelerator": "gpu",
        "devices": 1,
        "default_root_dir": f"{args.save_dir}",  # Tensorflow can be used to viz
        "num_nodes": 1,
        "num_sanity_val_steps": 0,  # runs a validation step before stating training
        "precision": args.precision,  # we use half precision to reduce  memory usage
        "max_epochs": args.epochs,
        "check_val_every_n_epoch": 1,  # run validation every epoch
        "callbacks": [
            checkpoint_cb
        ],  # we only run the checkpointing callback (you can add more)
        "reload_dataloaders_every_n_epochs": 1,  # we reload the dataset to shuffle the order
        "log_every_n_steps": 20,
    }

    if args.pl_seed:
        trainer_params["deterministic"] = True

    trainer = pl.Trainer(**trainer_params)

    # we call the trainer, we give it the model and the datamodule
    trainer.fit(model=model, datamodule=datamodule)

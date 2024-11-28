import pytorch_lightning as pl
import torch
from torch.optim import lr_scheduler, optimizer
import wandb
from torch.utils.data import Subset
import utils
from models import helper
from models.helper import L2Norm, Flatten
from torchvision import transforms as T
from torch import nn
import torch

IMAGENET_MEAN_STD = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}


class VPRModel(pl.LightningModule):
    """This is the main model for Visual Place Recognition
    we use Pytorch Lightning for modularity purposes.

    Args:
        pl (_type_): _description_
    """

    def __init__(
        self,
        # ---- Backbone
        backbone_arch="resnet50",
        backbone_config={},
        # ---- Aggregator
        agg_arch="ConvAP",
        agg_config={},
        # ---- Train hyperparameters
        lr=0.03,
        optimizer="sgd",
        weight_decay=1e-3,
        momentum=0.9,
        lr_sched="linear",
        lr_sched_args={
            "start_factor": 1,
            "end_factor": 0.2,
            "total_iters": 4000,
        },
        # ----- Loss
        loss_name="MultiSimilarityLoss",
        miner_name="MultiSimilarityMiner",
        miner_margin=0.1,
        faiss_gpu=True,
        args=None,
    ):
        super().__init__()

        # Backbone
        self.encoder_arch = backbone_arch
        self.backbone_config = backbone_config

        # Aggregator
        self.agg_arch = agg_arch
        self.agg_config = agg_config

        # Train hyperparameters
        self.lr = lr
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.lr_sched = lr_sched
        self.lr_sched_args = lr_sched_args

        # Loss
        self.loss_name = loss_name
        self.miner_name = miner_name
        self.miner_margin = miner_margin

        self.save_hyperparameters()  # write hyperparams into a file

        self.loss_fn = utils.get_loss(loss_name)
        self.miner = utils.get_miner(miner_name, miner_margin)
        self.batch_acc = (
            []
        )  # we will keep track of the % of trivial pairs/triplets at the loss level

        self.faiss_gpu = faiss_gpu
        self.args = args
        # ----------------------------------
        # get the backbone and the aggregator
        self.backbone = helper.get_backbone(backbone_arch, backbone_config)

        if "netvlad" in agg_arch.lower():
            # Create an instance of the AggConfig class
            # agg_config = AggConfig(agg_config)
            self.aggLayer = helper.get_aggregator(agg_arch, agg_config)

            # cluster using gsv single city
            if agg_config.initialize_clusters:
                from dataloaders.GSVCitiesDataset import GSVCitiesDataset
                # Instantiate GSVCitiesDataset with the desired city
                selected_city = "London"  # Replace with the city you want
                single_city_dataset = GSVCitiesDataset(
                    cities=[selected_city],
                    img_per_place=1,  # Adjust as needed
                    min_img_per_place=1,  # Adjust as needed
                    random_sample_from_each_place=True,
                    transform=T.Compose(
                        [
                            T.Resize(
                                self.args.resize,
                                interpolation=T.InterpolationMode.BILINEAR,
                            ),
                            T.ToTensor(),
                            T.Normalize(
                                mean=IMAGENET_MEAN_STD["mean"],
                                std=IMAGENET_MEAN_STD["std"],
                            ),  # Adjust mean and std if needed
                        ]
                    ),
                )
                self.aggLayer.initialize_netvlad_layer(
                    agg_config, single_city_dataset, self.backbone
                )

            if agg_config.l2 == "before_pool":
                self.aggLayer = nn.Sequential(L2Norm(), self.aggLayer, Flatten())
            elif agg_config.l2 == "after_pool":
                self.aggLayer = nn.Sequential(self.aggLayer, L2Norm(), Flatten())
            elif agg_config.l2 == "onlyFlatten":
                self.aggLayer = nn.Sequential(self.aggLayer, Flatten())

            if (
                agg_config.useFC
            ):  # fc_dim used so have a NV agg layer in nn.Seq aggregation layer
                if agg_config.nv_pca is not None:
                    netvlad_output_dim = agg_config.nv_pca

                netvlad_output_dim *= agg_config.clusters_num

                if agg_config.fc_output_dim == 0:
                    fcLayer = nn.Identity()
                    agg_config.fc_output_dim = netvlad_output_dim
                else:
                    fcLayer = nn.Linear(netvlad_output_dim, agg_config.fc_output_dim)
                    agg_config.fc_output_dim = agg_config.fc_output_dim

                self.aggregator = nn.Sequential(self.aggLayer, fcLayer, L2Norm())

            else:  # no fc_dim used
                self.aggregator = self.aggLayer
                del self.aggLayer

                # check wpca layer to be used? / can be used during evaluation only
                if agg_config.wpca:
                    if args.nv_pca is not None:
                        netvlad_output_dim = args.nv_pca
                    else:
                        netvlad_output_dim = agg_config.dim
                    netvlad_output_dim = agg_config.clusters_num * netvlad_output_dim
                    pca_conv = nn.Conv2d(
                        netvlad_output_dim,
                        agg_config.num_pcs,
                        kernel_size=(1, 1),
                        stride=1,
                        padding=0,
                    )
                    self.WPCA = nn.Sequential(*[pca_conv, Flatten(), L2Norm(dim=-1)])

        else:  # SALAD
            self.aggregator = helper.get_aggregator(agg_arch, agg_config)

            # check wpca layer to be used? / can be used during evaluation only
            if args.wpca:
                salad_output_dim = args.num_clusters * args.cluster_dim + args.token_dim
                pca_conv = nn.Conv2d(
                    salad_output_dim,
                    args.num_pcs,
                    kernel_size=(1, 1),
                    stride=1,
                    padding=0,
                )
                self.WPCA = nn.Sequential(*[pca_conv, Flatten(), L2Norm(dim=-1)])
        # For validation in Lightning v2.0.0
        self.val_outputs = []

    # the forward pass of the lightning model
    def forward(self, x):
        x = self.backbone(x)
        x = self.aggregator(x)
        return x

    # configure the optimizer
    def configure_optimizers(self):
        if self.optimizer.lower() == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
                momentum=self.momentum,
            )
        elif self.optimizer.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        elif self.optimizer.lower() == "adam":
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        else:
            raise ValueError(
                f'Optimizer {self.optimizer} has not been added to "configure_optimizers()"'
            )

        if self.lr_sched.lower() == "multistep":
            scheduler = lr_scheduler.MultiStepLR(
                optimizer,
                milestones=self.lr_sched_args["milestones"],
                gamma=self.lr_sched_args["gamma"],
            )
        elif self.lr_sched.lower() == "cosine":
            scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer, self.lr_sched_args["T_max"]
            )
        elif self.lr_sched.lower() == "linear":
            scheduler = lr_scheduler.LinearLR(
                optimizer,
                start_factor=self.lr_sched_args["start_factor"],
                end_factor=self.lr_sched_args["end_factor"],
                total_iters=self.lr_sched_args["total_iters"],
            )

        return [optimizer], [scheduler]

    # configure the optizer step, takes into account the warmup stage
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # warm up lr
        optimizer.step(closure=optimizer_closure)
        self.lr_schedulers().step()

    #  The loss function call (this method will be called at each training iteration)
    def loss_function(self, descriptors, labels):
        # we mine the pairs/triplets if there is an online mining strategy
        if self.miner is not None:
            miner_outputs = self.miner(descriptors, labels)
            loss = self.loss_fn(descriptors, labels, miner_outputs)

            # calculate the % of trivial pairs/triplets
            # which do not contribute in the loss value
            nb_samples = descriptors.shape[0]
            nb_mined = len(set(miner_outputs[0].detach().cpu().numpy()))
            batch_acc = 1.0 - (nb_mined / nb_samples)

        else:  # no online mining
            loss = self.loss_fn(descriptors, labels)
            batch_acc = 0.0
            if type(loss) == tuple:
                # somes losses do the online mining inside (they don't need a miner objet),
                # so they return the loss and the batch accuracy
                # for example, if you are developping a new loss function, you might be better
                # doing the online mining strategy inside the forward function of the loss class,
                # and return a tuple containing the loss value and the batch_accuracy (the % of valid pairs or triplets)
                loss, batch_acc = loss

        # keep accuracy of every batch and later reset it at epoch start
        self.batch_acc.append(batch_acc)
        # log it
        self.log(
            "b_acc",
            sum(self.batch_acc) / len(self.batch_acc),
            prog_bar=True,
            logger=True,
        )
        if not self.args.no_wandb:
            wandb.log({"b_acc": sum(self.batch_acc) / len(self.batch_acc)})

        return loss

    # This is the training step that's executed at each iteration
    def training_step(self, batch, batch_idx):
        places, labels = batch

        # Note that GSVCities yields places (each containing N images)
        # which means the dataloader will return a batch containing BS places
        BS, N, ch, h, w = places.shape

        # reshape places and labels
        images = places.view(BS * N, ch, h, w)
        labels = labels.view(-1)

        # Feed forward the batch to the model
        descriptors = self(
            images
        )  # Here we are calling the method forward that we defined above

        if torch.isnan(descriptors).any():
            raise ValueError("NaNs in descriptors")

        loss = self.loss_function(
            descriptors, labels
        )  # Call the loss_function we defined above

        self.log("loss", loss.item(), logger=True, prog_bar=True)
        if not self.args.no_wandb:
            wandb.log({"loss": loss.item()})
        return {"loss": loss}

    def on_train_epoch_end(self):
        # we empty the batch_acc list for next epoch
        self.batch_acc = []

    # For validation, we will also iterate step by step over the validation set
    # this is the way Pytorch Lghtning is made. All about modularity, folks.
    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        places, _ = batch
        descriptors = self(places)
        self.val_outputs[dataloader_idx].append(descriptors.detach().cpu())
        return descriptors.detach().cpu()

    def on_validation_epoch_start(self):
        # reset the outputs list
        self.val_outputs = [
            [] for _ in range(len(self.trainer.datamodule.val_datasets))
        ]

    def on_validation_epoch_end(self):
        """this return descriptors in their order
        depending on how the validation dataset is implemented
        for this project (MSLS val, Pittburg val), it is always references then queries
        [R1, R2, ..., Rn, Q1, Q2, ...]
        """
        val_step_outputs = self.val_outputs

        dm = self.trainer.datamodule
        # The following line is a hack: if we have only one validation set, then
        # we need to put the outputs in a list (Pytorch Lightning does not do it presently)
        if len(dm.val_datasets) == 1:  # we need to put the outputs in a list
            val_step_outputs = [val_step_outputs]

        for i, (val_set_name, val_dataset) in enumerate(
            zip(dm.val_set_names, dm.val_datasets)
        ):
            feats = torch.concat(val_step_outputs[i], dim=0)

            if "pitts" in val_set_name:
                # split to ref and queries
                num_references = val_dataset.dbStruct.numDb
                positives = val_dataset.getPositives()
            elif "msls" in val_set_name:
                # split to ref and queries
                num_references = val_dataset.num_references
                positives = val_dataset.pIdx
            else:
                print(f"Please implement validation_epoch_end for {val_set_name}")
                raise NotImplemented

            r_list = feats[:num_references]
            q_list = feats[num_references:]
            distances, predictions, pitts_dict = utils.get_validation_recalls(
                r_list=r_list,
                q_list=q_list,
                k_values=[1, 5, 10],  # , 15, 20, 50, 100],
                gt=positives,
                print_results=True,
                dataset_name=val_set_name,
                faiss_gpu=self.faiss_gpu,
            )
            del r_list, q_list, feats, num_references, positives

            self.log(f"{val_set_name}/R1", pitts_dict[1], prog_bar=False, logger=True)
            self.log(f"{val_set_name}/R5", pitts_dict[5], prog_bar=False, logger=True)
            self.log(f"{val_set_name}/R10", pitts_dict[10], prog_bar=False, logger=True)
            if not self.args.no_wandb:
                metrics = {
                    f"{val_set_name}-val/Recall@{k}": v for k, v in pitts_dict.items()
                }
                wandb.log(metrics)

        print("\n\n")

        # reset the outputs list
        self.val_outputs = []

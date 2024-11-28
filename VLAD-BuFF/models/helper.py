import numpy as np
from models import aggregators
from models import backbones
from torchvision import transforms as T
from torch import nn
import torch
import torch.nn.functional as F


class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim)


class Flatten(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        if len(x.shape) > 2:
            assert x.shape[2] == x.shape[3] == 1, f"{x.shape[2]} != {x.shape[3]} != 1"
            return x[:, :, 0, 0]
        else:
            return x


def get_backbone(backbone_arch="resnet50", backbone_config={}):
    """Helper function that returns the backbone given its name

    Args:
        backbone_arch (str, optional): . Defaults to 'resnet50'.
        backbone_config (dict, optional): this must contain all the arguments needed to instantiate the backbone class. Defaults to {}.

    Returns:
        nn.Module: the backbone as a nn.Model object
    """
    if "resnet" in backbone_arch.lower():
        return backbones.ResNet(backbone_arch, **backbone_config)

    elif "dinov2" in backbone_arch.lower():
        return backbones.DINOv2(model_name=backbone_arch, **backbone_config)


def get_aggregator(agg_arch="ConvAP", agg_config={}):
    """Helper function that returns the aggregation layer given its name.
    If you happen to make your own aggregator, you might need to add a call
    to this helper function.

    Args:
        agg_arch (str, optional): the name of the aggregator. Defaults to 'ConvAP'.
        agg_config (dict, optional): this must contain all the arguments needed to instantiate the aggregator class. Defaults to {}.
        args (dict, optional): all the arguments
    Returns:
        nn.Module: the aggregation layer
    """

    if "cosplace" in agg_arch.lower():
        assert "in_dim" in agg_config
        assert "out_dim" in agg_config
        return aggregators.CosPlace(**agg_config)

    elif "gem" in agg_arch.lower():
        if agg_config == {}:
            agg_config["p"] = 3
        else:
            assert "p" in agg_config
        return aggregators.GeMPool(**agg_config)

    elif "convap" in agg_arch.lower():
        assert "in_channels" in agg_config
        return aggregators.ConvAP(**agg_config)

    elif "mixvpr" in agg_arch.lower():
        assert "in_channels" in agg_config
        assert "out_channels" in agg_config
        assert "in_h" in agg_config
        assert "in_w" in agg_config
        assert "mix_depth" in agg_config
        return aggregators.MixVPR(**agg_config)

    elif "salad" in agg_arch.lower():
        assert "num_channels" in agg_config
        assert "num_clusters" in agg_config
        assert "cluster_dim" in agg_config
        assert "token_dim" in agg_config
        return aggregators.SALAD(**agg_config)

    elif "netvlad" in agg_arch.lower():
        return aggregators.NetVLAD(
            clusters_num=agg_config.clusters_num, dim=agg_config.dim, args=agg_config
        )

import math
import torch
import faiss
import logging
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.decomposition import PCA
import os


class MAC(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return LF.mac(x)

    def __repr__(self):
        return self.__class__.__name__ + "()"


class AntiBurst(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.ab_params = nn.Parameter(
            torch.from_numpy(
                np.array([args.ab_w, args.ab_b, args.ab_p], dtype=np.float32)
            )
        )

    def forward(self, x):
        """
        x of shape B, D, H, W
        """
        B, D, H, W = x.shape
        x = x.view(B, D, -1)
        selfDis = -2 + 2 * x.permute(0, 2, 1) @ x
        w = getWeights(selfDis, self.ab_params, self.args.ab_relu, self.args.ab_inv)
        x = x / w[:, None, :]
        return x.view(B, D, H, W)


class GAP(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if self.args.antiburst:
            self.ab = AntiBurst(args)

    def forward(self, x):
        if self.args.antiburst:
            x = self.ab(x)
        return F.adaptive_avg_pool2d(x, 1)

    def __repr__(self):
        return self.__class__.__name__ + "()"


class SPoC(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return LF.spoc(x)

    def __repr__(self):
        return self.__class__.__name__ + "()"


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, work_with_tokens=False):
        super().__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps
        self.work_with_tokens = work_with_tokens

    def forward(self, x):
        return LF.gem(x, p=self.p, eps=self.eps, work_with_tokens=self.work_with_tokens)

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )


class RMAC(nn.Module):
    def __init__(self, L=3, eps=1e-6):
        super().__init__()
        self.L = L
        self.eps = eps

    def forward(self, x):
        return LF.rmac(x, L=self.L, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + "(" + "L=" + "{}".format(self.L) + ")"


class Flatten(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        assert x.shape[2] == x.shape[3] == 1
        return x[:, :, 0, 0]


class RRM(nn.Module):
    """Residual Retrieval Module as described in the paper
    `Leveraging EfficientNet and Contrastive Learning for AccurateGlobal-scale
    Location Estimation <https://arxiv.org/pdf/2105.07645.pdf>`
    """

    def __init__(self, dim):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.flatten = Flatten()
        self.ln1 = nn.LayerNorm(normalized_shape=dim)
        self.fc1 = nn.Linear(in_features=dim, out_features=dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=dim, out_features=dim)
        self.ln2 = nn.LayerNorm(normalized_shape=dim)
        self.l2 = normalization.L2Norm()

    def forward(self, x):
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.ln1(x)
        identity = x
        out = self.fc2(self.relu(self.fc1(x)))
        out += identity
        out = self.l2(self.ln2(out))
        return out


def getWeights(dis, ab_params, relu=False, do_inv=False, do_soft=False):
    # dis is Batch x N_elems x N_elems
    if do_inv:
        dis = dis / ab_params[0]
    else:
        dis = dis * ab_params[0]
    dis = dis + ab_params[1]
    if relu:
        dis = F.relu(dis)
    if do_soft:
        w = torch.softmax(dis, dim=-1).sum(-1)
    else:
        w = torch.sigmoid(dis).sum(-1)
    w = w ** ab_params[2]  # .squeeze()
    return w


# based on https://github.com/lyakaap/NetVLAD-pytorch/blob/master/netvlad.py
class NetVLAD(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(
        self,
        clusters_num=64,
        dim=128,
        normalize_input=True,
        work_with_tokens=False,
        args=None,
    ):
        """
        Args:
            clusters_num : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
            args : None
                Arguments
        """
        super().__init__()
        self.expName = args.expName
        self.clusters_num = clusters_num
        self.dim = dim
        if args.nv_pca is not None:
            # self.pca_mean, self.pca_rot = None, None
            self.pca_mean = nn.Parameter(torch.rand(self.dim))
            self.pca_rot = nn.Parameter(torch.rand(args.nv_pca, self.dim))
            if args.nv_pca_alt:
                self.bottleneck = nn.Linear(self.dim, args.nv_pca, bias=True)
            elif args.nv_pca_alt_mlp:
                self.mlp = nn.Sequential(
                    nn.Linear(self.dim, args.nv_pca, bias=True),
                    nn.ReLU(),
                    nn.Linear(args.nv_pca, args.nv_pca, bias=True),
                )
            dims = args.nv_pca
        else:
            dims = dim
        self.alpha = 0
        self.normalize_input = normalize_input
        self.work_with_tokens = work_with_tokens
        if work_with_tokens:
            self.conv = nn.Conv1d(dims, clusters_num, kernel_size=1, bias=False)
        else:
            self.conv = nn.Conv2d(dims, clusters_num, kernel_size=(1, 1), bias=False)
        self.centroids = nn.Parameter(torch.rand(clusters_num, dims))
        self.args = args
        if self.args.antiburst:
            self.ab_params = nn.Parameter(
                torch.from_numpy(
                    np.array([args.ab_w, args.ab_b, args.ab_p], dtype=np.float32)
                ),
                requires_grad=not self.args.ab_fixed,
            )
            if self.args.ab_gen == 1:
                # create a 3 x num_cc matrix to select ab_params from
                self.ab_cc = nn.Parameter(torch.ones(clusters_num, 3))
        if self.args.ab_t is not None:
            self.ab_t = nn.Parameter(
                torch.from_numpy(np.array([self.args.ab_t], dtype=np.float32))
            )
        if self.args.ab_kp is not None:
            self.ab_kp = nn.Sequential(
                nn.Conv2d(dims, self.args.ab_kp, kernel_size=(1, 1)),
                nn.ReLU(),
                nn.Conv2d(self.args.ab_kp, 1, kernel_size=(1, 1)),
            )

    def init_params(self, centroids, descriptors, pcaData=None):
        centroids_assign = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
        dots = np.dot(centroids_assign, descriptors.T)
        dots.sort(0)
        dots = dots[::-1, :]  # sort, descending

        self.alpha = (-np.log(0.01) / np.mean(dots[0, :] - dots[1, :])).item()
        self.centroids = nn.Parameter(torch.from_numpy(centroids))
        if self.work_with_tokens:
            self.conv.weight = nn.Parameter(
                torch.from_numpy(self.alpha * centroids_assign).unsqueeze(2)
            )
        else:
            self.conv.weight = nn.Parameter(
                torch.from_numpy(self.alpha * centroids_assign)
                .unsqueeze(2)
                .unsqueeze(3)
            )
        self.conv.bias = None
        if pcaData is not None:
            if self.args.nv_pca_randinit:
                self.pca_mean = nn.Parameter(torch.rand(pcaData[0].shape))
                self.pca_rot = nn.Parameter(torch.rand(pcaData[1].shape))
            else:
                self.pca_mean = nn.Parameter(torch.from_numpy(pcaData[0]))
                self.pca_rot = nn.Parameter(torch.from_numpy(pcaData[1]))

    def forward(self, x, wParams=None, labels=None, dirPath=None):
        if self.work_with_tokens:
            x = x.permute(0, 2, 1)
            N, D, _ = x.shape[:]
        else:
            N, D, H, W = x.shape[:]
        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # Across descriptor dim
        x_flatten = x.view(N, D, -1)
        if self.args.nv_pca is not None:
            if self.args.nv_pca_alt:
                x_flatten = self.bottleneck(x_flatten.permute(0, 2, 1)).permute(0, 2, 1)
            elif self.args.nv_pca_alt_mlp:
                x_flatten = self.mlp(x_flatten.permute(0, 2, 1)).permute(0, 2, 1)
            else:
                x_flatten = x_flatten - self.pca_mean[None, :, None]
                x_flatten = (x_flatten.permute(0, 2, 1) @ self.pca_rot.T).permute(
                    0, 2, 1
                )
            if self.normalize_input:
                x_flatten = F.normalize(x_flatten, p=2, dim=1)  # Across descriptor dim
            D = x_flatten.shape[1]
            x = x_flatten.view(N, D, H, W)
        soft_assign = self.conv(x).view(N, self.clusters_num, -1)
        soft_assign = F.softmax(soft_assign, dim=1)
        # setting all none for self.storeSAB when no antiBurst
        soft_assign_alpha = None
        w_burst = None
        selfDis = None
        if self.args.antiburst or self.args.ab_testOnly or self.args.ab_t is not None:
            soft_assign_alpha = soft_assign.clone()
            x_flat_new = x_flatten
            selfDis = -2 + 2 * x_flat_new.permute(0, 2, 1) @ x_flat_new
            use_relu = True if self.args.ab_relu else False
            do_inv = True if self.args.ab_inv else False
            do_soft = True if self.args.ab_soft else False
            if self.args.ab_gen == 1:
                ab_params_gen = (
                    soft_assign.permute(0, 2, 1) @ self.ab_cc
                )  # -> Batch x N_elems x 3
                ab_params_gen = ab_params_gen.permute(2, 0, 1).unsqueeze(
                    -1
                )  # -> 3 x Batch x N_elems x 1
            if self.args.ab_testOnly:
                if self.args.ab_t is None:
                    ab_params = torch.tensor(
                        [self.args.ab_w, self.args.ab_b, self.args.ab_p]
                    )
                    w_burst = (
                        getWeights(selfDis, ab_params, use_relu, do_inv, do_soft)
                        if wParams is None
                        else getWeights(selfDis, wParams, use_relu)
                    )
                else:
                    w_burst = torch.count_nonzero(
                        F.relu(selfDis + self.args.ab_t), dim=-1
                    )
            else:
                if self.args.ab_t is None:
                    if self.args.ab_gen == 1:
                        ab_params = ab_params_gen
                    elif self.args.ab_wOnly:
                        ab_params = [self.ab_params[0], self.args.ab_b, self.args.ab_p]
                    else:
                        ab_params = self.ab_params
                    w_burst = getWeights(selfDis, ab_params, use_relu, do_inv, do_soft)
                else:
                    # w_burst = torch.sign(F.relu(selfDis + self.ab_t)).sum(-1)
                    w_burst = torch.sigmoid(
                        self.ab_params[0] * F.relu(selfDis + self.ab_t)
                    ).sum(-1)
            soft_assign = soft_assign / w_burst[:, None, :]
        elif self.args.ab_kp:
            w = torch.softmax(self.ab_kp(x).view(N, H * W), dim=-1)
            soft_assign = soft_assign / w[:, None, :]
        vlad = torch.zeros(
            [N, self.clusters_num, D], dtype=x_flatten.dtype, device=x_flatten.device
        )
        # not use for loop, fast
        if self.args.forLoopAlt:
            vlad = (
                (x_flatten[:, None, :, :] - self.centroids[None, :, :, None])
                * soft_assign[:, :, None, :]
            ).sum(-1)
        else:
            for D in range(
                self.clusters_num
            ):  # Slower than non-looped, but lower memory usage
                residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3) - self.centroids[
                    D : D + 1, :
                ].expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
                residual = residual * soft_assign[:, D : D + 1, :].unsqueeze(2)
                vlad[:, D : D + 1, :] = residual.sum(dim=-1)
        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(N, -1)  # Flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize
        if self.args.storeSAB:
            # store parameters
            os.makedirs(dirPath, exist_ok=True)
            results = {}
            for i in range(N):
                image_id = labels[
                    i
                ].item()  # Assuming labels are numpy array of image IDs
                file_path = os.path.join(dirPath, f"{image_id}.npz")
                if not os.path.exists(file_path):
                    results = {
                        "soft_assign_alpha": soft_assign_alpha[i].cpu().numpy()
                        if soft_assign_alpha is not None
                        else None,
                        "selfDis": selfDis[i].cpu().numpy()
                        if selfDis is not None
                        else None,
                        "w_burst": w_burst[i].cpu().numpy()
                        if w_burst is not None
                        else None,
                        "soft_assign_adj": soft_assign[i].cpu().numpy(),
                        "vlad": vlad[i].cpu().numpy(),
                    }
                    # Save the results as a .npz file
                    np.savez(file_path, **results)
        #                else:
        #                   print(f"Skipping {file_path}, already exists.")
        del soft_assign_alpha
        return vlad

    def initialize_netvlad_layer(self, args, cluster_ds, backbone):
        descriptors_num = 50000
        descs_num_per_image = 100
        images_num = math.ceil(descriptors_num / descs_num_per_image)
        total_cluster_ds_images = cluster_ds.total_nb_images
        random_sampler = SubsetRandomSampler(
            np.random.choice(len(cluster_ds), images_num, replace=False)
        )
        random_dl = DataLoader(
            dataset=cluster_ds,
            num_workers=args.num_workers,
            batch_size=args.infer_batch_size,
            sampler=random_sampler,
        )
        with torch.no_grad():
            backbone = backbone.eval()
            backbone = backbone.to(args.device)  # Move the model to the GPU
            logging.debug("Extracting features to initialize NetVLAD layer")
            descriptors = np.zeros(shape=(descriptors_num, self.dim), dtype=np.float32)
            # descriptors_size = (images_num * args.infer_batch_size * descs_num_per_image, self.dim)
            # descriptors = np.zeros(descriptors_size, dtype=np.float32)
            for iteration, (inputs) in enumerate(tqdm(random_dl, ncols=100)):
                inputs = inputs[0]
                inputs = inputs.view(
                    -1, *inputs.shape[2:]
                )  # BS, K, channels, height, width -> BS*K, channels, height, width
                inputs = inputs.to(args.device)
                logging.debug(f"Input shape: {inputs.shape}")
                outputs = backbone(inputs)
                norm_outputs = F.normalize(outputs, p=2, dim=1)
                image_descriptors = norm_outputs.view(
                    norm_outputs.shape[0], self.dim, -1
                ).permute(0, 2, 1)
                image_descriptors = image_descriptors.cpu().numpy()
                batchix = iteration * args.infer_batch_size * descs_num_per_image
                for ix in range(image_descriptors.shape[0]):
                    sample = np.random.choice(
                        image_descriptors.shape[1], descs_num_per_image, replace=False
                    )
                    startix = batchix + ix * descs_num_per_image
                    descriptors[
                        startix : startix + descs_num_per_image, :
                    ] = image_descriptors[ix, sample, :]
        if args.nv_pca is not None:
            if args.nv_pca_alt:
                w, b = (
                    self.bottleneck.weight.detach().cpu().numpy(),
                    self.bottleneck.bias.detach().cpu().numpy(),
                )
                descriptors = descriptors @ w.T + b
                pcaData = None
            elif args.nv_pca_alt_mlp:
                descriptors = (
                    self.mlp(torch.from_numpy(descriptors)).detach().cpu().numpy()
                )
                pcaData = None
            else:
                pca = PCA(args.nv_pca, random_state=0)
                pca.fit(descriptors)
                descriptors = pca.transform(descriptors)
                pcaData = pca.mean_, pca.components_
            dims = args.nv_pca
        else:
            pcaData = None
            dims = self.dim
        try:
            kmeans = faiss.Kmeans(dims, self.clusters_num, niter=100, verbose=True)
            kmeans.train(descriptors)
            logging.debug(f"NetVLAD centroids shape: {kmeans.centroids.shape}")
            self.init_params(kmeans.centroids, descriptors, pcaData)
            self = self.to(args.device)

        except Exception as e:
            import traceback

            traceback.print_exc()
            raise e


class CRNModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Downsample pooling
        self.downsample_pool = nn.AvgPool2d(
            kernel_size=3, stride=(2, 2), padding=0, ceil_mode=True
        )

        # Multiscale Context Filters
        self.filter_3_3 = nn.Conv2d(
            in_channels=dim, out_channels=32, kernel_size=(3, 3), padding=1
        )
        self.filter_5_5 = nn.Conv2d(
            in_channels=dim, out_channels=32, kernel_size=(5, 5), padding=2
        )
        self.filter_7_7 = nn.Conv2d(
            in_channels=dim, out_channels=20, kernel_size=(7, 7), padding=3
        )

        # Accumulation weight
        self.acc_w = nn.Conv2d(in_channels=84, out_channels=1, kernel_size=(1, 1))
        # Upsampling
        self.upsample = F.interpolate

        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize Context Filters
        torch.nn.init.xavier_normal_(self.filter_3_3.weight)
        torch.nn.init.constant_(self.filter_3_3.bias, 0.0)
        torch.nn.init.xavier_normal_(self.filter_5_5.weight)
        torch.nn.init.constant_(self.filter_5_5.bias, 0.0)
        torch.nn.init.xavier_normal_(self.filter_7_7.weight)
        torch.nn.init.constant_(self.filter_7_7.bias, 0.0)

        torch.nn.init.constant_(self.acc_w.weight, 1.0)
        torch.nn.init.constant_(self.acc_w.bias, 0.0)
        self.acc_w.weight.requires_grad = False
        self.acc_w.bias.requires_grad = False

    def forward(self, x):
        # Contextual Reweighting Network
        x_crn = self.downsample_pool(x)

        # Compute multiscale context filters g_n
        g_3 = self.filter_3_3(x_crn)
        g_5 = self.filter_5_5(x_crn)
        g_7 = self.filter_7_7(x_crn)
        g = torch.cat((g_3, g_5, g_7), dim=1)
        g = F.relu(g)

        w = F.relu(self.acc_w(g))  # Accumulation weight
        mask = self.upsample(w, scale_factor=2, mode="bilinear")  # Reweighting Mask

        return mask


class CRN(NetVLAD):
    def __init__(self, clusters_num=64, dim=128, normalize_input=True):
        super().__init__(clusters_num, dim, normalize_input)
        self.crn = CRNModule(dim)

    def forward(self, x):
        N, D, H, W = x.shape[:]
        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # Across descriptor dim

        mask = self.crn(x)

        x_flatten = x.view(N, D, -1)
        soft_assign = self.conv(x).view(N, self.clusters_num, -1)
        soft_assign = F.softmax(soft_assign, dim=1)

        # Weight soft_assign using CRN's mask
        soft_assign = soft_assign * mask.view(N, 1, H * W)

        vlad = torch.zeros(
            [N, self.clusters_num, D], dtype=x_flatten.dtype, device=x_flatten.device
        )
        for D in range(
            self.clusters_num
        ):  # Slower than non-looped, but lower memory usage
            residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3) - self.centroids[
                D : D + 1, :
            ].expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
            residual = residual * soft_assign[:, D : D + 1, :].unsqueeze(2)
            vlad[:, D : D + 1, :] = residual.sum(dim=-1)

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(N, -1)  # Flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize
        return vlad

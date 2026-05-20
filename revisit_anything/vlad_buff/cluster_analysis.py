import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from skimage.measure import label
from skimage.color import label2rgb
from PIL import Image
import seaborn as sns
import imageio
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import argparse
from eval import get_val_dataset
import pandas as pd
import mpld3
from mpld3 import plugins

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


def load_and_visualize(
    image_id,
    base_dir,
    ds,
    traverse,
    img_type,
    clusterNo=None,
    w=23,
    image_size=[322, 322],
    coloR=None,
    method_our=None,
    baseline_name=None,
    gif_path=None,
    analysis_dir=None,
):
    """
    Load and visualize the stored variables for the given image IDs.
    """
    if baseline_name is None:
        print("baseline_name cannot be None")
        return

    dir_path_nv = os.path.join(base_dir, "lightning_logs", baseline_name, ds)
    dir_path_vb = os.path.join(base_dir, "lightning_logs", method_our, ds)

    ev_dataset, num_ref, num_q, gt = get_val_dataset(ds, image_size)
    if traverse == "database":
        file_path_vb = os.path.join(dir_path_vb, f"{image_id}.npz")
        file_path_nv = os.path.join(dir_path_nv, f"{image_id}.npz")
    else:
        file_path_vb = os.path.join(dir_path_vb, f"{num_ref+image_id}.npz")
        file_path_nv = os.path.join(dir_path_nv, f"{num_ref+image_id}.npz")

    if os.path.exists(file_path_vb):
        print(file_path_vb)
        print(file_path_nv)
        data_vb = np.load(file_path_vb)
        data_nv = np.load(file_path_nv)

        soft_assign_alpha_nv = None
        soft_assign_alpha_vb = None
        soft_assign_adj_vb = None
        selfDis = None
        w_burst = None

        # VB
        soft_assign_alpha_vb = data_vb.get("soft_assign_alpha")
        selfDis = data_vb.get("selfDis")
        w_burst = data_vb.get("w_burst")
        w_burst = w_burst.reshape(w, w)
        w_burst = 1.0 / w_burst
        soft_assign_adj_vb = data_vb.get("soft_assign_adj")
        vlad_vb = data_vb.get("vlad")

        # NV / baseline_name
        soft_assign_alpha_nv = data_nv.get("soft_assign_adj")
        vlad_nv = data_nv.get("vlad")

        if traverse != "database":
            img_path = (
                ev_dataset.dataset_root
                + ev_dataset.images[ev_dataset.num_references + image_id]
            )

        else:
            img_path = ev_dataset.dataset_root + ev_dataset.images[image_id]

        vlad_vb, vlad_nv = vlad_vb.reshape(64, -1), vlad_nv.reshape(64, -1)

        frames = []
        clusters = range(64) if clusterNo is None else [clusterNo]
        for cluster in clusters:
            if clusterNo == -1:
                break

            if clusterNo is None:
                fig, ax = plt.subplots(figsize=(10, 6))
                image = Image.open(img_path)
                image_resized = image.resize((w, w))
                ax.imshow(image_resized)
                ax.set_title(f"Cluster: {cluster}")
                im1 = ax.imshow(
                    soft_assign_alpha_nv[cluster, :].reshape(w, w),
                    aspect="auto",
                    alpha=0.35,
                )
                plt.colorbar(im1)
                plt.axis("off")
                cluster_img_path = f"{analysis_dir}/{baseline_name}/{ds}_{baseline_name}_{img_type}_{image_id}_{cluster}.png"
                plt.savefig(cluster_img_path, bbox_inches="tight", pad_inches=0)
                plt.close(fig)

                fig, ax = plt.subplots(figsize=(10, 6))
                image = Image.open(img_path)
                image_resized = image.resize((w, w))
                ax.set_title(f"Cluster: {cluster}")
                ax.imshow(image_resized)
                im2 = ax.imshow(
                    soft_assign_adj_vb[cluster, :].reshape(w, w),
                    aspect="auto",
                    alpha=0.35,
                )
                plt.colorbar(im2)
                plt.axis("off")
                cluster_img_path = f"{analysis_dir}/{method_our}/{ds}_{method_our}_{img_type}_{image_id}_{cluster}.png"
                plt.savefig(cluster_img_path, bbox_inches="tight", pad_inches=0)
                plt.close(fig)

                continue

            # Visualize variables
            fig, axes = plt.subplots(2, 6, figsize=(30, 6))
            fig.suptitle(
                f"{img_type} Image ID: {image_id}, Cluster: {cluster}",
                fontsize=16,
                color=coloR,
            )

            image = Image.open(img_path)
            image_resized = image.resize((w, w))
            image_array = np.array(image_resized)

            # Row 1
            axes[0, 0].imshow(image)
            axes[0, 0].axis("off")

            def add_colorbar_to_all(im, fig, ax):
                cax = fig.add_axes([1, 0.55, 0.015, 0.35])  # Adjust as per the need
                plt.colorbar(im, cax=cax)

            if soft_assign_alpha_nv is not None:
                axes[0, 1].set_title(
                    f"{baseline_name}:: SA_A: {soft_assign_alpha_nv.shape}, Cluster: {cluster}"
                )
                axes[0, 1].imshow(image_resized)
                im1 = axes[0, 1].imshow(
                    soft_assign_alpha_nv[cluster, :].reshape(w, w),
                    aspect="auto",
                    alpha=0.75,
                )
                plt.colorbar(im1)
            if soft_assign_adj_vb is not None:
                axes[0, 2].set_title(
                    f"{method_our}:: SA_new (SA_A/w_burst): {soft_assign_adj_vb.shape}, Cluster: {cluster}"
                )
                axes[0, 2].imshow(image_resized)
                im2 = axes[0, 2].imshow(
                    soft_assign_adj_vb[cluster, :].reshape(w, w),
                    aspect="auto",
                    alpha=0.75,
                )
                plt.colorbar(im2)
            if soft_assign_alpha_vb is not None:
                axes[0, 3].set_title(
                    f"{method_our}:: SA_A: {soft_assign_alpha_vb.shape}, Cluster: {cluster}"
                )
                axes[0, 3].imshow(image_resized)
                im3 = axes[0, 3].imshow(
                    soft_assign_alpha_vb[cluster, :].reshape(w, w),
                    aspect="auto",
                    alpha=0.75,
                )
                plt.colorbar(im3)
            if w_burst is not None:
                axes[0, 4].set_title(f"{method_our}:: 1/w_burst: {w_burst.shape}")
                axes[0, 4].imshow(image_resized)
                im4 = axes[0, 4].imshow(w_burst, aspect="auto", alpha=0.75)
                plt.colorbar(im4)
            if selfDis is not None:
                axes[0, 5].set_title(f"{method_our}:: selfDis: {selfDis.shape}")
                im5 = axes[0, 5].imshow(selfDis, aspect="auto")
                plt.colorbar(im5)

            max_intensity = 0
            # Visualize pixel-level intensities in the second row
            if soft_assign_alpha_nv is not None:
                max_intensity = max(
                    max_intensity,
                    visualize_pixel_intensities(
                        soft_assign_alpha_nv[cluster, :].reshape(w, w), axes[1, 1]
                    ),
                )

            if soft_assign_adj_vb is not None:
                max_intensity = max(
                    max_intensity,
                    visualize_pixel_intensities(
                        soft_assign_adj_vb[cluster, :].reshape(w, w), axes[1, 2]
                    ),
                )

            if soft_assign_alpha_vb is not None:
                max_intensity = max(
                    max_intensity,
                    visualize_pixel_intensities(
                        soft_assign_alpha_vb[cluster, :].reshape(w, w), axes[1, 3]
                    ),
                )

            if w_burst is not None:
                visualize_pixel_intensities(w_burst, axes[1, 4])

            axes[1, 0].axis("off")
            axes[1, 5].axis("off")

            # Set the same scale for all plots
            # for ax in axes[1, 1:4]:
            #   ax.set_ylim(0, max_intensity)

            plt.tight_layout()

            # Save the frame if we are creating a GIF
            if clusterNo is None:
                frame_path = f"{analysis_dir}/frame_{cluster}.png"
                plt.savefig(frame_path)
                frames.append(frame_path)
                plt.close(fig)
            else:
                # plt.show()
                frame_path = f"{analysis_dir}/{ds}_{baseline_name}_{img_type}_{image_id}_{cluster}.png"
                plt.savefig(frame_path)
                plt.close(fig)

        if clusterNo is None:
            with imageio.get_writer(
                f"{analysis_dir}/{ds}_{baseline_name}_{img_type}_{image_id}_{gif_path}",
                mode="I",
                duration=0.1,
            ) as writer:
                for frame_path in frames:
                    image = imageio.imread(frame_path)
                    writer.append_data(image)
                    os.remove(frame_path)  # Clean up frame files

        return [vlad_vb, vlad_nv], None

    else:
        print(f"****No data found for {traverse} Image ID: {image_id}*****")
        print(file_path_vb)
        print(file_path_nv)
        print("\n")
        return

    return [], None


def visualize_pixel_intensities(data, ax):
    pixel_values = data.flatten()
    if len(pixel_values) > 0:
        ax.bar(range(len(pixel_values)), pixel_values)
        ax.set_xlabel("Pixel Index")
        ax.set_ylabel("Intensity")
        ax.set_ylim([0, np.max(pixel_values)])
    #  ax.axis('off')
    return np.max(pixel_values) if len(pixel_values) > 0 else 0


def compute_triplet_margin(query, positive, negative):
    return np.linalg.norm(query - negative, axis=1) - np.linalg.norm(
        query - positive, axis=1
    )


def rank_clusters(margins):
    return np.argsort(margins)


def compute_cluster_rank_difference(r_vb, r_nv):
    r_vb, r_nv = list(r_vb), list(r_nv)
    r_d = [r_nv.index(cidx) - r_vb.index(cidx) for cidx in r_nv]
    return [r_d, r_nv[np.argmax(r_d)]]


def HoD(
    qVLAD,
    pVLAD,
    nVLAD,
    base_dir,
    ds,
    query,
    clusterNo,
    method_our=None,
    baseline_name=None,
):
    index = 0
    vlad_q_vb, vlad_p_vb, vlad_n_vb = qVLAD[index], pVLAD[index], nVLAD[index]
    vlad_q_nv, vlad_p_nv, vlad_n_nv = (
        qVLAD[index + 1],
        pVLAD[index + 1],
        nVLAD[index + 1],
    )

    # Compute distances
    dist_pos_vb = np.linalg.norm(vlad_q_vb - vlad_p_vb, axis=1)
    dist_neg_vb = np.linalg.norm(vlad_q_vb - vlad_n_vb, axis=1)
    dist_pos_nv = np.linalg.norm(vlad_q_nv - vlad_p_nv, axis=1)
    dist_neg_nv = np.linalg.norm(vlad_q_nv - vlad_n_nv, axis=1)

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    # vlad-buff
    axes[0].hist(
        dist_pos_vb, bins=30, alpha=0.5, label="Query-Positive Distances", color="g"
    )
    axes[0].hist(
        dist_neg_vb, bins=30, alpha=0.5, label="Query-Negative Distances", color="r"
    )
    axes[0].legend()
    axes[0].set_title(f"Histogram of Distances {method_our}")
    axes[0].set_xlabel("Distance")
    axes[0].set_ylabel("Frequency")

    # NV
    axes[1].hist(
        dist_pos_nv, bins=30, alpha=0.5, label="Query-Positive Distances", color="g"
    )
    axes[1].hist(
        dist_neg_nv, bins=30, alpha=0.5, label="Query-Negative Distances", color="r"
    )
    axes[1].legend()
    axes[1].set_title(f"Histogram of Distances {baseline_name}")
    axes[1].set_xlabel("Distance")
    axes[1].set_ylabel("Frequency")

    plt.tight_layout()
    # plt.show()
    frame_path = f"{base_dir}/{ds}_{query}_HoD.png"
    plt.savefig(frame_path)
    plt.close(fig)


def HoPD(
    qVLAD,
    pVLAD,
    nVLAD,
    base_dir,
    ds,
    query,
    clusterNo,
    method_our=None,
    baseline_name=None,
):
    index = 0
    vlad_q_vb, vlad_p_vb, vlad_n_vb = qVLAD[index], pVLAD[index], nVLAD[index]
    vlad_q_nv, vlad_p_nv, vlad_n_nv = (
        qVLAD[index + 1],
        pVLAD[index + 1],
        nVLAD[index + 1],
    )
    # Combine all VLADs
    combined_vlads_vb = np.vstack([vlad_q_vb, vlad_p_vb, vlad_n_vb])
    combined_vlads_nv = np.vstack([vlad_q_nv, vlad_p_nv, vlad_n_nv])

    # Compute pairwise distances
    pairwise_distances_vb = np.linalg.norm(
        combined_vlads_vb[:, None] - combined_vlads_vb, axis=2
    )
    pairwise_distances_nv = np.linalg.norm(
        combined_vlads_nv[:, None] - combined_vlads_nv, axis=2
    )

    # Create labels for the heatmap
    heatmap_labels_vb = (
        ["Q" + str(i) for i in range(len(vlad_q_vb))]
        + ["P" + str(i) for i in range(len(vlad_p_vb))]
        + ["N" + str(i) for i in range(len(vlad_n_vb))]
    )
    heatmap_labels_nv = (
        ["Q" + str(i) for i in range(len(vlad_q_nv))]
        + ["P" + str(i) for i in range(len(vlad_p_nv))]
        + ["N" + str(i) for i in range(len(vlad_n_nv))]
    )

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    # vlad-buff
    sns.heatmap(
        pairwise_distances_vb,
        xticklabels=heatmap_labels_vb,
        yticklabels=heatmap_labels_vb,
        cmap="viridis",
        ax=axes[0],
    )
    axes[0].set_title(f"Heatmap of Pairwise Distances {method_our}")

    # NV
    sns.heatmap(
        pairwise_distances_nv,
        xticklabels=heatmap_labels_nv,
        yticklabels=heatmap_labels_nv,
        cmap="viridis",
        ax=axes[1],
    )
    axes[1].set_title(f"Heatmap of Pairwise Distances {baseline_name}")

    plt.tight_layout()
    # plt.show()
    frame_path = f"{base_dir}/{ds}_{query}_HoPD.png"
    plt.savefig(frame_path)
    plt.close(fig)


def plot_tsne(
    qVLAD,
    pVLAD,
    nVLAD,
    base_dir,
    ds,
    query,
    pos,
    neg,
    clusterNo,
    method_our=None,
    baseline_name=None,
):
    index = 0
    vlad_q_vb, vlad_p_vb, vlad_n_vb = qVLAD[index], pVLAD[index], nVLAD[index]
    vlad_q_nv, vlad_p_nv, vlad_n_nv = (
        qVLAD[index + 1],
        pVLAD[index + 1],
        nVLAD[index + 1],
    )
    # Combine all VLAD descriptors for both methods
    all_vlads_vb = np.vstack([vlad_q_vb, vlad_p_vb, vlad_n_vb])
    all_vlads_nv = np.vstack([vlad_q_nv, vlad_p_nv, vlad_n_nv])

    # Perform t-SNE
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    tsne_result_vb = tsne.fit_transform(all_vlads_vb)
    tsne_result_nv = tsne.fit_transform(all_vlads_nv)

    # Split t-SNE results back into query, positive, and negative
    tsne_q_vb, tsne_p_vb, tsne_n_vb = np.split(
        tsne_result_vb, [len(vlad_q_vb), len(vlad_p_vb) + len(vlad_n_vb)]
    )
    tsne_q_nv, tsne_p_nv, tsne_n_nv = np.split(
        tsne_result_nv, [len(vlad_q_nv), len(vlad_p_nv) + len(vlad_n_nv)]
    )

    # Create file paths for images
    image_paths_vb = {
        "query": [
            f"{base_dir}/{method_our}/{ds}_{method_our}_Anchor_{query}_{cluster}.png"
            for cluster in range(len(tsne_q_vb))
        ],
        "positive": [
            f"{base_dir}/{method_our}/{ds}_{method_our}_Pos_{pos}_{cluster}.png"
            for cluster in range(len(tsne_p_vb))
        ],
        "negative": [
            f"{base_dir}/{method_our}/{ds}_{method_our}_Neg_{neg}_{cluster}.png"
            for cluster in range(len(tsne_n_vb))
        ],
    }

    image_paths_nv = {
        "query": [
            f"{base_dir}/{baseline_name}/{ds}_{baseline_name}_Anchor_{query}_{cluster}.png"
            for cluster in range(len(tsne_q_nv))
        ],
        "positive": [
            f"{base_dir}/{baseline_name}/{ds}_{baseline_name}_Pos_{pos}_{cluster}.png"
            for cluster in range(len(tsne_p_nv))
        ],
        "negative": [
            f"{base_dir}/{baseline_name}/{ds}_{baseline_name}_Neg_{neg}_{cluster}.png"
            for cluster in range(len(tsne_n_nv))
        ],
    }

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    # Your method
    scatter_vb_query = axes[0].scatter(
        tsne_q_vb[:, 0], tsne_q_vb[:, 1], color="r", label="query"
    )
    scatter_vb_positive = axes[0].scatter(
        tsne_p_vb[:, 0], tsne_p_vb[:, 1], color="g", label="positive"
    )
    scatter_vb_negative = axes[0].scatter(
        tsne_n_vb[:, 0], tsne_n_vb[:, 1], color="b", label="negative"
    )
    axes[0].set_title(f"t-SNE of VLAD Descriptors {method_our}")
    axes[0].legend()

    # baseline_name's method
    scatter_nv_query = axes[1].scatter(
        tsne_q_nv[:, 0], tsne_q_nv[:, 1], color="r", label="query"
    )
    scatter_nv_positive = axes[1].scatter(
        tsne_p_nv[:, 0], tsne_p_nv[:, 1], color="g", label="positive"
    )
    scatter_nv_negative = axes[1].scatter(
        tsne_n_nv[:, 0], tsne_n_nv[:, 1], color="b", label="negative"
    )
    axes[1].set_title(f"t-SNE of VLAD Descriptors {baseline_name}")
    axes[1].legend()

    def create_tooltip(image_paths, category, num_points):
        labels = []
        for i in range(num_points):
            img_path = image_paths[category][i]
            if os.path.exists(img_path):
                img_html = f'<img src="{img_path}" width="300">'
                labels.append(img_html)
            else:
                labels.append(f"Image not found: {img_path}")
        return labels

    labels_vb_query = create_tooltip(image_paths_vb, "query", len(tsne_q_vb))
    labels_vb_positive = create_tooltip(image_paths_vb, "positive", len(tsne_p_vb))
    labels_vb_negative = create_tooltip(image_paths_vb, "negative", len(tsne_n_vb))

    labels_nv_query = create_tooltip(image_paths_nv, "query", len(tsne_q_nv))
    labels_nv_positive = create_tooltip(image_paths_nv, "positive", len(tsne_p_nv))
    labels_nv_negative = create_tooltip(image_paths_nv, "negative", len(tsne_n_nv))

    tooltip_vb_query = plugins.PointHTMLTooltip(
        scatter_vb_query, labels=labels_vb_query, voffset=10, hoffset=10
    )
    plugins.connect(fig, tooltip_vb_query)
    tooltip_vb_positive = plugins.PointHTMLTooltip(
        scatter_vb_positive, labels=labels_vb_positive, voffset=10, hoffset=10
    )
    plugins.connect(fig, tooltip_vb_positive)
    tooltip_vb_negative = plugins.PointHTMLTooltip(
        scatter_vb_negative, labels=labels_vb_negative, voffset=10, hoffset=10
    )
    plugins.connect(fig, tooltip_vb_negative)

    tooltip_nv_query = plugins.PointHTMLTooltip(
        scatter_nv_query, labels=labels_nv_query, voffset=10, hoffset=10
    )
    plugins.connect(fig, tooltip_nv_query)
    tooltip_nv_positive = plugins.PointHTMLTooltip(
        scatter_nv_positive, labels=labels_nv_positive, voffset=10, hoffset=10
    )
    plugins.connect(fig, tooltip_nv_positive)
    tooltip_nv_negative = plugins.PointHTMLTooltip(
        scatter_nv_negative, labels=labels_nv_negative, voffset=10, hoffset=10
    )
    plugins.connect(fig, tooltip_nv_negative)

    plt.tight_layout()

    # Save the interactive plot as an HTML file
    html_path = os.path.join(base_dir, f"{ds}_{query}_tsne.html")
    mpld3.save_html(fig, html_path)
    plt.close(fig)

    # Modify HTML to use relative paths
    with open(html_path, "r") as file:
        html_content = file.read()

    html_content = html_content.replace(f"{base_dir}/", "./")

    with open(html_path, "w") as file:
        file.write(html_content)

    print(f"Interactive t-SNE plot saved as {html_path}")


def ca(
    qVLAD,
    pVLAD,
    nVLAD,
    base_dir,
    ds,
    query,
    clusterNo,
    method_our=None,
    baseline_name=None,
):
    index = 0
    vlad_q_vb, vlad_p_vb, vlad_n_vb = qVLAD[index], pVLAD[index], nVLAD[index]
    vlad_q_nv, vlad_p_nv, vlad_n_nv = (
        qVLAD[index + 1],
        pVLAD[index + 1],
        nVLAD[index + 1],
    )

    # Combine all VLAD descriptors for both methods
    all_vlads_vb = np.vstack([vlad_q_vb, vlad_p_vb, vlad_n_vb])
    all_vlads_nv = np.vstack([vlad_q_nv, vlad_p_nv, vlad_n_nv])

    # Perform KMeans clustering
    kmeans_vb = KMeans(n_clusters=3)
    labels_vb = kmeans_vb.fit_predict(all_vlads_vb)

    kmeans_nv = KMeans(n_clusters=3)
    labels_nv = kmeans_nv.fit_predict(all_vlads_nv)

    # PCA for visualization
    pca_vb = PCA(n_components=2).fit_transform(all_vlads_vb)
    pca_nv = PCA(n_components=2).fit_transform(all_vlads_nv)

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    # vlad-buff
    axes[0].scatter(pca_vb[:, 0], pca_vb[:, 1], c=labels_vb, cmap="viridis")
    axes[0].set_title(f"Cluster Analysis {method_our}")
    axes[0].legend()

    # NV
    axes[1].scatter(pca_nv[:, 0], pca_nv[:, 1], c=labels_nv, cmap="viridis")
    axes[1].set_title(f"Cluster Analysis {baseline_name}")
    axes[1].legend()

    plt.tight_layout()
    # plt.show()
    frame_path = f"{base_dir}/{ds}_{query}_ca.png"
    plt.savefig(frame_path)
    plt.close(fig)


def cs(
    qVLAD,
    pVLAD,
    nVLAD,
    base_dir,
    ds,
    query,
    clusterNo,
    method_our=None,
    baseline_name=None,
):
    index = 0
    vlad_q_vb, vlad_p_vb, vlad_n_vb = qVLAD[index], pVLAD[index], nVLAD[index]
    vlad_q_nv, vlad_p_nv, vlad_n_nv = (
        qVLAD[index + 1],
        pVLAD[index + 1],
        nVLAD[index + 1],
    )

    # Compute cosine similarity
    cosine_sim_pos_vb = cosine_similarity(vlad_q_vb, vlad_p_vb)
    cosine_sim_neg_vb = cosine_similarity(vlad_q_vb, vlad_n_vb)
    cosine_sim_pos_nv = cosine_similarity(vlad_q_nv, vlad_p_nv)
    cosine_sim_neg_nv = cosine_similarity(vlad_q_nv, vlad_n_nv)

    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # vlad-buff - positive
    sns.heatmap(cosine_sim_pos_vb, cmap="coolwarm", ax=axes[0, 0])
    axes[0, 0].set_title(f"Cosine Similarity {method_our} - Query vs Positive")

    # vlad-buff - negative
    sns.heatmap(cosine_sim_neg_vb, cmap="coolwarm", ax=axes[0, 1])
    axes[0, 1].set_title(f"Cosine Similarity {method_our} - Query vs Negative")

    # NV - positive
    sns.heatmap(cosine_sim_pos_nv, cmap="coolwarm", ax=axes[1, 0])
    axes[1, 0].set_title(f"Cosine Similarity {baseline_name} - Query vs Positive")

    # NV - negative
    sns.heatmap(cosine_sim_neg_nv, cmap="coolwarm", ax=axes[1, 1])
    axes[1, 1].set_title(f"Cosine Similarity {baseline_name} - Query vs Negative")

    plt.tight_layout()
    # plt.show()
    frame_path = f"{base_dir}/{ds}_{query}_cs.png"
    plt.savefig(frame_path)
    plt.close(fig)


def plot_pca(
    qVLAD,
    pVLAD,
    nVLAD,
    base_dir,
    ds,
    query,
    pos,
    neg,
    clusterNo,
    method_our=None,
    baseline_name=None,
):
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    index = 0
    vlad_q_vb, vlad_p_vb, vlad_n_vb = qVLAD[index], pVLAD[index], nVLAD[index]
    vlad_q_nv, vlad_p_nv, vlad_n_nv = (
        qVLAD[index + 1],
        pVLAD[index + 1],
        nVLAD[index + 1],
    )

    # Combine all VLAD descriptors for both methods
    all_vlads_vb = np.vstack([vlad_q_vb, vlad_p_vb, vlad_n_vb])
    all_vlads_nv = np.vstack([vlad_q_nv, vlad_p_nv, vlad_n_nv])

    # Perform PCA
    pca = PCA(n_components=2)
    pca_result_vb = pca.fit_transform(all_vlads_vb)
    pca_result_nv = pca.fit_transform(all_vlads_nv)

    # Split PCA results back into query, positive, and negative
    pca_q_vb, pca_p_vb, pca_n_vb = np.split(
        pca_result_vb, [len(vlad_q_vb), len(vlad_q_vb) + len(vlad_p_vb)]
    )
    pca_q_nv, pca_p_nv, pca_n_nv = np.split(
        pca_result_nv, [len(vlad_q_nv), len(vlad_q_nv) + len(vlad_p_nv)]
    )

    # Create relative file paths for images
    image_paths_vb = {
        "query": [
            f"{base_dir}/{method_our}/{ds}_{method_our}_Anchor_{query}_{cluster}.png"
            for cluster in range(len(pca_q_vb))
        ],
        "positive": [
            f"{base_dir}/{method_our}/{ds}_{method_our}_Pos_{pos}_{cluster}.png"
            for cluster in range(len(pca_p_vb))
        ],
        "negative": [
            f"{base_dir}/{method_our}/{ds}_{method_our}_Neg_{neg}_{cluster}.png"
            for cluster in range(len(pca_n_vb))
        ],
    }

    image_paths_nv = {
        "query": [
            f"{base_dir}/{baseline_name}/{ds}_{baseline_name}_Anchor_{query}_{cluster}.png"
            for cluster in range(len(pca_q_nv))
        ],
        "positive": [
            f"{base_dir}/{baseline_name}/{ds}_{baseline_name}_Pos_{pos}_{cluster}.png"
            for cluster in range(len(pca_p_nv))
        ],
        "negative": [
            f"{base_dir}/{baseline_name}/{ds}_{baseline_name}_Neg_{neg}_{cluster}.png"
            for cluster in range(len(pca_n_nv))
        ],
    }

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    # Your method
    scatter_vb_query = axes[0].scatter(
        pca_q_vb[:, 0], pca_q_vb[:, 1], color="r", label="query"
    )
    scatter_vb_positive = axes[0].scatter(
        pca_p_vb[:, 0], pca_p_vb[:, 1], color="g", label="positive"
    )
    scatter_vb_negative = axes[0].scatter(
        pca_n_vb[:, 0], pca_n_vb[:, 1], color="b", label="negative"
    )
    axes[0].set_title(f"PCA of VLAD Descriptors {method_our} ")
    axes[0].legend()

    # baseline_name's method
    scatter_nv_query = axes[1].scatter(
        pca_q_nv[:, 0], pca_q_nv[:, 1], color="r", label="query"
    )
    scatter_nv_positive = axes[1].scatter(
        pca_p_nv[:, 0], pca_p_nv[:, 1], color="g", label="positive"
    )
    scatter_nv_negative = axes[1].scatter(
        pca_n_nv[:, 0], pca_n_nv[:, 1], color="b", label="negative"
    )
    axes[1].set_title(f"PCA of VLAD Descriptors {baseline_name}")
    axes[1].legend()

    def create_tooltip(image_paths, category, num_points):
        labels = []
        for i in range(num_points):
            img_path = image_paths[category][i]
            if os.path.exists(img_path):
                img_html = f'<img src="{img_path}" width="300">'
                labels.append(img_html)
            else:
                labels.append(f"Image not found: {img_path}")
        return labels

    labels_vb_query = create_tooltip(image_paths_vb, "query", len(pca_q_vb))
    labels_vb_positive = create_tooltip(image_paths_vb, "positive", len(pca_p_vb))
    labels_vb_negative = create_tooltip(image_paths_vb, "negative", len(pca_n_vb))

    labels_nv_query = create_tooltip(image_paths_nv, "query", len(pca_q_nv))
    labels_nv_positive = create_tooltip(image_paths_nv, "positive", len(pca_p_nv))
    labels_nv_negative = create_tooltip(image_paths_nv, "negative", len(pca_n_nv))

    tooltip_vb_query = plugins.PointHTMLTooltip(
        scatter_vb_query, labels=labels_vb_query, voffset=10, hoffset=10
    )
    plugins.connect(fig, tooltip_vb_query)
    tooltip_vb_positive = plugins.PointHTMLTooltip(
        scatter_vb_positive, labels=labels_vb_positive, voffset=10, hoffset=10
    )
    plugins.connect(fig, tooltip_vb_positive)
    tooltip_vb_negative = plugins.PointHTMLTooltip(
        scatter_vb_negative, labels=labels_vb_negative, voffset=10, hoffset=10
    )
    plugins.connect(fig, tooltip_vb_negative)

    tooltip_nv_query = plugins.PointHTMLTooltip(
        scatter_nv_query, labels=labels_nv_query, voffset=10, hoffset=10
    )
    plugins.connect(fig, tooltip_nv_query)
    tooltip_nv_positive = plugins.PointHTMLTooltip(
        scatter_nv_positive, labels=labels_nv_positive, voffset=10, hoffset=10
    )
    plugins.connect(fig, tooltip_nv_positive)
    tooltip_nv_negative = plugins.PointHTMLTooltip(
        scatter_nv_negative, labels=labels_nv_negative, voffset=10, hoffset=10
    )
    plugins.connect(fig, tooltip_nv_negative)

    plt.tight_layout()

    # Save the interactive plot as an HTML file
    html_path = f"{base_dir}/{ds}_{query}_pca.html"
    mpld3.save_html(fig, html_path)
    plt.close(fig)

    # Modify HTML to use relative paths
    with open(html_path, "r") as file:
        html_content = file.read()

    html_content = html_content.replace(f"{base_dir}/", "./")

    with open(html_path, "w") as file:
        file.write(html_content)

    print(f"Interactive PCA plot saved as {html_path}")


def extract_method_name(path):
    # Split the path and extract relevant parts
    parts = path.split("/")
    method_name = parts[-2] + "_" + parts[-1].split("_predictions.npz")[0]
    return method_name


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--dataset_name", type=str, help="Name of the dataset")
    parser.add_argument("--method_our", type=str, help="Name of the your method")
    parser.add_argument("--baseline_name", type=str, help="Name of the baseline")
    parser.add_argument(
        "--resize",
        type=int,
        default=[322, 322],
        nargs=2,
        help="Resizing shape for images (HxW).",
    )
    parser.add_argument(
        "--baseline_path",
        type=str,
        help="Paths to the baseline .npz files, e.g., ./wpca8192_last.ckpt_MSLS_predictions.npz",
        required=True,
    )
    parser.add_argument(
        "--your_method_path",
        type=str,
        help="Path to your method .npz file, e.g., ./wpca8192_last.ckpt_MSLS_predictions.npz",
        required=True,
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./logs",
        help="log directory",
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    baseline = extract_method_name(args.baseline_path)
    method = extract_method_name(args.your_method_path)

    # Load CSV file and set image IDs
    predictions = f"{args.log_dir}/comparison/{args.dataset_name}/{method}/{args.dataset_name}_prediction_analysis.csv"
    if not os.path.exists(predictions):
        print("========no prediction file==============")
    df = pd.read_csv(predictions)

    for index, row in df.iterrows():
        if row["Baseline"] == baseline:
            if row["CorrectedByYourMethod"]:
                folder = "correct"
                correct = True
                image_ids = [
                    row["QueryIndex"],
                    row["YourMethodTop1Prediction"],
                    row["BaselineTop1Prediction"],
                ]
            else:
                correct = False
                folder = "incorrect"
                image_ids = [
                    row["QueryIndex"],
                    row["BaselineTop1Prediction"],
                    row["YourMethodTop1Prediction"],
                ]

            print(image_ids)
            query = image_ids[0]
            pos = image_ids[1]
            neg = image_ids[2]

            analysis_dir = f"{args.log_dir}/comparison/{args.dataset_name}/{method}/clustersAnalysis/{folder}/{args.method_our}_{args.baseline_name}/{query}"
            os.makedirs(analysis_dir, exist_ok=True)

            analysis_img_baseline_name_dir = f"{args.log_dir}/comparison/{args.dataset_name}/{method}/clustersAnalysis/{folder}/{args.method_our}_{args.baseline_name}/{query}/{args.baseline_name}"
            os.makedirs(analysis_img_baseline_name_dir, exist_ok=True)

            analysis_img_method_our_dir = f"{args.log_dir}/comparison/{args.dataset_name}/{method}/clustersAnalysis/{folder}/{args.method_our}_{args.baseline_name}/{query}/{args.method_our}"
            os.makedirs(analysis_img_method_our_dir, exist_ok=True)

            clusterNo = None
            qVLAD, q_regions = load_and_visualize(
                image_ids[0],
                args.log_dir,
                args.dataset_name,
                traverse="queries",
                img_type="Anchor",
                coloR="black",
                clusterNo=clusterNo,
                method_our=args.method_our,
                baseline_name=args.baseline_name,
                gif_path="query.gif",
                analysis_dir=analysis_dir,
            )
            pVLAD, p_regions = load_and_visualize(
                image_ids[1],
                args.log_dir,
                args.dataset_name,
                traverse="database",
                img_type="Pos",
                coloR="green",
                clusterNo=clusterNo,
                method_our=args.method_our,
                baseline_name=args.baseline_name,
                gif_path="pos.gif",
                analysis_dir=analysis_dir,
            )
            nVLAD, n_regions = load_and_visualize(
                image_ids[2],
                args.log_dir,
                args.dataset_name,
                traverse="database",
                img_type="Neg",
                coloR="red",
                clusterNo=clusterNo,
                method_our=args.method_our,
                baseline_name=args.baseline_name,
                gif_path="neg.gif",
                analysis_dir=analysis_dir,
            )

            if qVLAD is None or pVLAD is None or nVLAD is None:
                print("Error: One or more VLAD descriptors could not be loaded.")
                continue
            # Compute triplet margins
            margin_vb = compute_triplet_margin(qVLAD[0], pVLAD[0], nVLAD[0])
            # print(f'margin_vb: {margin_vb}')
            margin_other = compute_triplet_margin(qVLAD[1], pVLAD[1], nVLAD[1])
            #  print(f'margin_other: {margin_other}')

            # Rank clusters based on margins
            rank_vb = rank_clusters(margin_vb)
            #   print(f'rank_vb: {rank_vb}')
            rank_other = rank_clusters(margin_other)
            #    print(f'rank_other: {rank_other}')
            # Compute cluster rank differences
            if correct:
                cluster_rank_difference = compute_cluster_rank_difference(
                    rank_vb, rank_other
                )
            else:
                cluster_rank_difference = compute_cluster_rank_difference(
                    rank_other, rank_vb
                )

            # print(f'(rank_other - rank_vb): {cluster_rank_difference[0]}')
            print(
                f"{query}:: Cluster with maximum change in : {cluster_rank_difference[1].item()}"
            )

            clusterNo = cluster_rank_difference[1].item()
            qVLAD, q_regions = load_and_visualize(
                image_ids[0],
                args.log_dir,
                args.dataset_name,
                traverse="queries",
                img_type="Anchor",
                coloR="black",
                method_our=args.method_our,
                clusterNo=clusterNo,
                baseline_name=args.baseline_name,
                gif_path="query.gif",
                analysis_dir=analysis_dir,
            )
            pVLAD, p_regions = load_and_visualize(
                image_ids[1],
                args.log_dir,
                args.dataset_name,
                traverse="database",
                img_type="Pos",
                coloR="green",
                method_our=args.method_our,
                clusterNo=clusterNo,
                baseline_name=args.baseline_name,
                gif_path="pos.gif",
                analysis_dir=analysis_dir,
            )
            nVLAD, n_regions = load_and_visualize(
                image_ids[2],
                args.log_dir,
                args.dataset_name,
                traverse="database",
                img_type="Neg",
                coloR="red",
                method_our=args.method_our,
                clusterNo=clusterNo,
                baseline_name=args.baseline_name,
                gif_path="neg.gif",
                analysis_dir=analysis_dir,
            )

            plot_pca(
                qVLAD,
                pVLAD,
                nVLAD,
                analysis_dir,
                args.dataset_name,
                query,
                pos,
                neg,
                clusterNo,
                method_our=args.method_our,
                baseline_name=args.baseline_name,
            )

            plot_tsne(
                qVLAD,
                pVLAD,
                nVLAD,
                analysis_dir,
                args.dataset_name,
                query,
                pos,
                neg,
                clusterNo,
                method_our=args.method_our,
                baseline_name=args.baseline_name,
            )

            HoD(
                qVLAD,
                pVLAD,
                nVLAD,
                analysis_dir,
                args.dataset_name,
                query,
                clusterNo,
                method_our=args.method_our,
                baseline_name=args.baseline_name,
            )

            HoPD(
                qVLAD,
                pVLAD,
                nVLAD,
                analysis_dir,
                args.dataset_name,
                query,
                clusterNo,
                method_our=args.method_our,
                baseline_name=args.baseline_name,
            )


if __name__ == "__main__":
    main()

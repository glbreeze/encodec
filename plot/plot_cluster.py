import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.colors as mcolors
import random

def load_and_visualize(file_path="codebook_stats.pth", output_dir='./tsne_plots'):
    # Load the saved statistics
    stats = torch.load(file_path)
    embeddings = stats["embeddings"].cpu().numpy()  # [N, D]
    index_map = stats["index_map"]            # str(code_tuple) -> List[int]
    index_map = {eval(k): v for k, v in index_map.items()}
    codebook_embeddings = [cb.numpy() for cb in stats["codebook_embeddings"]]  # List[Tensor[L, D]]

    import pdb; pdb.set_trace()
    first_cluster_num = 100
    subcluster_code_range = 500
    
    # Get first layer codebook embeddings and indices
    first_layer_codes = {k: v for k, v in index_map.items() if len(k) == 1 and k[0]<first_cluster_num}
    first_layer_centers = codebook_embeddings[0][:first_cluster_num]
    
    # Prepare data for first layer visualization
    first_layer_indices = []
    first_layer_labels = []
    for code_tuple, indices in first_layer_codes.items():
        first_layer_indices.extend(indices)
        first_layer_labels.extend([code_tuple[0]] * len(indices))
    
    first_layer_embeddings = embeddings[np.array(first_layer_indices)]

    max_points = 10000
    if len(first_layer_embeddings) > max_points:
        sampled_idx = np.random.choice(len(first_layer_embeddings), size=max_points, replace=False)
        first_layer_embeddings = first_layer_embeddings[sampled_idx]
        first_layer_labels = np.array(first_layer_labels)
        first_layer_labels = first_layer_labels[sampled_idx]
        
    # ===== Plot 1: t-SNE of first layer clusters ===== 
    # plot_tsne_clusters(first_layer_embeddings, 
    #                   first_layer_labels,
    #                   first_layer_centers,
    #                   title="t-SNE Visualization of First Quantization Layer Clusters", 
    #                   output_path=os.path.join(output_dir, "tsne_first_layer.png"))
    
    # ===== Plot 2: Subclusters of 10 random parent codes =====
    random_parent_codes = random.sample(list(first_layer_codes.keys()), min(10, len(first_layer_codes)))
    random_parent_codes = [code[0] for code in random_parent_codes] # convert to int
    
    subcluster_codes = []
    subcluster_centers = []
    subcluster_indices = []
    subcluster_labels = []
    parent_to_subclusters = {}

    for parent_code in random_parent_codes:
        parent_to_subclusters[parent_code] = []
        for sub_code in range(subcluster_code_range):
            key = (parent_code, sub_code)
            subcluster_codes.append(key)
            subcluster_centers.append(codebook_embeddings[0][parent_code] + codebook_embeddings[1][sub_code])
            parent_to_subclusters[parent_code].append(key)

            if key in index_map:
                indices = index_map[key]
                subcluster_indices.extend(indices)
                subcluster_labels.extend([key] * len(indices))
        
    subcluster_embeddings = embeddings[np.array(subcluster_indices)]
    subcluster_centers = np.stack(subcluster_centers)

    # Map labels to IDs including empty clusters
    label_to_id = {label: i for i, label in enumerate(subcluster_codes)}
    label_ids = [label_to_id[label] for label in subcluster_labels]

    # t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    combined_data = np.vstack([subcluster_embeddings, subcluster_centers])
    combined_2d = tsne.fit_transform(combined_data)
    embeddings_2d = combined_2d[:len(subcluster_embeddings)]
    centers_2d = combined_2d[len(subcluster_embeddings):]

    parent_cmap = plt.get_cmap("tab10", len(parent_to_subclusters))
    subcluster_color_map = {}

    for i, (parent, sub_keys) in enumerate(parent_to_subclusters.items()):
        base_rgb = np.array(mcolors.to_rgb(parent_cmap(i)))
        for j, sub_key in enumerate(sub_keys):
            factor = 0.5 + 0.5 * (j / max(1, len(sub_keys) - 1))  # fade scale: 0.5 to 1.0
            faded_color = base_rgb * factor + (1 - factor)
            subcluster_color_map[sub_key] = faded_color
    
    plt.figure(figsize=(15, 10))
    ax = plt.gca()

    # Plot points
    for label, label_id in label_to_id.items():
        mask = np.array(label_ids) == label_id
        color = subcluster_color_map.get(label, "gray")

        if np.any(mask):
            ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], color=color, label=f'{label}', alpha=0.5, s=10)
            ax.scatter(centers_2d[label_id, 0], centers_2d[label_id, 1], color=color, marker='X', s=40)

    ax.set_title("t-SNE of 10 Random Parent Clusters with Subclusters")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "tsne_subclusters.png"))
    plt.close()

        
def plot_tsne_clusters(embeddings, labels, centers, ax=None, title="", show_legend=False, output_path=None):
    # Compute t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    combined_data = np.vstack([embeddings, centers])
    combined_2d = tsne.fit_transform(combined_data)
    
    embeddings_2d = combined_2d[:len(embeddings)]
    centers_2d = combined_2d[len(embeddings):]
    
    # Create plot
    if ax is None:
        plt.figure(figsize=(10, 8))
        ax = plt.gca()
    
    # Plot individual points
    colors = plt.cm.get_cmap('tab20', len(centers))
    
    for label in range(len(centers)):
        mask = np.array(labels) == label
        ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], color=colors(label), label=f'Cluster {label}', alpha=0.4, s=10)
        ax.scatter(centers_2d[label, 0], centers_2d[label, 1], color=colors(label), marker='X', s=30)
    
    ax.set_title(title)
    if show_legend:
        ax.legend()
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        plt.close()

if __name__ == "__main__":
    output_dir = './tsne_plots'
    if not os.path.exists: 
        os.makedirs(output_dir, exist_ok=True)
    load_and_visualize(output_dir=output_dir)
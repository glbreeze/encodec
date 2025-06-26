import os
import torch
import faiss
import numpy as np
import argparse
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
import random

def main(
    folder='/scratch/lg154/sseg/encodec/outputs/2025-06-23/tree_rvq', 
    file_name="codebook_stats_layers1.pth", 
    n_subcluster=1024**2
    ):

    # Load the saved statistics
    file_path = os.path.join(folder, file_name)
    stats = torch.load(file_path)
    
    codebook_embed = stats['codebook_embeddings'][0]
    embeddings = stats["embeddings"].cpu()  # [N, D]
    index_map = {eval(k)[0]: v for k, v in stats["index_map"].items()}
    counts = {k: len(v) for k, v in index_map.items()}
    means = {k[0]: v.cpu().numpy() for k, v in stats["means"].items()}
    variances = {k[0]: v.cpu().numpy() for k, v in stats["variances"].items()}

    total_var = sum(np.sum(v) * counts[k] for k, v in variances.items())
    num_subcluster = {
        k: max(1, round(n_subcluster * np.sum(var) * counts[k] / total_var))
        for k, var in variances.items()
    }

    subcluster_centroids = []

    for k in sorted(variances.keys()):
        indices = index_map.get(k, [])

        emb_subset = embeddings[indices].numpy().astype(np.float32)
        n_clusters = min(num_subcluster[k], len(indices))

        if len(indices) < 2 or n_clusters==1:
            print(f"Skipping small cluster {k}: only add the mean!")
            centroid = emb_subset.mean(axis=0)
            subcluster_centroids.append(torch.from_numpy(centroid))
            continue
        
        print(f"Running k-means for cluster {k}: {len(indices)} samples → {n_clusters} subclusters")
         
        gpu_resources = faiss.StandardGpuResources()
        kmeans = faiss.Kmeans(d=emb_subset.shape[1], k=n_clusters, niter=20, gpu=True)
        kmeans.train(emb_subset)
        subcluster_centroids.append(torch.from_numpy(kmeans.centroids))

    subcluster_centroids = torch.cat(subcluster_centroids, dim=0)
    print(f"Total subclusters: {subcluster_centroids.shape[0]}")

    save_path = os.path.join(folder, "subcluster_centroids.pth")
    torch.save({'subcluster_centroids': subcluster_centroids}, save_path)
    print(f"[Saved] Subcluster centroids saved to {save_path}")

    # # Plot 1: Histogram of counts
    # cluster_keys = sorted([k for k in counts.keys()])
    # count_vals = np.array([counts[k] for k in cluster_keys])
    # sq_dist_vals = np.array([variances[k].sum().item() * counts[k] for k in cluster_keys])
    # mean_centroid_dist = []
    # for k in cluster_keys:
    #     centroid = codebook_embed[k]
    #     dist = torch.norm(means[k] - centroid.to(means[k].device)).item()
    #     mean_centroid_dist.append(dist)
    # mean_centroid_dist = np.array(mean_centroid_dist)

    # fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    # # Plot 1: Count (sorted)
    # sorted_idx = np.argsort(-count_vals)
    # axs[0].bar(range(len(sorted_idx)), count_vals[sorted_idx])
    # axs[0].set_ylabel("Count")
    # axs[0].set_title("Sample Count per Cluster (Sorted)")

    # # Plot 2: Sum of Squared Distance (sorted)
    # sorted_idx = np.argsort(-sq_dist_vals)
    # axs[1].bar(range(len(sorted_idx)), sq_dist_vals[sorted_idx])
    # axs[1].set_ylabel("Sum Sq Dist")
    # axs[1].set_title("Sum of Squared Distances to Cluster Mean (Sorted)")

    # # Plot 3: Mean–Centroid Distance (sorted)
    # sorted_idx = np.argsort(-mean_centroid_dist)
    # axs[2].bar(range(len(sorted_idx)), mean_centroid_dist[sorted_idx])
    # axs[2].set_ylabel("L2 Distance")
    # axs[2].set_title("Distance Between Cluster Mean and Centroid (Sorted)")

    # axs[2].set_xlabel("Sorted Cluster Index")
    # plt.tight_layout()
    # outpath = os.path.join(folder, "sorted_codebook_stats.png")
    # plt.savefig(outpath)
    # plt.close()
    # print(f"Saved combined plot to: {outpath}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_cluster", type=int, default=1048576, help="Number of subclusters")
    args = parser.parse_args()
    main(n_subcluster=args.n_cluster)

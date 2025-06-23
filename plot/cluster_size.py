import os
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import random

def main(output_dir='tsne_plots', file_path="codebook_stats.pth"):
    # Load the saved statistics
    stats = torch.load(file_path)
    embeddings = stats["embeddings"].cpu().numpy()  # [N, D]
    index_map = stats["index_map"]  # str(code_tuple) -> List[int]
    index_map = {eval(k): v for k, v in index_map.items()}

    subcluster_size = {k: len(v) for k, v in index_map.items() if len(k) == 2}

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Write to CSV
    output_file = os.path.join(output_dir, "subcluster_size.csv")
    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["layer1", "layer2", "Size"])
        for layer1 in range(1024): 
            for layer2 in range(1024):
                writer.writerow([layer1, layer2, subcluster_size.get((layer1, layer2), 0)])

    print(f"Subcluster sizes written to {output_file}")

if __name__ == '__main__':
    main(output_dir='tsne_plots')

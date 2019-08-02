import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm_notebook

import mbtools 

# This function is really, REALLY inefficient. It implicitly converts counts_df to dense,
# which is absolutely unacceptable given the size of that dataframe.
# I have to find a way to compute the same statistics better.

def describe_read_counts(counts_df, desc, clustering_df=None, feature_name="SNP"):
    """
    @counts_df — pd.SparseDataframe — raw counts
    @desc — description of counts_df
    @clustering_df — pd.DataFrame — assigns cluster label to each barcode
    
    This function summarizes the information about the raw counts.
    It computes such summary statistics as:
        1. Fraction of NaNs (per barcode)
        2. Average number of reads mapped to SNP (per barcode)
        3. Total number of reads (per barcode)
        4. Number of SNPs with at least one mapped read (per barcode). 
    If clustering information is provided, it also computes the distribution
    of reads across genome, taking cluster structure into account. 
    To do so, it adds up the reads mapping to each SNP (individually) for each 
    barcode in the cluster and then divides the sum by the cluster size.
    """
    
    print("{} columns, {} barcodes, {} {}s".format(
        counts_df.shape[1],
        len(mbtools.extract_barcodes(counts_df)),
        counts_df.shape[0],
        feature_name
    ))

    fig, axes = plt.subplots(2, 2, figsize=(15,10), constrained_layout=True)
    fig.suptitle(desc)
    axes[0,0].set_title("Fraction of NaNs per cell");
    counts_df.isna().mean().hist(ax=axes[0,0]);

    axes[0,1].set_title(f"Reads per {feature_name} (on average) per cell");
    mbtools.extract_counts(counts_df, suffix="dp").mean().hist(ax=axes[0,1]);

    axes[1,0].set_title("Reads per barcode (in total)");
    mbtools.extract_counts(counts_df, suffix="dp").sum().hist(ax=axes[1,0]);

    axes[1,1].set_title(f"{feature_name}s with at least one mapped read (one point per cell)");
    (mbtools.extract_counts(counts_df, suffix="dp") > 0).sum().hist(ax=axes[1,1]);
    fig.show()

    if clustering_df is not None:
        cluster_to_barcodes = mbtools.extract_clusters(clustering_df)
        cluster_label_list = mbtools.extract_cluster_labels(clustering_df)
        print("Cluster labels: ", cluster_label_list)

        n_clusters = len(cluster_label_list)
        grid_shape = (n_clusters // 2 + n_clusters % 2, 2)
        sns.set()
        fig, axes = plt.subplots(*grid_shape, figsize=(15, 2 * n_clusters), constrained_layout=True)
        fig.suptitle("Distribution of reads in the genome (per cluster, DP counts averaged within cluster)")
        for i, label in enumerate(tqdm_notebook(cluster_label_list, desc="processing clusters")):
            ax = axes[i // grid_shape[1], i % grid_shape[1]]
            ax.set_title(f"Cluster label: {label}")
            ax.set_xlabel(f"{feature_name} id")
            ax.set_ylabel("mean DP count")    
            dp_counts = np.zeros(counts_df.shape[0])
            for barcode in cluster_to_barcodes[label]:
                dp_counts = dp_counts + counts_df[f"{barcode}_dp"].to_dense().fillna(0).values
            dp_counts /= len(cluster_to_barcodes[label])
            ax.plot(dp_counts, label=label)
        fig.show()
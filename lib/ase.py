import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm_notebook


def compute_ase(counts_df, barcode_list):
    ase_df = pd.concat([
        pd.DataFrame({
            f"{barcode}" 
            : counts_df[f"{barcode}_ad"].to_dense().fillna(0) 
            / counts_df[f"{barcode}_dp"].to_dense().fillna(0)
        }) 
        for barcode in tqdm_notebook(barcode_list, desc="ase from read counts")], 
        axis=1
    )
    return ase_df


def plot_ase(ase_df, row_cluster=False, title="", figsize=(10,10), **clustermap_kwargs):
    try: 
        sns.clustermap(
            ase_df\
            .fillna(value=0.5)\
            .astype(np.float32)\
            .values,
            cmap="viridis",
            figsize=figsize,
            row_cluster=row_cluster,
            xticklabels=[colname for colname in ase_df.columns 
                         if colname != "GENE_ID"], 
            **clustermap_kwargs
        ).fig.suptitle(title);
    except RecursionError:
        print("RecursionError: increase stack size")
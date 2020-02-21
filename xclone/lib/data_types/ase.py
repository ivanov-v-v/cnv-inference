import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm_notebook


def compute_ase(counts_df, barcode_list, verbose=False):
    if verbose:
        barcode_list = tqdm_notebook(barcode_list, 
                                     "ase from read counts")
    ase_df = pd.concat([
        pd.DataFrame({
            f"{barcode}" 
            : counts_df[f"{barcode}_ad"].to_dense().fillna(0) 
            / counts_df[f"{barcode}_dp"].to_dense().fillna(0)
        }) for barcode in barcode_list
    ], axis=1)
    return ase_df


def plot_ase(
    ase_df, 
    row_cluster=False, 
    title="", 
    xlabel="",
    ylabel="",
    figsize=(10,10), 
    outfile=None, 
    **clustermap_kwargs
):
    try: 
        g = sns.clustermap(
            ase_df\
            .fillna(value=0.5)\
            .astype(np.float32)\
            .values,
            cmap="viridis",
            figsize=figsize,
            row_cluster=row_cluster,
            xticklabels=ase_df.columns,
            yticklabels=False,
            **clustermap_kwargs
        )
        
        g.ax_heatmap.set(xlabel=xlabel, ylabel=ylabel)
        
        for i, label in enumerate(g.ax_heatmap.yaxis.get_ticklabels()):
            if i % 10 != 0: 
                label.set_visible(False)
                
        g.fig.suptitle(title)
        
        if outfile is not None:
            g.fig.savefig(
                outfile, 
                format=outfile.split('.')[-1], 
                dpi=300
            )
    except RecursionError:
        print("RecursionError: increase stack size")
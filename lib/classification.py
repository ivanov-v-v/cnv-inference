import os

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm, tqdm_notebook

import toolkit


class AseMleClassifier:
    def __init__(self, dna_clustering_df, rna_clustering_df, verbose=False):
        self.dna_clustering_df = dna_clustering_df
        self.rna_clustering_df = rna_clustering_df
        self._errstream = sys.stdout if verbose else open(os.devnull, "w+")
        
    def _assign_label(self, barcode):
        loglikelihood = []
        for cluster_label in self.dna_ase_df.columns:
            if cluster_label == "GENE_ID": continue
                
            Nm = self.rna_counts_df[f"{barcode}_ad"]
            Np = self.rna_counts_df[f"{barcode}_dp"] - Nm
            profile = self.dna_ase_df[cluster_label].values
            
            loglikelihood.append(
                (((profile > 0) | (Nm == 0)) * Nm * np.log(profile)).dropna().values.sum()
                + (((profile < 1) | (Np == 0)) * Np * np.log(1 - profile)).dropna().values.sum()
            )
#         print(loglikelihood)
        return np.nanargmax(loglikelihood)
        
    def predict(self, scRNA_counts_df, scDNA_ase_df):
        # Different genes are expressed in the datasets,
        # so we need to find the intersection and drop out the rest
        self.rna_counts_df = scRNA_counts_df
        self.dna_ase_df = scDNA_ase_df
        
        self.labels = np.array(
            [
                self._assign_label(barcode) for barcode in 
                tqdm(toolkit.extract_barcodes(self.rna_counts_df), 
                     desc="barcode processing", file=self._errstream)
            ]
        )

        return self.labels
    
    def classification_report(self, labels, title, outfile=None):
        sns.set(style="whitegrid", font_scale=1.5)
        fig, ax = plt.subplots(2, 1, figsize=(20,25))
        ax[0].set_title("Cluster label assigned by MLE", fontsize=20)
        sns.countplot(
            labels, 
#             palette={
#                 1 : "#3182bd", #"C0",
#                 2 : "#2ca25f", #"C2",
#                 3 : "#feb24c"#"C1"
#             },
            ax=ax[0]
        )
        
        sns.set(style="whitegrid", font_scale=1.5);
    
        ax[1].set_title(title)
        
        sns.scatterplot(
            x="TSNE_1", y="TSNE_2", 
            hue=labels, 
            data=self.rna_clustering_df, 
            legend="full",
#             palette={
#                 1 : "#3182bd", #"C0",
#                 2 : "#2ca25f", #"C2",
#                 3 : "#feb24c"#"C1"
#             },
            ax=ax[1]
        );
        ax[1].legend().get_frame().set_facecolor("white");
        ax[1].legend(frameon=False, bbox_to_anchor=(1,0.5), loc="center left")
        fig.subplots_adjust(right=0.75)
        if outfile is not None:
            fig.savefig(outfile, format=outfile.split('.')[-1], dpi=300)
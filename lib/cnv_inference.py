import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm_notebook

import mbtools


class AseMleClassifier:
    def __init__(self, dna_clustering_df, rna_clustering_df):
        self.dna_clustering_df = dna_clustering_df
        self.rna_clustering_df = rna_clustering_df
        
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
                tqdm_notebook(mbtools.extract_barcodes(self.rna_counts_df), 
                              desc="barcode processing")
            ]
        )

        return self.labels
    
    def classification_report(self, labels, title):
        sns.set(style="whitegrid")
        plt.figure(figsize=(20,10))
        plt.title("Cluster label assigned by MLE", fontsize=20)
        sns.countplot(labels)
        
        sns.set(style="whitegrid", font_scale=1.5);
        plt.figure(figsize=(20,10))
    
        plt.title(title)
        
        sns.scatterplot(
            x="TSNE_1", y="TSNE_2", 
            hue=labels, 
            data=self.rna_clustering_df, 
            palette="jet"
        );
        plt.legend().get_frame().set_facecolor("white");
        plt.legend(frameon=False, bbox_to_anchor=(1,0.5), loc="center left")
        plt.subplots_adjust(right=0.75)
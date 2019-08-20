import os
import sys
import warnings

import numpy as np
from tqdm import tqdm, tqdm_notebook

import toolkit

class AseMleClassifier:
    def __init__(self, ase_template_df, verbose=False):
        self.cluster_labels = ase_template_df.columns
        self.ase_template_df = ase_template_df
        self._logfile = sys.stdout if verbose else open(os.devnull, "w+")
        
    def _assign_label(self, barcode):
        Nm = self._counts_df[f"{barcode}_ad"]
        Np = self._counts_df[f"{barcode}_dp"] - Nm

        loglikelihood = []

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for cluster_label in self.cluster_labels:
                if cluster_label in ["GENE_ID", "CHROM", "POS"]:
                    continue

                profile = self.ase_template_df[cluster_label].values

                maternal_mask = (profile > 0) | (Nm == 0)
                maternal_addend = (
                    maternal_mask
                    * Nm
                    * np.log(profile)
                ).dropna().values.sum()

                paternal_mask = ((profile < 1) | (Np == 0))
                paternal_addend = (
                    paternal_mask
                    * Np
                    * np.log(1 - profile)
                ).dropna().values.sum()

                loglikelihood.append(maternal_addend + paternal_addend)

        return self.cluster_labels[np.nanargmax(loglikelihood)]
        
    def predict(self, counts_df):
        # Different genes are expressed in the datasets,
        # so we need to find the intersection and drop out the rest
        self._counts_df = counts_df
        self.labels = np.array(
            [
                self._assign_label(barcode) for barcode in 
                tqdm(toolkit.extract_barcodes(self._counts_df),
                     desc="barcode processing", file=self._logfile)
            ]
        )

        return self.labels
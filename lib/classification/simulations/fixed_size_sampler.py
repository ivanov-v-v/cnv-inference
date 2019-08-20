from collections import defaultdict

import os
import sys

import numpy as np
import pandas as pd

import scipy.stats as sps

import toolkit

class FixedSizeSampler:
    def __init__(
            self,
            # ASE templates
            ase_template_df: pd.DataFrame,
            # phased read counts (AD and DP)
            counts_df: pd.DataFrame,
            TOTAL_COUNT_DEPTH: int,
            CELLS_PER_PROFILE: int,
            # verbosity flag
            verbose: bool = False
    ):

        self.TOTAL_COUNT_DEPTH = TOTAL_COUNT_DEPTH
        self.CELLS_PER_PROFILE = CELLS_PER_PROFILE

        self.count_profile = toolkit.extract_counts(
            counts_df,
            suffix="dp"
        ).sum(axis=1)
        self.count_profile /= self.count_profile.sum()

        self.ase_template_df = ase_template_df
        self.cluster_labels = ase_template_df.columns
        self._logfile = sys.stdout if verbose else open(os.devnull, "w+")

    def sample(self):
        true_labels = np.hstack([
            np.full(self.CELLS_PER_PROFILE, label, dtype=str)
            for label in self.cluster_labels
        ])

        sampled_counts = defaultdict(list)
        for label in self.cluster_labels:
            ase_profile = self.ase_template_df[label] \
                              .values[:, np.newaxis]
            dp_counts = sps.multinomial.rvs(
                n=self.TOTAL_COUNT_DEPTH,
                p=self.count_profile,
                size=self.CELLS_PER_PROFILE
            ).T.astype(np.float64)
            dp_counts[dp_counts == 0] = np.nan
            sampled_counts["dp"].append(dp_counts)
            ad_counts = dp_counts * ase_profile
            sampled_counts["ad"].append(ad_counts)

        for count_tag in ["ad", "dp"]:
            sampled_counts[count_tag] = pd.DataFrame(
                np.column_stack(sampled_counts[count_tag]),
                columns=np.hstack([
                                      f"{label}_{i}_{count_tag}"
                                      for i in range(self.CELLS_PER_PROFILE)
                                  ] for label in self.cluster_labels)
            )

        result_df = pd.concat([
            sampled_counts["ad"],
            sampled_counts["dp"]
        ], axis=1)
        return result_df, true_labels
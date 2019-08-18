import re

import numpy as np
import pandas as pd
from tqdm import tqdm_notebook


def extract_barcodes(df):
    barcode_list = []
    barcode_pattern = r"(^(A|C|G|T)+-1)|(.*_(ad|dp)$)"
    for colname in df.columns:
        if re.match(barcode_pattern, colname) is not None:
            barcode_list.append('_'.join(colname.split("_")[:-1]))
    return np.unique(barcode_list)


def filter_barcodes(df, keep_list):
    barcode_pattern = r"(^(A|C|G|T)+-1)|(.*_(ad|dp)$)"
    remaining_list = []
    keep_set = set(keep_list)
    for colname in df.columns:
        if re.match(barcode_pattern, colname) is None:
            remaining_list.append(colname)
        else: 
            maybe_barcode = colname.split("_")[0] 
            if maybe_barcode in keep_set:
                remaining_list.append(colname)
    # https://stackoverflow.com/questions/40636514/selecting-pandas-dataframe-column-by-list
    return df[df.columns.intersection(remaining_list)]
        

def extract_snps(df):
    return np.array([
        f"{chrom},{pos}"
        for chrom, pos in 
        tqdm_notebook(df[["CHROM", "POS"]]\
                      .to_dense()\
                      .values\
                      .astype(np.int64), 
                      desc="extracting snps...")
    ])


def extract_counts(df, suffix="dp"):
    barcode_list = extract_barcodes(df)
    return df[[f"{barcode}_{suffix}" for barcode in barcode_list]]


def assert_gene_ordering(df, genome_df):
    true_gene_ids = np.hstack(np.where(genome_df.GENE_ID == gene_id) 
                              for gene_id in tqdm_notebook(df.GENE_ID, "asserting correct gene ordering"))
    # assert correct (strictly ascending) gene ordering in gene_counts_df
    assert np.all(np.diff(true_gene_ids) > 0) 
    
    
def extract_clusters(clustering_df, entry_col="BARCODE"):
    cluster_to_entries = {}
    for label, indices in clustering_df.groupby("LABEL").groups.items():
        cluster_to_entries[label] = clustering_df[entry_col][indices].values
    return cluster_to_entries


def extract_cluster_labels(clustering_df):
    return sorted(clustering_df.LABEL.unique())


""" TODO: check for bugs """
def aggregate_by_barcode_groups(counts_df, clustering_df, verbose=False):
    assert np.all(np.isin(clustering_df.BARCODE, 
                          extract_barcodes(counts_df)))
    
    cluster_to_barcodes = extract_clusters(clustering_df)
    cluster_labels = extract_cluster_labels(clustering_df)
    
    result_df = pd.DataFrame()

    if verbose:
        cluster_labels = tqdm_notebook(cluster_labels, 
                                      desc="adding up read counts in each cluster")
    for label in cluster_labels:
        for suffix in ["ad", "dp"]:
            result_df[f"{label}_{suffix}"] = np.zeros(counts_df.shape[0])    
            na_mask = np.full(counts_df.shape[0], True)
            for barcode in cluster_to_barcodes[label]:
                counts = counts_df[f"{barcode}_{suffix}"].to_dense().values
                na_mask = na_mask & np.isnan(counts)
                result_df[f"{label}_{suffix}"] += np.nan_to_num(counts)
            result_df[f"{label}_{suffix}"][na_mask] = np.nan
    return result_df


def select_by_score(gene_counts_df, scoring_df, smaller_is_better=True, n_out=1000):
    assert np.all(np.isin(["SCORE", "GENE_ID"], scoring_df.columns))
    
    result_df = gene_counts_df.to_dense().merge(scoring_df[["SCORE", "GENE_ID"]])
    result_df.sort_values(by="SCORE", ascending=smaller_is_better, inplace=True)
    result_df = result_df.iloc[:n_out, :]
    result_df.sort_index(inplace=True)
    result_df.drop(columns=["SCORE"], inplace=True)
    return result_df
import numpy as np

import toolkit
from data_types import ase_matrix

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm_notebook

def test_deletion_ase_for_genes(genes_to_test, data, tags):
    required_data = ["genome", "gene_counts", "gene_to_snps", "clustering"]
    assert np.all(np.isin(required_data, list(data.keys()))),\
        f"{np.setdiff1d(required_data, list(data.keys()))} not found in data"
    required_tags = ["data", "sample", "clustering"]
    assert np.all(np.isin(required_tags, list(tags.keys()))),\
        f"{np.setdiff1d(required_tags, list(tags.keys()))} not found in tags"

    for test_gene_name in tqdm_notebook(genes_to_test, "processing genes"):
        print(test_gene_name)
        test_loc = np.ravel(
            np.where(
                data["genome"]["GENE_NAME"].apply(
                    lambda gene_list: test_gene_name in gene_list.split(',')
                )
            )
        )
        if not test_loc:
            print(f"{test_gene_name} is not found in the reference genome (hg19)")
            continue

        test_gene_id = data["genome"]["GENE_ID"].iloc[test_loc].values[0]
        snps_covered = data["gene_to_snps"][test_gene_id]

        if not snps_covered:
            print(f"{test_gene_name} doesn't intersect phased SNPs")
            continue

        unique_labels = data['clustering']['LABEL'].unique()
        unique_labels.sort()
        n_clusters = unique_labels.size

        fig, axes = plt.subplots(
            1, n_clusters,
            figsize=(10 * n_clusters, 10),
            sharey=True
        )
        fig.suptitle(
            f"Gene {test_gene_name}"
            f" (intersects {len(snps_covered)} phased SNPs)\n"
            "ASE ratios,"
            f" (genes, {tags['data']}, {tags['sample']}, {tags['clustering']} clustering)."
            " Colored by DP count.\n",
            weight="bold"
        )
        sns.set(font_scale=1.5)

        for i, label in enumerate(unique_labels):
            ax = axes[i]
            cluster_barcodes = data['clustering'] \
                .query(f"LABEL == {label}")["BARCODE"]
            cluster_counts = data['gene_counts'][
                data['gene_counts'].columns.intersection(
                    np.hstack((
                        ["GENE_ID"],
                        np.hstack([
                                      f"{barcode}_ad",
                                      f"{barcode}_dp"
                                  ] for barcode in cluster_barcodes
                                  )))
                )
            ]
            tmp_df = cluster_counts[
                cluster_counts['GENE_ID'] \
                    .isin([test_gene_id])
            ]
            informative_column_list = [
                [f"{barcode}_ad", f"{barcode}_dp"]
                for barcode in cluster_barcodes
                if (~np.isnan(tmp_df[f"{barcode}_dp"].values)
                    and tmp_df[f"{barcode}_dp"].values > 0)
            ]
            if not informative_column_list:
                print(f"{test_gene_name} is not expressed in cluster {label}")
                continue

            tmp_df = tmp_df[
                tmp_df.columns.intersection(
                    np.hstack(informative_column_list)
                )
            ]
            tmp_ase_df = ase_matrix.compute_ase(
                tmp_df,
                toolkit.extract_barcodes(tmp_df)
            )
            assert np.all(tmp_ase_df <= 1)
            ax.set_title(
                f"Cluster label: {label}. {len(cluster_barcodes)} cells",
                weight="bold"
            )
            ax.set_xlabel("ASE ratio", weight="bold")
            ax.set_ylabel("DP count", weight="bold")
            tmp_counts = toolkit.extract_counts(tmp_df).values.ravel()
            sns.scatterplot(
                ax=ax,
                x=tmp_ase_df.values.T.ravel(),
                y=tmp_counts,
                s=75,
                hue=tmp_counts,
                palette="magma"
            )
            ax.set_xlim(-0.1, 1.1)
            ax.legend().get_frame().set_facecolor("white")
        fig.show()


def test_deletion_ase_for_snps(genes_to_test, data):
    required_data = ["non_merged_genome", "raw_counts", "clustering"]
    assert np.all(np.isin(required_data, list(data.keys()))),\
        f"{np.setdiff1d(required_data, list(data.keys()))} not found in data"

    # vcf is 1-based, bed is 0-based
    on_3_mask = data["raw_counts"]["CHROM"].to_dense() == 3
    snp_on_3_df = data["raw_counts"][["CHROM", "POS"]].to_dense()[on_3_mask]
    snp_on_3_df["POS"] -= 1

    cluster_labels = toolkit.extract_cluster_labels(data["clustering"])

    for gene_name in tqdm_notebook(genes_to_test):
        gene_info = data["non_merged_genome"][
            data["non_merged_genome"]["GENE_NAME"]
            == gene_name
            ]
        if gene_info.index.size == 0:
            print(f"{gene_name} not in index")
            continue

        if gene_info.shape[0] > 1:
            print(f"{gene_name} is ambiguous in hg19")
            continue

        gene_start, gene_end = gene_info[["START", "END"]].values.ravel()

        snp_covered_mask = (
                (snp_on_3_df["POS"] >= gene_start)
                & (snp_on_3_df["POS"] < gene_end)
        )

        if ~np.any(snp_covered_mask):
            print(f"{gene_name} doesn't intersect phased SNPs")
            continue

        clustered_snp_counts_df = toolkit.aggregate_by_barcode_groups(
            data["raw_counts"],
            data["clustering"],
        ).to_dense()[on_3_mask][snp_covered_mask]

        clustered_snp_ase_df = ase_matrix.compute_ase(
            clustered_snp_counts_df,
            cluster_labels,
        )

        fig, axes = plt.subplots(1, 3, figsize=(30, 10))
        fig.suptitle(f"Each dot is a SNP intersecting {gene_name}.",
                     weight="bold")
        for i, label in enumerate(cluster_labels):
            ax = axes[i]
            ax.set_title(
                f"Cluster label: {label}.",
                weight="bold"
            )
            ax.set_xlabel("ASE ratio", weight="bold")
            ax.set_ylabel("DP count", weight="bold")

            ase_ratios = clustered_snp_ase_df[label].values.ravel()
            dp_counts = clustered_snp_counts_df[f"{label}_dp"].values.ravel()

            sns.scatterplot(
                ax=ax,
                x=ase_ratios,
                y=dp_counts,
                s=75,
                hue=dp_counts,
                palette="magma"
            )
            ax.set_xlim(-0.1, 1.1)
            ax.legend().get_frame().set_facecolor("white")
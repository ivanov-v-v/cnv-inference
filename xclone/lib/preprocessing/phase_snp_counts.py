import numpy as np
import xclone.lib.pandas_utils
from tqdm import tqdm

def filter_chromosomes(data, chromosomes_to_keep):
    snp_df = data["raw_snp_counts"][["CHROM", "POS"]].to_dense()
    keep_mask = snp_df[np.isin(snp_df["CHROM"], chromosomes_to_keep)]
    data["raw_snp_counts"] = xclone.lib.pandas_utils.filter_rows_by_mask(data["raw_snp_counts"], keep_mask)
    data["phasing"] = xclone.lib.pandas_utils.filter_by_isin(data["phasing"], "CHROM", chromosomes_to_keep)
    data["blocks"] = xclone.lib.pandas_utils.filter_by_isin(data["blocks"], "CHROM", chromosomes_to_keep)
    return data


def drop_non_phased_snps(data):
    all_snps = xclone.lib.pandas_utils.extract_snps(data["raw_snp_counts"]).to_dense()
    phased_snps = xclone.lib.pandas_utils.extract_snps(data["phasing"]).to_dense()
    keep_mask = np.isin(all_snps, phased_snps)
    data["raw_snp_counts"] = xclone.lib.pandas_utils.filter_rows_by_mask(data["raw_snp_counts"], keep_mask)
    return data


def phase_snp_counts(data):
    counts_df = data["raw_snp_counts"].copy()
    counts_df["PHASE"] = data["phasing"]["PHASE"]
    male_alt = counts_df.PHASE == 1

    print("Ensuring that non-phased SNPs were filtered out")
    assert np.all(np.isin(xclone.lib.pandas_utils.extract_snps(counts_df),
                          xclone.lib.pandas_utils.extract_snps(data["phasing"])))

    for barcode in tqdm(xclone.lib.pandas_utils.extract_barcodes(counts_df), desc="cell_barcode"):
        ad = counts_df[f"{barcode}_ad"].to_dense()
        dp = counts_df[f"{barcode}_dp"].to_dense()
        ad[male_alt] = dp[male_alt].sub(ad[male_alt], fill_value=0)
        counts_df[f"{barcode}_ad"] = ad.to_sparse()

    old_nan_stats = data["raw_snp_counts"].drop(columns=["PHASE"]).isna().mean(axis=0)
    new_nan_stats = counts_df.drop(columns=["PHASE"]).isna().mean(axis=0)

    assert old_nan_stats == new_nan_stats

    print("{:.2f}% of non-missing read counts".format(
        100 * np.mean(~counts_df.iloc[:, 2:].isna().values.astype(bool))
    ))

    data["snp_counts"] = counts_df
    return data
from collections import OrderedDict
from functools import partial
import multiprocessing as mp
from tqdm import tqdm

import numpy as np
import pandas as pd

import xclone.lib.system_utils
import xclone.lib.pandas_utils

def extract_snps(data):
    data["snp"] = data["snp_counts"] \
        [["CHROM", "POS"]] \
        .to_dense()
    return data


def intersect_snps_with_blocks(data, n_jobs=16):
    data["chrom_to_blocks"] = {
        str(chrom): data["blocks"][data["blocks"]["CHROM"] == chrom]
        for chrom in data["blocks"]["CHROM"].unique()
    }

    # Here block coverage for each phased SNP is computed
    # TODO: rewrite this using "bedtools intersect".
    # This part doesn't scale well.

    def _snp_vs_blocks_intersection_handler(snp_tuple):
        chrom, pos = snp_tuple
        # 1-based to 0-based
        pos -= 1  # because CellSNP is 1-based, but .bed files are 0-based
        blocks_on_chrom = data["chrom_to_blocks"].get(str(chrom), None)
        if blocks_on_chrom is None:
            return ""
        mask = ((blocks_on_chrom.START <= pos)
                & (pos < blocks_on_chrom.END))
        return '@'.join(blocks_on_chrom[mask].BLOCK_ID)

    pool = mp.Pool(n_jobs)
    result = pool.map(
        partial(_snp_vs_blocks_intersection_handler, blocks_on_chrom=chrom_to_blocks),
        tqdm(data["snp"].values, "SNP processing")
    )
    pool.close()
    pool.join()

    """ Here the raw results computed in parallel are parsed """
    snp_to_blocks, block_to_snps = {}, OrderedDict()

    # This way we ensure proper block ordering
    for block_id in data["blocks"].BLOCK_ID:
        block_to_snps[block_id] = []

    for i, row in tqdm(enumerate(data["snp"].values), "mapping SNPs to blocks"):
        chrom, pos = row
        snp = f"{chrom},{pos}"
        snp_to_blocks[snp] = result[i].split("@") if result[i] else []
        for block in snp_to_blocks[snp]:
            block_to_snps[block].append(snp)

    data["snp_to_blocks"] = snp_to_blocks
    data["block_to_snps"] = block_to_snps
    return data


def compute_block_counts(data, n_jobs=16):
    data["snp_to_idx"] = {
        snp: i for i, snp in
        tqdm(enumerate(xclone.lib.pandas_utils.extract_snps(data["snp_counts"])),
             "mapping snps to their index numbers by position in the blocks")
    }

    data["block_to_snp_ids"] = {
        block: np.array([data["snp_to_idx"][snp] for snp in snp_list])
        for block, snp_list in tqdm(data["block_to_snps"].items(),
                                    desc="mapping block to SNP ids (for faster row selection)")
    }

    def _block_counts_computation_handler(barcode):
        block_to_ad, block_to_dp = [], []
        # I need to convert these columns to dense format
        # because I need only a subset of rows on each iteration
        ad = np.array(data["snp_counts"][f"{barcode}_ad"])
        dp = np.array(data["snp_counts"][f"{barcode}_dp"])

        # block_to_snps is an OrderedDict, so we can guarantee
        # that all the blocks are processed in a correct order
        for snp_ids in data["block_to_snp_ids"].values():
            if len(snp_ids) > 0:
                dp_sum = np.nansum(dp[snp_ids])
                block_to_dp.append(np.nan if dp_sum == 0 else dp_sum)
                block_to_ad.append(np.nan if dp_sum == 0
                                   else np.nansum(ad[snp_ids]))
            else:
                block_to_ad.append(np.nan)
                block_to_dp.append(np.nan)

        return pd.SparseDataFrame({
            f"{barcode}_ad": block_to_ad,
            f"{barcode}_dp": block_to_dp
        })

    pool = mp.Pool(n_jobs)
    result_list = pool.map(
        _block_counts_computation_handler,
        tqdm(xclone.lib.pandas_utils.extract_barcodes(data["snp_counts"]),
             desc="cell_barcode processing")
    )
    pool.close()
    pool.join()

    data["block_counts"] = pd.concat(result_list, axis=1)
    return data
#!/icgc/dkfzlsdf/analysis/B260/users/v390v/.conda/envs/xclone/bin/python 
# This scipt takes all the input from the $DATA_TMP/N5CC3E folder,
# converts it to pickle files in the format required by the pipeline,
# and then loads everything to the $DATA_TMP/N5CC3E folder and cleans
# up the temporary storage directory.


import os
import warnings
import xclone_config
project_config = xclone_config
os.chdir(project_config.ROOT)

import numpy as np
import pandas as pd
from tqdm import tqdm, tqdm_notebook

import util


sample = "N5CC3E"
modality = "scATAC"
tmp_dir = f"{project_config.TMP}/{sample}/{modality}"
processed_dir = f"{project_config.PROCESSED}/{sample}/{modality}"
os.system(f"mkdir -p {processed_dir}")

phasing_df = pd.read_csv(f"{tmp_dir}/phasing.csv")
util.pickle_dump(phasing_df, f"{processed_dir}/phasing.pkl")
print("Uploaded phasing.pkl")
haplotype_blocks_df = pd.read_csv(f"{tmp_dir}/haplotype_blocks.csv")
util.pickle_dump(haplotype_blocks_df, f"{processed_dir}/haplotype_blocks.pkl")
print("Uploaded haplotype_blocks.pkl")

warnings.simplefilter("ignore", FutureWarning)
chunksize = 2 ** 14
for i in range(3):
    csv_batches = pd.read_csv(f"{tmp_dir}/raw_snp_counts_T{i+1}.csv", chunksize=chunksize)
    raw_snp_counts_df = pd.concat(
        [chunk.to_sparse() for chunk in 
         tqdm(csv_batches, desc=f"reading counts in chunks of size {chunksize}")]
    )
    print(raw_snp_counts_df.shape)
    assert raw_snp_counts_df.shape[0] <= phasing_df.shape[0], "SNPs in the count matrix are not aligned with phased reads"
    util.pickle_dump(raw_snp_counts_df, f"{processed_dir}/raw_snp_counts_T{i+1}.pkl")
    print(f"Uploaded raw_snp_counts_T{i+1}.pkl")

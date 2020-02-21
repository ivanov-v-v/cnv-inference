# This is an interface script for preprocessing of CellSNP's output VCF files.
# Under the hood calls a bash script that does all the heavy processing.

import argparse 
import os

import numpy as np

parser = argparse.ArgumentParser(description="parse the output of CellSNP and filter out non-phased SNPs")
parser.add_argument("raw_vcfgz", type=str, help="raw gzipped vcf file containing allele frequencies for all cells in the sample")
parser.add_argument("phasing_csv", type=str, help="csv file with phasing information")
parser.add_argument("-o", "--out", type=str, default="preprocessed.csv", help="location for the preprocessed csv")
parser.add_argument("-v", "--verbose", action="store_true")
parser.add_argument("-t", "--tmp_dir", type=str, default=".{}_preprocessing".format(np.random.rand()), help="directory to store checkpoints and intermediate results")

args = parser.parse_args()
print("arguments parsed: ", args)

try: 
    os.system(f"mkdir -p {args.tmp_dir}")
    os.system("time bash $XC_SCRIPTS/cellsnp_output_preprocessing/preprocess_counts_vcf.sh {} {} {} {}".format(
        args.raw_vcfgz, 
        args.phasing_csv,
        args.out,
        args.tmp_dir
    ))
finally: 
    os.system(f"rm -rf {args.tmp_dir}")

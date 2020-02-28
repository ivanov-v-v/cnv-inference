import argparse
import functools
import logging
import os
import pandas as pd
import xclone.lib.system_utils
import tqdm

parser = argparse.ArgumentParser(
    description="Parses called CNAs.\n"
                "Splits the calls.tsv-file into:\n"
                "\t- AD counts;\n"
                "\t- DP counts;\n"
                "\t- CNA profiles of individual cells."
)
parser.add_argument("--chisel_outs_dir", type=str, help="CHISEL's output directory")
parser.add_argument("--output_dir", type=str, default="./xclone_input", help="directory to store the parsing results")
raw_colnames_list = "#CHR START END CELL NORM_COUNT COUNT RDR A_COUNT B_COUNT BAF CLUSTER CN_STATE"
parser.add_argument(
    "--usecols", type=str, default=raw_colnames_list,
    help=f"Space-separated list of columns to use (default = all: '{raw_colnames_list}')"
)
clean_colnames_list = "CHROM START END BARCODE NORM_COUNT COUNT RDR RD AD BAF LABEL COPY_STATE"
parser.add_argument(
    "--names", type=str, default=clean_colnames_list,
    help=f"Space-separated list of column names to use (default = all: '{clean_colnames_list}')"
)
parser.add_argument("--save_as_csv", action="store_true")
parser.add_argument("--save_individual_cna", action="store_true")
parser.add_argument("-v", "--verbose", action="store_true")

args = parser.parse_args()

args.chisel_outs_dir = os.path.abspath(args.chisel_outs_dir)
args.output_dir = os.path.abspath(args.output_dir)

args.usecols = args.usecols.split()
args.names = args.names.split()

os.system(f"rm -rf {args.output_dir}")
os.system(f"mkdir -p {args.output_dir}")

logging_level = logging.DEBUG if args.verbose else logging.WARNING
logger = logging.getLogger("calls_parse_logger")
logger.setLevel(logging_level)# create formatter
formatter = logging.Formatter("%(asctime)s\t%(levelname)s\t%(message)s")
# add formatter to ch
# create file handler which logs even debug messages
fh = logging.FileHandler(f'{args.output_dir}/parsing_log.txt')
fh.setLevel(logging_level)
fh.setFormatter(formatter)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)
ch.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(ch)
logger.debug(f"Parsed arguments:\n{args}")

log_assert = functools.partial(xclone.lib.system_utils.log_assert,
                               logger=logger, logger_name="calls_parse_logger")

log_assert(
    len(args.usecols) == len(args.names),
    "Numbers of columns to use differs from the number of names provided:\n"
    f"\t- usecols={args.usecols};"
    f"\t- names={args.names}."
)
required_names = ["CHROM", "START", "END", "BARCODE", "LABEL", "COPY_STATE", "AD", "RD"]
log_assert(
    all([col in args.names for col in required_names]),
    f"String passed as '--names' argument must contain all of the fields from the list: {required_names}"
)


args.input_tsv = os.path.join(args.chisel_outs_dir, "calls.tsv")
calls_df = pd.read_csv(args.input_tsv, sep='\t', usecols=args.usecols)
calls_df.columns = args.names
logger.debug(calls_df.columns)

calls_df["DP"] = calls_df["AD"] + calls_df["RD"]
bins_df = calls_df[["CHROM", "START", "END"]].drop_duplicates().reset_index(drop=True)
bins_df.to_pickle(os.path.join(args.output_dir, "blocks.pkl"))
counts_df = bins_df.copy()
cna_df = bins_df.copy()
baf_df = bins_df.copy()
clustering_df = calls_df[["BARCODE", "LABEL"]].sort_values(by="LABEL").drop_duplicates().reset_index(drop=True)
clustering_df.to_pickle(os.path.join(args.output_dir, "clustering.pkl"))

grouped_by_cell = calls_df.groupby(by="BARCODE")
if args.verbose:
    grouped_by_cell = tqdm.tqdm(grouped_by_cell, desc="processing cells")
for barcode, cell_df in grouped_by_cell:
    logger.debug(barcode)
    cell_df.reset_index(inplace=True, drop=True)
    barcode_10x = f"{barcode}-1"
    counts_df[f"{barcode_10x}_ad"] = cell_df["AD"].copy()
    counts_df[f"{barcode_10x}_dp"] = cell_df["DP"].copy()
    cna_df[barcode_10x] = cell_df["COPY_STATE"].copy()
    baf_df[barcode_10x] = cell_df["BAF"].copy()
    if args.save_individual_cna:
        sc_cna_df = cell_df[["CHROM", "START", "END", "COPY_STATE"]].rename(columns={"COPY_STATE" : "COPY_NUMBER"})
        sc_cna_df.to_csv(os.path.join(args.output_dir, f"xci_{barcode}-1_chisel.csv"), index=False)

counts_df.to_pickle(os.path.join(args.output_dir, "block_counts.pkl"))
cna_df.to_pickle(os.path.join(args.output_dir, "copy_number_aberrations.pkl"))
baf_df.to_pickle(os.path.join(args.output_dir, "baf.pkl"))

if args.save_as_csv:
    bins_df.to_csv(os.path.join(args.output_dir, "blocks.csv"))
    clustering_df.to_csv(os.path.join(args.output_dir, "clustering.csv"), index=False)
    counts_df.to_csv(os.path.join(args.output_dir, "block_counts.csv"), index=False)
    cna_df.to_csv(os.path.join(args.output_dir, "copy_number_aberrations.csv"), index=False)
    baf_df.to_csv(os.path.join(args.output_dir, "baf.csv"), index=False)

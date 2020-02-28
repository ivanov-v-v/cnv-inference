import argparse
import os
from pprint import pprint
import re
from tqdm import tqdm
import pysam
from copy import copy, deepcopy

parser = argparse.ArgumentParser(description="Takes single-cell BAM as an input and appends a cell barcode to it.")
parser.add_argument("--input_bam", type=str, help="BAM file to process.")
parser.add_argument("--cell_barcode", type=str, help="Barcode, nonempty string over 'ACGT' alphabet.")
parser.add_argument("--output_bam", type=str, help="BAM file to store the results.")

args = parser.parse_args()
print(len(args.cell_barcode))
barcode_pattern = re.compile(r"[ACGT]{16}")
assert (barcode_pattern.match(args.cell_barcode) is not None) \
       and (len(args.cell_barcode) == 16),\
    "Cell barcode must be a string comprising of exactly 16 nucleic bases"
args.cell_barcode = "{}-1".format(args.cell_barcode)

for path_attr in ["input_bam", "output_bam"]:
    setattr(args, path_attr, os.path.abspath(getattr(args, path_attr)))

print("Parsed args:")
pprint(vars(args))

n_reads = int(pysam.view("-c", args.input_bam))

with pysam.AlignmentFile(args.input_bam, "rb") as input_bam:
    with pysam.AlignmentFile(args.output_bam, "wb", template=input_bam) as output_bam:
        for read in tqdm(input_bam, total=n_reads, desc="processing reads"):
            read.tags += [("CB", args.cell_barcode)]
            output_bam.write(read)
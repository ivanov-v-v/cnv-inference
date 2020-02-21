#!/usr/bin/bash
# This script takes the haploblock file in the bed format,
# removes the comment lines, drops out irrelevant columns (not in [CHROM, START, END, HAPLOBLOCK]),
# and loads the cleaned up file to $XC_TMP/N5CC3E in csv format

SAMPLE="N5CC3E"
MODALITY="scATAC"
IN_DIR="$XC_RAW/$SAMPLE/$MODALITY"
IN_BED="$IN_DIR/haploblocks_info.bed"
OUT_DIR="$XC_TMP/$SAMPLE/$MODALITY"
mkdir -p $OUT_DIR
OUT_CSV="$OUT_DIR/haplotype_blocks.csv"

cat $IN_BED | pv \
| sed "s/chrom/CHROM/g" \
| sed "s/start/START/g" \
| sed "s/end/END/g" \
| sed "s/phased_SNPs/SNPS_COVERED/g" \
| sed "s/HAPLOBLOCK/BLOCK_ID/g" \
| grep -v "Error" \
| xsv select -d '\t' BLOCK_ID,CHROM,START,END,SNPS_COVERED > $OUT_CSV

xsv index $OUT_CSV

#!/usr/bin/bash
# This script takes the phasing file in the vcf format,
# removes the comment lines, drops out irrelevant columns (neither CHROM nor POS nor PHASE),
# and loads the cleaned up file to $DATA_TMP/N5CC3E in csv format

SAMPLE="N5CC3E"
MODALITY="scATAC"
IN_DIR="$XC_RAW/$SAMPLE/$MODALITY"
IN_VCF="$IN_DIR/phasing.vcf"
OUT_DIR="$XC_TMP/$SAMPLE/$MODALITY"
mkdir -p $OUT_DIR
OUT_CSV="$OUT_DIR/phasing.csv"

cat $IN_VCF | pv \
| sed "s/^#//g" | grep -v "^#" \
| sed "s/K08K-N5CC3E_control1/PHASE/g" \
| sed "s/0|1/0/g" | sed "s/1|0/1/g" \
| xsv select -d '\t' CHROM,POS,PHASE > $OUT_CSV

xsv index $OUT_CSV

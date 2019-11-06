#!/usr/bin/bash

SAMPLE="N5CC3E"
MODALITY="scATAC"
RAW_DIR="$XC_RAW/$SAMPLE/$MODALITY"
TMP_DIR="$XC_TMP/$SAMPLE/$MODALITY"
PHASING_CSV="$TMP_DIR/phasing.csv"

for i in {1..3} 
do 
    IN_VCFGZ="$RAW_DIR/cellSNP/$SAMPLE-T$i/cellSNP.cells.vcf.gz"
    OUT_CSV="$TMP_DIR/raw_snp_counts_T"$i".csv"
    python $XC_SCRIPTS/cellsnp_output_preprocessing/preprocess_cellsnps_output.py $IN_VCFGZ $PHASING_CSV --out $OUT_CSV &
done
wait

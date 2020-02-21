#!/usr/bin/bash
# This scripts drops out all the SNPs that are not phased

DATA_CSV=$1
PHASING_CSV=$2
OUT_DIR=$3

xsv select CHROM,POS $DATA_CSV > "$OUT_DIR/raw_snps.txt" &
xsv select CHROM,POS $PHASING_CSV | sed -n '2,$p' > "$OUT_DIR/phased_snps.txt" &
wait
grep -Fn -f "$OUT_DIR/phased_snps.txt" "$OUT_DIR/raw_snps.txt" > "$OUT_DIR/common_snps.txt"
awk -F ':' '{print $1"p"}' "$OUT_DIR/common_snps.txt" > "$OUT_DIR/match_ids.txt"
sed -nf "$OUT_DIR/match_ids.txt" $DATA_CSV > filtered_$DATA_CSV

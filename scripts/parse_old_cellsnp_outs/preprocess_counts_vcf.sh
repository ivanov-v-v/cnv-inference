#!/usr/bin/bash
# This script orchestrates the entire preprocessing pipeline
# It starts with unpacking and cleaning up the .vcf.gz file outputted by CellSNP

THIS_SCRIPT_DIR="$XC_SCRIPTS/cellsnp_output_preprocessing"
IN_VCFGZ=$1
PHASING_CSV=$2
OUT_CSV=$3
TMP_DIR=$4
INTERIM_CSV="$TMP_DIR/preprocessed.csv"

echo "initiating phase 1..."
echo "$TMP_DIR"

gunzip -c $IN_VCFGZ | pv \
| sed "s/^#//g" | grep -v "^#" \
| sed -E "s/(A|G|C|T)+-1/&_ad,&_dp/g" \
| sed "s/.:.:.:.:.:./,/g" \
| sed -r "s/^chr//g" \
| python "$THIS_SCRIPT_DIR/extract_counts.py" \
| xsv select -d '\t' !ALT,FILTER,FORMAT,ID,INFO,REF,QUAL \
| sed 's/\"//g' \
> $INTERIM_CSV

#| sed "s/.:.:.:.:.:./,/g" \        
#| sed -r "s/\"([^:\"]+\/[^:\"]+):([^:\"]+):([0-9]+):([0-9]+):([0-9]+):(([0-9]+\,){4}[0-9]+)\"/\3,\4/g" \    
#| sed -E "s/([A-Z]|[a-z]|[0-9]|\.|\_)+.bam/&_ad,&_dp/g" \
 
printf "dropped useless columns (ALT, FILTER, FORMAT, ID, REF, QUAL);\nreplaced .:.:.:.:.:. with empty strings\n"

echo "initiating phase 2..."

xsv select CHROM,POS $INTERIM_CSV > "$TMP_DIR/raw_snps.txt" &
xsv select CHROM,POS $PHASING_CSV > "$TMP_DIR/phased_snps.txt" &
wait

grep -Fxn -f "$TMP_DIR/phased_snps.txt" "$TMP_DIR/raw_snps.txt" \
| pv | awk -F ':' '{print $1}' > "$TMP_DIR/match_ids.txt"
g++ --std=c++11 "$THIS_SCRIPT_DIR/select_by_index.cpp" -o "$THIS_SCRIPT_DIR/select_by_index.o"
"$THIS_SCRIPT_DIR/select_by_index.o" $INTERIM_CSV  $TMP_DIR/match_ids.txt $OUT_CSV
xsv index $OUT_CSV
echo "dropped out non-phased SNPs"


#!/usr/bin/bash
# This script orchestrates the entire preprocessing pipeline
# It starts with unpacking and cleaning up the .vcf.gz file outputted by CellSNP
IN_VCFGZ=$1
OUT_CSV=$2

time gunzip -c $IN_VCFGZ \
| sed "s/.:.:.:.:.:./,/g" \
| sed "s/^#//g" | grep -v "^#" \
| xsv select -d '\t' !FILTER,FORMAT,ID,INFO,QUAL \
| sed -E "s/(A|G|C|T)+-1/&_ad,&_dp/g" \
| sed -r "s/\"([^:\"]+\/[^:\"]+):([^:\"]+):([0-9]+):([0-9]+):([0-9]+):(([0-9]+\,){4}[0-9]+)\"/\3,\4/g" \
| sed 's/\"//g' \
> $OUT_CSV

xsv index $OUT_CSV

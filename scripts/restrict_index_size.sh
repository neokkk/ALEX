#!/bin/bash

set -e

root=$(dirname $(dirname $(realpath $0)))
echo "Root: $root"

datasets="books"
# datasets="books history face osm"
benchmark="$root/build/benchmark"
output_file="./out_osm.txt"

for dataset in $datasets; do
    file="$root/resources/$dataset"
    $benchmark --keys_file=$file --keys_file_type=binary \
        --init_num_keys=10000000 --total_num_keys=200000000 \
        --batch_size=1000000 --index_size=16476643716 \
        --insert_frac=1 --lookup_distribution=uniform > $output_file
done

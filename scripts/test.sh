#!/bin/bash

set -x

root=$(dirname $(dirname $(realpath $0)))
echo "Root: $root"

datasets="books"
benchmark="$root/build/benchmark"

for dataset in $datasets; do
    file="$root/datasets/$dataset"
    output_file="./out_${dataset}.txt"
    $benchmark --keys_file=$file --keys_file_type=binary \
        --init_num_keys=10000000 --total_num_keys=20000000 \
        --batch_size=10000 --insert_frac=1 > $output_file
done

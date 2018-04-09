#!/bin/bash

sim_lengths=(500 1000 10000 50000)
cd data
dirs=$(echo */)
cd ..

for dir in $dirs; do
  for len in ${sim_lengths[@]}; do
    wd=data/"$dir"$len
    echo "processing $wd"
    python3 time_series_3d.py --working_dir $wd
  done
  echo "plotting $dir"
  python3 4merge_pdf.py --working_dir data --treatment $dir
done

mkdir -p summary
cp data/*/OUTPUT_*.pdf ./summary

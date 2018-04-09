#!/bin/bash

sim_lengths=(500 1000 10000 20000 50000)
RPATH="exec"
cd $RPATH
dirs=$(echo $RPATH/*/)
cd ..
if command -v bsub >/dev/null 2>&1; then
    # bsub is present
    prefix=( bsub -R "rusage[mem=40000]" -W 1:00 -J MERGE_$dir )
else
  prefix=()
fi

for dir in $dirs; do
  echo "plotting $dir"
  command=( "${prefix[@]}" # combine arrays
            python3 4merge_pdf.py --working_dir ./ --treatment $dir )
  "${command[@]}"
done

# mkdir -p summary
# cp $RPATH/*/OUTPUT_*.pdf ./summary/

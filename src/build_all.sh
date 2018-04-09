#!/bin/bash

##############################################################################################
# How to run:                                                                                #
# First of all create the working directory and create the files arrays.sh and parameters.sh #
# Examples of these files can be found in this directory.                                    #
# Then enter the directory src/ and execute this script with argument the working directory  #
# example: cd src/; ./build_all.sh ../working_dir                                            #
##############################################################################################

if [ $# -eq 0 ]
then
    echo "No arguments supplied"
    exit 1
fi
WD="$1"

if [ -d "$WD" ]; then
    echo "Destination directory found $WD"
else
  echo "Copying params to $WD"
  mkdir -p $WD
  cp ./arrays.sh $WD/
  cp ./parameters_example.sh $WD/parameters.sh
fi

count=0
TDIR=exec/
ARGS="-j 10"
mkdir -p obj


# remove old results
if [ -d "$WD/$TDIR" ]; then
    echo "Previous results found, moving them to directory $WD/old_$TDIR"
    rm -fr "$WD/old_$TDIR"
    mv "$WD/$TDIR" "$WD/old_$TDIR"
fi

# activate python
#if [ -d env ]; then
#    echo "activating venv"
#    source env/bin/activate
#else
#  if command -v virtualenv; then
#      echo "creating venv"
#      virtualenv -p python3 env
#     source env/bin/activate
#      pip install pandas matplotlib
#  else
#    echo "Warning, unable to set up python virtualenv: command not found."
#    echo "Continuing in the hope that global python supports the required modules"
# fi
#fi

# read parameters
source "$WD"/arrays.sh
oIFS=$IFS
IFS=';' read -ra array <<< "$I_array"
IFS=';' read -ra ENAMES <<< "$I_ENAMES"
IFS=';' read -ra EFLAG <<< "$I_EFLAG"
IFS=';' read -ra PNAMES <<< "$I_PNAMES"
IFS=$oIFS

#check if the variables in arrays.sh are not empty
if [ -z "$array" ]; then printf "array is unset"; exit 1; else echo "array is set to:"; for i in "${!array[@]}"; do echo ${array[$i]}; done; fi
if [ -z "$ENAMES" ]; then printf "ENAMES is unset"; exit 1; else echo "ENAMES is set to:"; for i in "${!ENAMES[@]}"; do echo ${ENAMES[$i]}; done; fi
if [ -z "$EFLAG" ]; then printf "EFLAG is unset"; exit 1; else echo "EFLAG is set to:"; for i in "${!EFLAG[@]}"; do echo ${EFLAG[$i]}; done; fi
if [ -z "$PNAMES" ]; then printf "PNAMES is unset"; exit 1; else echo "PNAMES is set to:"; for i in "${!PNAMES[@]}"; do echo ${PNAMES[$i]}; done; fi

echo ""
echo "## Starting"
echo ""

for i in "${!ENAMES[@]}"; do
  exec_name=${ENAMES[$i]}
  exec_name=$(tr -dc '[[:print:]]' <<< "$exec_name")

  echo "----------------"
  echo $exec_name
  echo "----------------"

  params=${PNAMES[$i]}
  params=$(tr -dc '[[:print:]]' <<< "$params")
  #########################################START
  var=${array[$i]}
  var=$(tr -dc '[[:print:]]' <<< "$var")
  CEFLAG=${EFLAG[$i]}
  CEFLAG=$(tr -dc '[[:print:]]' <<< "$CEFLAG")
  make clean
  echo $WD'/'$TDIR$count\_$exec_name
  mkdir -p $WD'/'$TDIR$count\_$exec_name

  rm -rf $WD'/'$TDIR$count\_$exec_name/$exec_name # remove old executable
  CMND='make '$ARGS' '$exec_name' OUTNAME='$WD'/'$TDIR$count\_$exec_name'/'$exec_name" VARS=\""$var" "$CEFLAG"\""
  echo ""
  echo "### COMPILING ###"
  echo $CMND
  eval $CMND
  cp $WD/$params $WD/$TDIR$count\_$exec_name/$params
  echo ""
  echo "### SPAWNING SIMULATIONS ###"
  CMND='./run_simple.sh '$WD' '$TDIR$count\_$exec_name' '$exec_name' '$params
  echo $CMND
  eval $CMND
  # merge plots in one
  if command -v bsub >/dev/null 2>&1; then
      # bsub is present
      prefix=( bsub -R "rusage[mem=1000]" -W 1:00 -J MERGE_$count -w "done(analysis_*)" )
      python=(python3.3)
  else
    prefix=( )
    python=(python3)
  fi

  echo ""
  echo "### MERGE PDFs ###"


  command=( "${prefix[@]}" # combine arrays
            "${python[@]}" 4merge_pdf.py --working_dir $WD/$TDIR$count\_$exec_name --treatment $count)
  echo "${command[@]}"
  "${command[@]}"

  echo ""
  echo "### MERGE measures ###"
  command=( "${prefix[@]}" # combine arrays
            "${python[@]}" 4merge_measures.py --working_dir $WD/$TDIR$count\_$exec_name  --treatment $count)
  echo "${command[@]}"
  "${command[@]}"
  count=$((count+1))
  #    #########################################END
  # done
done

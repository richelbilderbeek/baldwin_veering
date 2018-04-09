#!/bin/bash

#####################################################################
# Give as argument the working directory, the counter, the name of  #
# the executable and the name of the file containing the parameters #
#####################################################################

## LOAD MODULES
##COMPILING
##$  module load gcc
##$  module load open_mpi
##EXECUTING
##$  module load open_mpi

date
whoami

#lengths=(500 1000)
lengths=(100 200 500 600 700 800 900 1000 1500 3000 10000)

max_age=1000000
life_bonus=0.997
max_food_in_cell=10
tot_num_food=40


#unchanged
start_num_agents=100
max_agents=200
sim_length=20000
samples=100

field_size=20
fov_radius=3
food_proportion=1.0
skill0_level_0=0.5
food_energy=0.01
social_ratio=0

seed_iteration=-1
famine_iteration=-1
save_iters=( ) #( --save-pop 900 1000 1100 1500 )
load_dir="" # "../working_dir"
load_pop=( ) # ( 1000 )
stable_after=( )
load_logic=""
measures=( --measures )

W=4
#Threads
P=6

if [ $# -ne 4 ]
then
    echo "Wrong number of arguments supplied"
else
  # read parameters from file
  executable=$3
  WD=$1/$2
  echo "loading parameters from $1/$4"
  source "$1/$4"
fi

MAX_C_N=24 #Max number of cores per node
#Processes
M=$samples
#Total
N=$(($P*$M))
NPN=$(($MAX_C_N/$P)) #Add checks to make sure it is integer i.e. floor?
NTILE=$(($P*$NPN))
echo Total $N per node $NPN

export OMP_NUM_THREADS=$P

src_wd=$(pwd)
cd $WD
for season_length in "${lengths[@]}"; do

  echo ""
  echo "### SEASON LENGTH ### "$season_length" #### "
  echo ""

  mkdir -p $season_length
  cp $executable ./$season_length/
  cd $season_length
  mkdir -p results
  export OMP_NUM_THREADS=$P
  # check if bsub is present
  if command -v bsub >/dev/null 2>&1; then
      # bsub is present
      bnameT=$2$season_length
      bname=$(printf "%s" "$bnameT" | sed 's=\/==g'| sed 's=\\==g')
      echo $bname
      prefix=(bsub -R rusage[mem=2000] -W $W:00 -J $bname -n $N -R "span[ptile="$NTILE"]" )
      prefix2=(bsub -W 4:00 -J zip_"$bname" -w "done($bname)" )
      prefix3=(bsub -W 4:00 -n 1 -R "rusage[mem=100000]" -J analysis_$bname -w "done(zip_$bname)")
      #compiler=( "unset LSB_AFFINITY_HOSTFILE ; mpirun -n $M --map-by socket:PE=$P" )
      compiler=( "unset LSB_AFFINITY_HOSTFILE ; mpirun -np $M --npernode $NPN --cpus-per-proc $P" )
      python=(python3.3)
  else
    export TMPDIR=/tmp            # workaround for a bug on OSX
    prefix=()
    prefix2=()
    prefix3=()
    compiler=(mpirun -n $M)
    python=(python3)
  fi
  # build up the load path
  if [ ${#load_pop[@]} -eq 0 ]; then
      echo "initializing population randomly"
  else
    data_dir="$src_wd/$1/$load_dir/results/"
    if [ -d "$data_dir" ]; then
        echo "Loading population from directory $data_dir"
        load_param=( --load-pop "$data_dir/pop_dump_${load_pop[0]}" --load-logic $load_logic )
        stable_after=( --stable_after 0 )
    else
      echo "Warning, Directory $data_dir not found. Ignoring it and initializing population randomly"
      load_param=( )
    fi
  fi
  command=( "${prefix[@]}" # combine arrays
            "${compiler[@]}" ./$executable -n $start_num_agents -f $tot_num_food -s $season_length -S $samples -l $sim_length -v $fov_radius -F $food_proportion -P $skill0_level_0 --field-size $field_size --food-energy $food_energy --max-pop $max_agents --food-qty $max_food_in_cell --max-age $max_age --seed-iter $seed_iteration --famine-iter $famine_iteration --social-ratio $social_ratio --life-bonus $life_bonus
            "${save_iters[@]}" "${load_param[@]}")
  echo ""
  echo "### Running simulations ###"
  echo "${command[@]}"

  "${command[@]}"
  export OMP_NUM_THREADS=1
  command2=( "${prefix2[@]}" bzip2 results/*)

  echo ""
  echo "### Compressing Results ###"
  echo "${command2[@]}"

  "${command2[@]}"
  # run analysis
  command3=( "${prefix3[@]}" "${python[@]}" "$src_wd/time_series_3d.py" --working_dir ./ --season_len $season_length "${stable_after[@]}" "${measures[@]}")

  echo ""
  echo "### Performing Analysis ###"
  echo "${command3[@]}"

  "${command3[@]}"
  cd ..
done

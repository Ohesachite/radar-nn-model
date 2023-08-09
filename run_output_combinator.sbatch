#!/bin/bash

#SBATCH --job-name=RadarNN
#SBATCH --account=gts-ks207
#SBATCH --nodes=1 
#SBATCH --gres=gpu:V100:2
#SBATCH --mem-per-gpu=12G
#SBATCH --time=08:00:00
#SBATCH -qembers
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=alanliu2@gatech.edu

argsstring=""

trainmode=false

while [[ $# -gt 0 ]]; do
  case $1 in
    -m|--model-paths)
      argsstring+="--sw-model-path=$2 --deci-model-path=$3 "
      shift
      shift
      shift
      ;;
    -h|--hyper)
      argsstring+="--momentum=$2 --int-layers=$3 --lr=$4 "
      shift
      shift
      shift
      shift
      ;;
    -trd|--train-data)
      argsstring+="--train-path=$2 "
      shift
      shift
      ;;
    -ted|--test-data)
      argsstring+="--test-path=$2 "
      shift
      shift
      ;;
    -o|--output)
      argsstring+="--result-file=$2 "
      shift
      shift
      ;;
  esac
done

echo "$argsstring"

cd $HOME/scratch/radar-nn-model

module load pytorch
module load python/3.9

srun python train-output-combinator.py $argsstring
#!/bin/bash

#SBATCH --job-name=RadarP4
#SBATCH --account=gts-ks207
#SBATCH --nodes=1 
#SBATCH --gres=gpu:V100:2
#SBATCH --time=08:00:00
#SBATCH -qinferno
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=alanliu2@gatech.edu

argsstring=""

trainmode=false

while [[ $# -gt 0 ]]; do
  case $1 in
    -b|--batch-size)
      argsstring+="-b=$2 "
      shift
      shift
      ;;
    -t)
      trainmode=true
      shift
      ;;
    -st|--store)
      argsstring+="--output-dir=$2 "
      shift
      shift
      ;;
    -ld|--load)
      argsstring+="--resume=$2 "
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
    -e|--milestones)
      argsstring+="--epochs=$2 --lr-warmup-epochs=$3 "
      shift
      shift
      shift
      haslrm=false
      while [[ $# -gt 0 ]]; do
        dr=false
        case $1 in
          -*|--*)
            dr=true
            ;;
          *)
            if [ "$haslrm" = false ]; then
              haslrm=true
              argsstring+="--lr-milestones "
            fi
            argsstring+="$1 "
            shift
            ;; 
        esac
        if [ "$dr" = true ]; then
          break
        fi
      done
      ;;
  esac
done

echo "$argsstring"

cd $HOME/scratch/radar-nn-model

module load pytorch
module load python/3.9

srun python train-msr.py $argsstring

#!/bin/bash

#SBATCH --job-name=IN5550
#SBATCH --mail-type=FAIL
#SBATCH --account=nn9447k
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=4G

# Increase this number when you really need parallel computing
# (don't set it to more than 6 or 8 cores):
#SBATCH --ntasks-per-node=6

source /cluster/bin/jobsetup
set -o errexit

module purge
module use -a /projects/nlpl/software/modulefiles/
module add nlpl-in5550

echo $SUBMITDIR

#'sentenceEmbedding', 'convNet', 'BiLSTM', 'LastStateEncoder'

echo "train classififer"
python runner_abel.py --dir $SUBMITDIR --encoder LastStateEncoder --name lastState_LSTM

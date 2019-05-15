#!/bin/bash

#SBATCH --job-name=IN5550
#SBATCH --mail-type=FAIL
#SBATCH --account=nn9447k
#SBATCH --time=07:45:00
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
module purge
module load nlpl-gensim nlpl-pytorch

echo $SUBMITDIR

#'sentenceEmbedding', 'convNet', 'BiLSTM', 'LastStateEncoder'

echo "train sentence embedding classififer with batch size $1 "
python ../runner_abel.py --dir $SUBMITDIR --name "sentenceE mbedding_batch_$1" --batch_size $1


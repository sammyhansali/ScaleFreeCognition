#!/bin/bash

# Run it like this for file in Experiments/MMDDYYYY/*; do    bash submit_jobs.sh $file; done
FILE_PATH=$1
FILE_NAME=$(basename "$FILE_PATH" .py)
CURRENT_DATE=$(date +"%Y/%b/%d")
mkdir -p /cluster/tufts/levinlab/shansa01/SFC/jobs_logs/$CURRENT_DATE

sbatch << EOT
#!/bin/sh
#SBATCH -J "$FILE_NAME"
#SBATCH --time=07-00:00:00 #requested time (DD-HH:MM:SS)
#SBATCH -p preempt
#SBATCH -N 1
#SBATCH -n 70
#SBATCH --mem=32g
#SBATCH --output="/cluster/tufts/levinlab/shansa01/SFC/jobs_logs/$CURRENT_DATE/${FILE_NAME}.out"
#SBATCH --error="/cluster/tufts/levinlab/shansa01/SFC/jobs_logs/$CURRENT_DATE/${FILE_NAME}.err"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sh1436@scarletmail.rutgers.edu

## commands to run
module load anaconda/2021.11
source activate mesamultineat
cd /cluster/tufts/levinlab/shansa01/SFC/ScaleFreeCognition

## Test 1
# No pos, No fit inputs
for i in {1..20}
do
	python "$FILE_PATH"
done

EOT
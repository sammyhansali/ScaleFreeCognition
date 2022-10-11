#!/bin/sh
#SBATCH -J multicellularity_random   #job name
#SBATCH --time=00-23:00:00  #requested time (DD-HH:MM:SS)
#SBATCH -p batch    #running on "mpi" partition/queue
#SBATCH -N 1    #1 nodes
#SBATCH -n 1   #2 tasks total
#SBATCH -c 40    #1 cpu cores per task
#SBATCH --mem=32g  #requesting 2GB of RAM total
#SBATCH --output=MyJob.%j.%N.out  #saving standard output to file, %j=JOBID, %N=NodeName
#SBATCH --error=MyJob.%j.%N.err   #saving standard error to file, %j=JOBID, %N=NodeName
#SBATCH --mail-type=ALL    #email optitions
#SBATCH --mail-user=leo.lopez@tufts.edu

#[commands_you_would_like_to_exe_on_the_compute_nodes]
# for example, running a python script# 1st, load the module
module load anaconda/2021.05
source activate mesa_20210916
# run python<<
python run.py #make sure myscript.py exists in the current directory

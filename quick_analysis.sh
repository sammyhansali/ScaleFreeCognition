file=$(find /cluster/tufts/levinlab/shansa01/ScaleFreeCognition/SCS_Results/ -name "$1*")
echo $file
cd $file

module load anaconda/2021.11
source activate mesamultineat
python analysis.py


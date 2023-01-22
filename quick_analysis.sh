file=$(find /cluster/tufts/levinlab/shansa01/ScaleFreeCognition/SCS_Results/ -name "$1*")
echo $file
cd $file

python analysis.py


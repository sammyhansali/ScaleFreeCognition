file_to_copy="/cluster/tufts/levinlab/shansa01/ScaleFreeCognition/multicellularity/model_for_analysis.py"
# file_to_copy="/cluster/tufts/levinlab/shansa01/ScaleFreeCognition/multicellularity/schedule.py"
# directories=("path/to/dir1" "path/to/dir2" "path/to/dir3")
# pass in 'ls SCS_results/2023-01-06_*/multicellularity'

## "$@" lets you pass in any number of files, and it will iterate through them.
for dir in "$@"
do
  cp "$file_to_copy" "$dir"
done
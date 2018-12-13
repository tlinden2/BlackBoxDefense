#!/bin/bash
if [ $# -lt 2 ]
then
	echo "Usage: $(basename $0) output_dir ckpts_dir"
	exit 1
fi
output_dir=$1
result_dir="$output_dir/data" # dir of originial data
labels_file="$2/conv_actions_labels.txt"
# graph_file="$2/conv_actions_frozen.pb"
graph_file="$2/conv.ckpt-18000" # dir of ckpts

python3 evaluate_realdata_accuracy.py --output_dir=$result_dir --labels_file=$labels_file --graph_file=$graph_file

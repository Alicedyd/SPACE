#!/bin/bash

# Use the current directory if not specified
current_dir=${1:-.}

# Find directories with maxdepth=1 that match the pattern
find "$current_dir" -maxdepth 1 -type d -name "checkpoints_*" | while read -r dirc; do
    echo "Processing directory: $dirc"
    python combine_validation_results.py -i "$dirc" -o "$dirc/combine.csv"
    echo "Created $dirc/combine.csv"
done

echo "All matching directories processed."
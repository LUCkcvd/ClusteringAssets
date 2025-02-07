#!/bin/bash

# Check that the Python script exists in the current directory
if [ ! -f clusterFormatting.py ]; then
    echo "Error: clusterFormatting.py was not found in $(pwd)"
    exit 1
fi

# Recursively traverse all subdirectories starting from the current directory
find . -type d | while read -r dir; do
    # Check if result_lines.log exists in this subdirectory
    if [ -f "$dir/result_lines.log" ]; then
        echo "Processing $dir/result_lines.log in directory $dir"
        python3 clusterFormatting.py "$dir/result_lines.log" "$dir"
    fi
done


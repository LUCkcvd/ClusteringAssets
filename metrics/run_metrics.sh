#!/bin/bash
# This script recursively searches the "log" directory for any clusters.txt file
# and runs the metrics_distributed.py Python script on it. The output directory is
# set to be the same directory that contains the clusters.txt file.

# Check that the "log" directory exists
if [ ! -d "log" ]; then
    echo "Error: 'log' directory does not exist."
    exit 1
fi

# Use find to locate all clusters.txt files in subdirectories of log.
find log -type f -name "clusters.txt" | while read -r clusters_file; do
    # Get the directory containing the clusters.txt file
    output_dir=$(dirname "$clusters_file")
    
    echo "------------------------------"
    echo "Found clusters file: $clusters_file"
    echo "Output directory: $output_dir"
    
    # Run the metrics_distributed.py script with the clusters file as the first parameter
    # and the output directory (same as the clusters file location) as the second parameter.
    python3 metrics_distributed.py "$clusters_file" "$output_dir"
    
    # Optionally, check for errors.
    if [ $? -ne 0 ]; then
        echo "ERROR: metrics_distributed.py failed for clusters file: $clusters_file"
    else
        echo "Successfully processed: $clusters_file"
    fi
done


#!/bin/bash

# Input and output file paths are provided as arguments
INPUT_FILE="$1"
OUTPUT_FILE="$2"

# Check if the input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file $INPUT_FILE does not exist."
    exit 1
fi

# Process the input file and save to the output file
awk '{gsub(/std::vector of length/, "\nstd::vector of length"); print}' "$INPUT_FILE" > "$OUTPUT_FILE"


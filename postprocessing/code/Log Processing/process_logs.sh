#!/bin/bash

# Define the base log directory
BASE_LOG_DIR="log"

# Get the directory where this script resides
SCRIPT_DIR="$(dirname "$(realpath "$0")")"

# Paths to the scripts
PROCESSING_SCRIPT="$SCRIPT_DIR/resultLogProcessing.sh"
PYTHON_SCRIPT="$SCRIPT_DIR/nodeToTitle.py"
DBLP_TITLES_FILE="$SCRIPT_DIR/DBLPtitles.txt"

# Ensure the scripts are executable
chmod +x "$PROCESSING_SCRIPT"

# Iterate over all results.log files in the log directory
find "$BASE_LOG_DIR" -name "results.log" | while read -r RESULTS_FILE; do
    # Get the parent directory of the results.log file
    PARENT_DIR=$(dirname "$RESULTS_FILE")
    
    # Define output files
    RESULT_LINES_FILE="$PARENT_DIR/result_lines.log"
    JOURNAL_TITLE_FILE="$PARENT_DIR/JournalTitleClusters.txt"
    
    echo "Processing results.log in $PARENT_DIR"
    
    # Step 1: Run the resultLogProcessing.sh script
    if [ -f "$PROCESSING_SCRIPT" ]; then
        bash "$PROCESSING_SCRIPT" "$RESULTS_FILE" "$RESULT_LINES_FILE"
        if [ $? -ne 0 ]; then
            echo "Error running resultLogProcessing.sh in $PARENT_DIR"
            continue
        fi
        echo "Processed results.log to $RESULT_LINES_FILE"
    else
        echo "Error: Processing script not found at $PROCESSING_SCRIPT"
        continue
    fi
    
    # Step 2: Run the nodeToTitle.py script
    if [ -f "$PYTHON_SCRIPT" ]; then
        if [ -f "$DBLP_TITLES_FILE" ]; then
            python3 "$PYTHON_SCRIPT" "$RESULT_LINES_FILE" "$DBLP_TITLES_FILE" "$JOURNAL_TITLE_FILE"
            if [ $? -ne 0 ]; then
                echo "Error running nodeToTitle.py in $PARENT_DIR"
                continue
            fi
            echo "Processed $RESULT_LINES_FILE to $JOURNAL_TITLE_FILE"
        else
            echo "Error: DBLPtitles.txt not found at $DBLP_TITLES_FILE"
        fi
    else
        echo "Error: Python script not found at $PYTHON_SCRIPT"
    fi
done

echo "Processing complete."


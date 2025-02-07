#!/usr/bin/env python3
import sys
import os

def parse_and_format_clusters(input_file, output_file):
    """
    Parses clustering results in the given input file and formats them into the desired output format.
    :param input_file: File containing clustering results in the provided format.
    :param output_file: File to save the formatted clusters.
    """
    try:
        with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
            # Skip the first two lines
            next(infile)
            next(infile)

            cluster_id = 1  # Start with cluster ID 1

            for line in infile:
                # Look for lines containing vector elements
                if "std::vector of length" in line:
                    # Extract the cluster members from the line content between '{' and '}'
                    start = line.find('{') + 1
                    end = line.find('}')
                    if start == 0 or end == -1:
                        continue
                    members = line[start:end].split(',')
                    members = [member.strip() for member in members if member.strip()]

                    # Write each member with its cluster ID
                    for member in members:
                        outfile.write(f"{cluster_id} {member}\n")
                    
                    cluster_id += 1  # Increment the cluster ID for the next cluster

    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Ensure the proper number of command line arguments was provided
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} inputfile desiredoutputdirectory")
        sys.exit(1)
    
    # Get command line arguments
    input_file = sys.argv[1]
    output_directory = sys.argv[2]

    # Validate the input file
    if not os.path.exists(input_file):
        print(f"Error: The input file '{input_file}' does not exist.")
        sys.exit(1)
    
    # Validate that the output directory exists
    if not os.path.isdir(output_directory):
        print(f"Error: The directory '{output_directory}' does not exist.")
        sys.exit(1)

    # Define the output file path
    output_file = os.path.join(output_directory, "clusters.txt")

    # Parse and format the clusters
    parse_and_format_clusters(input_file, output_file)
    print(f"Formatted clusters written to {output_file}")


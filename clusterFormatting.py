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
                    # Extract the cluster members from the next line
                    start = line.find('{') + 1
                    end = line.find('}')
                    members = line[start:end].split(',')  # Extract members
                    members = [member.strip() for member in members if member.strip()]  # Clean members

                    # Write each member with its cluster ID
                    for member in members:
                        outfile.write(f"{member} {cluster_id}\n")

                    cluster_id += 1  # Increment the cluster ID for the next cluster

    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    # Input file in the original format
    input_file = "result_lines.log"
    # Output file in the desired format
    output_file = "clusters.txt"

    # Convert and format the clusters
    parse_and_format_clusters(input_file, output_file)
    print(f"Formatted clusters written to {output_file}")


import sys
import re

# Function to load the ID to description mapping from DBLPtitles.txt
def load_id_description_mapping(filename):
    id_to_description = {}
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                id_to_description[parts[0]] = parts[1]
            else:
                print(f"Warning: Could not parse line: {line.strip()}")
    return id_to_description

# Function to process the vectors file
def replace_and_sort_by_length_desc(vector_filename, mapping, output_filename):
    clusters = []
    with open(vector_filename, 'r') as f:
        for idx, line in enumerate(f):
            ids_match = re.search(r'\{([0-9, ]+)\}', line)
            if ids_match:
                ids_str = ids_match.group(1)
                ids = [id.strip() for id in ids_str.split(',')]
                clusters.append((idx, ids))
            else:
                print(f"Warning: Could not extract IDs from line: {line.strip()}")

    clusters_with_descriptions = []
    for idx, cluster in clusters:
        descriptions = []
        for id in cluster:
            descriptions.append(mapping.get(id, id))  # Use ID if no description
        clusters_with_descriptions.append((idx, descriptions))

    clusters_sorted = sorted(clusters_with_descriptions, key=lambda x: len(x[1]), reverse=True)

    with open(output_filename, 'w') as output_file:
        for _, (original_idx, cluster) in enumerate(clusters_sorted):
            cluster_size = len(cluster)
            output_file.write(f'Cluster {original_idx}, size {cluster_size}\n')
            for description in cluster:
                output_file.write(f'{description}\n')
            output_file.write('\n')

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 nodeToTitle.py <input_vector_file> <mapping_file> <output_file>")
        sys.exit(1)

    vector_file = sys.argv[1]
    mapping_file = sys.argv[2]
    output_file = sys.argv[3]

    id_description_mapping = load_id_description_mapping(mapping_file)
    replace_and_sort_by_length_desc(vector_file, id_description_mapping, output_file)


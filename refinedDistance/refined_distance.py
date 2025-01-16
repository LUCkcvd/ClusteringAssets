import random  # Import the random module to generate random numbers

# Step 1: Read edges from the file in the same directory
with open('undirect_graph.txt', 'r') as f:  # Open the file containing edges in read mode
    edges = [line.strip().split() for line in f]  # Read each line, strip whitespace, and split into a list of node pairs

# Generate random weights and probabilities for each edge
weighted_edges = []  # Initialize an empty list to store edges with weights and probabilities
for edge in edges:  # Iterate through each edge
    u, v = edge  # Unpack the nodes from each edge
    weight = random.uniform(0.1, 1.0)  # Generate a random weight between 0.1 and 1.0
    probability = random.uniform(0.1, 1.0)  # Generate a random probability between 0.1 and 1.0
    weighted_edges.append((u, v, weight, probability))  # Append the edge along with its weight and probability to the list


# Save the weighted edges to a new file in the same directory
with open('weighted_undirect_graph.txt', 'w') as f:  # Open a file in write mode to store the weighted edges
    for u, v, weight, probability in weighted_edges:  # Iterate through each weighted edge
        f.write(f"{u} {v} {weight:.2f} {probability:.2f}\n")  # Write the nodes, weight, and probability to the file formatted to two decimal places

# Step 2: Function to compute refined distance
def compute_refined_distance(edges):
    refined_edges = []  # Initialize an empty list to store edges with refined distances
    for edge in edges:  # Iterate through each edge
        u, v, weight, probability = edge  # Unpack the nodes, weight, and probability from each edge
        
        # Calculate expected closeness (C_uv)
        C_uv = (1 / weight) * probability  # Compute the closeness using the inverse of the weight multiplied by the probability
        
        # Calculate refined distance (d'_uv)
        refined_distance = 1 / C_uv if C_uv != 0 else float('inf')  # Compute the refined distance, ensuring no division by zero
        
        # Store the refined distance back to the edge
        refined_edges.append((u, v, weight, probability, refined_distance))  # Append the edge along with its refined distance to the list

    return refined_edges  # Return the list of edges with refined distances

# Read the weighted edges from the file in the same directory
with open('weighted_undirect_graph.txt', 'r') as f:  # Open the file containing weighted edges in read mode
    edges = [line.strip().split() for line in f]  # Read each line, strip whitespace, and split into a list of components
    edges = [(u, v, float(weight), float(probability)) for u, v, weight, probability in edges]  # Convert weights and probabilities to float

# Compute refined distances
refined_edges = compute_refined_distance(edges)  # Call the function to compute refined distances

# Save the refined edges to a new file in the same directory
with open('refined_weighted_undirect_graph.txt', 'w') as f:  # Open a file in write mode to store the refined edges
    for u, v, weight, probability, refined_distance in refined_edges:  # Iterate through each refined edge
        f.write(f"{u} {v} {refined_distance:.2f}\n")  # Write the nodes, and refined distance to the file formatted to two decimal places

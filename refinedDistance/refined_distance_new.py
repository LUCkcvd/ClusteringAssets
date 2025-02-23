import random  # Import the random module to generate random numbers

# ----------------------------
# Set seed for reproducibility
# ----------------------------
random.seed(25)

# Step 1: Read edges from the file in the same directory
with open('undirect_graph.txt', 'r') as f:  # Open the file containing edges in read mode
    edges = [line.strip().split() for line in f]  # Read each line, strip whitespace, and split into a list of node pairs

# Define minimum value to prevent division by zero
MIN_VALUE = 0.000001

# Generate random weights (original distance d) and probabilities (p) for each edge
weighted_edges = []  # Initialize an empty list to store edges with weights and probabilities
for edge in edges:
    u, v = edge
    # Generate non-zero random numbers in (0,1]
    d = 1 -random.uniform(0,1)
    p = 1 - random.uniform(0,1) 
    weighted_edges.append((u, v, d, p))

# Save the weighted edges to a new file in the same directory
with open('weighted_undirect_graph.txt', 'w') as f:
    for u, v, d, p in weighted_edges:
        f.write(f"{u} {v} {d:.2f} {p:.2f}\n")

# Step 2: Function to compute refined metrics
def compute_refined_metrics(edges):
    """
    Given edges with original distance (d) and probability (p), this function computes
    refined distances using closeness measure:
    
    C_uv = (1/d) * p
    refined_distance = 1/C_uv
    
    Returns a list of edges with computed refined distances.
    """
    refined_edges = []
    for u, v, d, p in edges:
        # Ensure values are at least MIN_VALUE
        safe_d = max(d, MIN_VALUE)
        safe_p = max(p, MIN_VALUE)
        
        # Calculate closeness with safe values
        C_uv = (1/safe_d) * safe_p
        
        # Calculate refined distance with protection against zero
        refined_distance = 1/max(C_uv, MIN_VALUE)
        
        # Normalize the refined distance to be between 0.1 and 1
        refined_distance = max(0.1, min(1.0, refined_distance/10.0))
        
        refined_edges.append((u, v, refined_distance))
    return refined_edges

# Read the weighted edges from the file
with open('weighted_undirect_graph.txt', 'r') as f:
    edges = [line.strip().split() for line in f]
    edges = [(u, v, float(d), float(p)) for u, v, d, p in edges]

# Compute refined metrics
refined_edges = compute_refined_metrics(edges)

# Save the final adjusted weight (w') to a new file in node1 node2 weight formatting
with open('refined__graph.txt', 'w') as f:
    for u, v, w_prime in refined_edges:
        f.write(f"{u} {v} {w_prime:.2f}\n")

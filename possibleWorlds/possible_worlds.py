import random
import time
import os

def load_edges(file_path):
    edges = []
    with open(file_path, 'r') as file:
        for line in file:
            u, v, weight, probability = line.strip().split()
            edges.append((int(u), int(v), float(weight), float(probability)))
    return edges

def generate_worlds(edges, num_worlds=1000, seed=42, output_dir="worlds"):
    # Ensure reproducibility
    random.seed(seed)
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    start_time = time.time()
    
    for world_index in range(num_worlds):
        world = []
        for u, v, weight, probability in edges:
            if random.random() < probability:
                world.append((u, v, weight))
        
        # Write the current world to its own file
        world_filename = os.path.join(output_dir, f"world_{world_index + 1}.txt")
        with open(world_filename, 'w') as file:
            for u, v, weight in world:
                file.write(f"{u} {v} {weight}\n")
        
        # Calculate and display estimated time remaining
        elapsed_time = time.time() - start_time
        avg_time_per_world = elapsed_time / (world_index + 1)
        time_remaining = avg_time_per_world * (num_worlds - world_index - 1)
        print(f"World {world_index + 1}/{num_worlds} generated. Estimated time remaining: {time_remaining:.2f} seconds.", end='\r')
    
    print("\nAll worlds generated.")

# Usage example
edges = load_edges("weighted_undirect_graph.txt")
generate_worlds(edges)


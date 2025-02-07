def analyze_nodes():
    nodes = set()
    min_node = float('inf')
    max_node = -float('inf')
    
    with open("graph.txt", "r") as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                u, v, _ = line.split()
                # Convert to integers for numeric comparison
                u, v = int(u), int(v)
                nodes.add(u)
                nodes.add(v)
                min_node = min(min_node, u, v)
                max_node = max(max_node, u, v)
    
    print(f"Total unique nodes: {len(nodes)}")
    print(f"Node ID range: {min_node} to {max_node}")
    print(f"This means there are {len(nodes)} unique nodes, with IDs spanning from {min_node} to {max_node}")
    print("\nFirst few nodes (if available):")
    sorted_nodes = sorted(list(nodes))[:5]
    for node in sorted_nodes:
        print(node)

if __name__ == "__main__":
    analyze_nodes()

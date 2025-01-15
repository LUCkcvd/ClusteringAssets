import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import random

def parse_graph(file_path):
    """
    Parse the graph file to build the adjacency matrix.
    :param file_path: Path to the graph file (format: node1 node2 weight).
    :return: NetworkX graph object.
    """
    G = nx.Graph()
    with open(file_path, 'r') as f:
        for line in f:
            u, v, weight = line.strip().split()
            G.add_edge(int(u), int(v), weight=float(weight))
    return G

def parse_clusters(file_path):
    """
    Parse the clusters file to map nodes to their clusters.
    :param file_path: Path to the clusters file (format: node clusterID).
    :return: 
        - clusters: Dictionary where keys are cluster IDs and values are lists of nodes.
        - node_to_cluster: Dictionary mapping each node to its cluster ID.
    """
    clusters = defaultdict(list)
    node_to_cluster = {}
    with open(file_path, 'r') as f:
        for line in f:
            node, cluster = line.strip().split()
            node = int(node)
            cluster = int(cluster)
            clusters[cluster].append(node)
            node_to_cluster[node] = cluster
    return clusters, node_to_cluster

def visualize_graph(G, clusters, node_to_cluster, sample_size=None, title="Graph Visualization"):
    """
    Visualize a graph with clusters highlighted in different colors.
    :param G: NetworkX graph.
    :param clusters: Dictionary of cluster_id -> list of nodes.
    :param node_to_cluster: Dictionary mapping nodes to their cluster IDs.
    :param sample_size: Number of nodes to sample for visualization. If None, use the full graph.
    :param title: Title for the graph visualization.
    """
    # Sample the graph if requested
    if sample_size and sample_size < G.number_of_nodes():
        sampled_nodes = random.sample(list(G.nodes()), sample_size)  # Convert to list for sampling
        G = G.subgraph(sampled_nodes)

        # Filter clusters based on sampled nodes
        clusters = {
            cluster_id: [node for node in nodes if node in G.nodes()]
            for cluster_id, nodes in clusters.items()
        }
        node_to_cluster = {node: cluster_id for node, cluster_id in node_to_cluster.items() if node in G.nodes()}

    # Create a color map for clusters
    cluster_ids = list(clusters.keys())
    cluster_colors = {cluster_id: plt.cm.tab20(i % 20) for i, cluster_id in enumerate(cluster_ids)}

    # Assign colors to nodes based on their cluster
    node_colors = [cluster_colors[node_to_cluster[node]] if node in node_to_cluster else "gray" for node in G.nodes()]

    # Draw the graph
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, seed=42)  # Positioning algorithm
    nx.draw(
        G,
        pos,
        node_color=node_colors,
        with_labels=True,
        node_size=50,
        font_size=8,
        edge_color="lightgray",
    )

    # Add a legend for clusters
    for cluster_id, color in cluster_colors.items():
        plt.scatter([], [], color=color, label=f"Cluster {cluster_id}")
    plt.legend(loc="upper right", fontsize="small", title="Clusters", ncol=2)

    plt.title(title)
    plt.axis("off")
    plt.show()

# Example usage
graph_file = "graph.txt"
clusters_file = "clusters.txt"

# Parse the graph and clusters
G = parse_graph(graph_file)
clusters, node_to_cluster = parse_clusters(clusters_file)

# Visualize the graph (optionally sample for large graphs)
visualize_graph(G, clusters, node_to_cluster, sample_size=500, title="Sampled Graph Visualization")


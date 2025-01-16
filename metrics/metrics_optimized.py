import sys
import os
import numpy as np
from collections import defaultdict
from itertools import combinations
import mmap

def read_graph_lazy(filename):
    """Memory-efficient graph reading using memory mapping."""
    nodes = set()
    edge_count = 0
    
    # First pass: count edges and collect nodes
    with open(filename, "r") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        for line in iter(mm.readline, b""):
            line = line.decode().strip()
            if not line or line.startswith("#"):
                continue
            u, v, _ = line.split()
            nodes.add(u)
            nodes.add(v)
            edge_count += 1
        mm.close()
    
    # Create arrays for efficient storage with proper dtypes
    edges = np.zeros((edge_count, 3), dtype=np.float64)  # More efficient than object dtype
    nodes_list = sorted(nodes)
    node_to_idx = {node: idx for idx, node in enumerate(nodes_list)}
    
    # Second pass: fill arrays
    idx = 0
    with open(filename, "r") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        for line in iter(mm.readline, b""):
            line = line.decode().strip()
            if not line or line.startswith("#"):
                continue
            u, v, w = line.split()
            edges[idx, 0] = node_to_idx[u]
            edges[idx, 1] = node_to_idx[v]
            edges[idx, 2] = float(w)
            idx += 1
        mm.close()
    
    return edges, nodes_list, node_to_idx

def normalize_weight(w):
    """Normalize weight to [0,1) range using vectorized operations."""
    return 0.99 * (1 / (1 + np.exp(-w)))

def build_adjacency_dict(edges):
    """Build adjacency dictionary for O(E) space complexity."""
    adj_dict = defaultdict(dict)
    for u_idx, v_idx, w in edges:
        w_norm = normalize_weight(w)
        weight = 1.0 / (1.0 - w_norm)
        # Add both directions for undirected graph
        adj_dict[int(u_idx)][int(v_idx)] = weight
        adj_dict[int(v_idx)][int(u_idx)] = weight
    return adj_dict

def compute_delta_dprime_matrix(edges, n_nodes):
    """Compute delta_dprime matrix using adjacency dictionary for O(E) complexity."""
    return build_adjacency_dict(edges)

def compute_w_of_u(adj_dict, node):
    """Compute w(u) for a node using adjacency dictionary."""
    return sum(adj_dict.get(node, {}).values())

def read_clusters(filename, node_to_idx):
    """Read clusters with numeric indices, handling missing nodes."""
    clusters = defaultdict(set)
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cID, member = line.split()
            if member in node_to_idx:  # Only add nodes that exist in graph
                clusters[cID].add(node_to_idx[member])
    return clusters

def compute_AVU_dprime(Ci, Cj, adj_dict):
    """Compute AVU'' using adjacency dictionary for O(E) complexity."""
    numerator = 0.0
    sum_w_Ci = 0.0
    sum_w_Cj = 0.0
    
    # Compute numerator and Ci sum
    for u in Ci:
        u_edges = adj_dict.get(u, {})
        sum_w_Ci += sum(u_edges.values())
        for v in Cj:
            if v in u_edges:
                numerator += u_edges[v]
    
    # Compute Cj sum
    for v in Cj:
        sum_w_Cj += sum(adj_dict.get(v, {}).values())
    
    denom = max(1.0, sum_w_Ci) + max(1.0, sum_w_Cj)
    return numerator / denom if denom != 0 else 0.0

def compute_AVI_dprime(Ci, adj_dict, n_nodes):
    """Compute AVI'' using adjacency dictionary for O(E) complexity."""
    sum_intra = 0.0
    sum_out = 0.0
    
    for u in Ci:
        u_edges = adj_dict.get(u, {})
        w_u = sum(u_edges.values())
        if w_u == 0:
            continue
            
        # Compute intra-cluster sum
        for v in Ci:
            if v in u_edges:
                sum_intra += u_edges[v]
        
        # Compute out-cluster sum
        out_sum = sum(weight for v, weight in u_edges.items() if v not in Ci)
        sum_out += out_sum / w_u if w_u > 0 else 0
    
    denom = sum_intra + sum_out
    return sum_intra / denom if denom != 0 else 0.0

def main():
    print("[1/7] Starting QANUI calculation...")
    
    print("[2/7] Reading 'graph.txt'...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    graph_path = os.path.join(script_dir, "graph.txt")
    edges, nodes_list, node_to_idx = read_graph_lazy(graph_path)
    n_nodes = len(nodes_list)
    print(f"    Found {n_nodes} total nodes.")
    
    print("[3/7] Reading 'clusters.txt'...")
    clusters_path = os.path.join(script_dir, "clusters.txt")
    clusters = read_clusters(clusters_path, node_to_idx)
    clusterIDs = sorted(clusters.keys())
    print(f"    Found {len(clusterIDs)} clusters.")
    
    print("[4/7] Building adjacency dictionary...")
    adj_dict = compute_delta_dprime_matrix(edges, n_nodes)
    
    print("[5/7] Computing all AVU'' and AVI''...")
    AVU_values = {}
    AVI_values = {}
    
    # Compute AVU'' for all pairs
    for i, j in combinations(range(len(clusterIDs)), 2):
        Ci = clusters[clusterIDs[i]]
        Cj = clusters[clusterIDs[j]]
        val = compute_AVU_dprime(Ci, Cj, adj_dict)
        AVU_values[(clusterIDs[i], clusterIDs[j])] = val
        AVU_values[(clusterIDs[j], clusterIDs[i])] = val
    
    # Compute diagonal AVU values
    for i in range(len(clusterIDs)):
        Ci = clusters[clusterIDs[i]]
        val = compute_AVU_dprime(Ci, Ci, adj_dict)
        AVU_values[(clusterIDs[i], clusterIDs[i])] = val
    
    # Compute AVI'' for each cluster
    for cID in clusterIDs:
        Ci = clusters[cID]
        val = compute_AVI_dprime(Ci, adj_dict, n_nodes)
        AVI_values[cID] = val
    
    print("[6/7] Computing final metrics...")
    
    if len(clusterIDs) == 1:
        cID = clusterIDs[0]
        finalAVI = AVI_values[cID]
        finalAVU = AVU_values[(cID, cID)]
    else:
        finalAVI = np.mean(list(AVI_values.values()))
        
        # Average AVU over distinct pairs
        avu_values = [AVU_values[(clusterIDs[i], clusterIDs[j])]
                     for i, j in combinations(range(len(clusterIDs)), 2)]
        finalAVU = np.mean(avu_values) if avu_values else 0.0
    
    if finalAVU == 0.0:
        finalQANUI = float("inf")
    else:
        finalQANUI = ((2.0 * finalAVI * (1.0 / finalAVU)) /
                      (finalAVI + (1.0 / finalAVU))) / 2
    
    print("[7/7] Results:\n")
    
    print("===== PARTIAL AVU''(C_i, C_j) Results =====")
    for i in range(len(clusterIDs)):
        for j in range(i, len(clusterIDs)):
            val = AVU_values[(clusterIDs[i], clusterIDs[j])]
            print(f"  AVU''({clusterIDs[i]}, {clusterIDs[j]}) = {val:.5f}")
    print()
    
    print("===== AVI''(C_i) Results =====")
    for cID in clusterIDs:
        val = AVI_values[cID]
        print(f"  AVI''({cID}) = {val:.5f}")
    print()
    
    print("===== FINAL METRICS =====")
    print(f"  Final AVI'' = {finalAVI:.5f}")
    print(f"  Final AVU'' = {finalAVU:.5f}" if finalAVU != float('inf')
          else "  Final AVU'' = Inf")
    if finalQANUI == float("inf"):
        print("  Final QANUI' = Infinity (division by zero in 1/AVU)")
    else:
        print(f"  Final QANUI' = {finalQANUI:.5f}")
    
    print("\nDone.")

if __name__ == "__main__":
    main()

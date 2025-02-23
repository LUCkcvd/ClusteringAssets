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
    
    # Enable garbage collection for better memory management
    import gc
    gc.enable()
    
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
    """Normalize weight to [0,1) range using vectorized operations with numerical stability."""
    # Clip weights to prevent overflow/underflow in exp
    w_clipped = np.clip(w, -100, 100)
    # Use 0.98 instead of 0.99 to ensure better numerical stability
    return 0.98 * (1 / (1 + np.exp(-w_clipped)))

def build_adjacency_dict(edges):
    """Build adjacency dictionary for O(E) space complexity with numerical stability."""
    adj_dict = defaultdict(dict)
    for u_idx, v_idx, w in edges:
        w_norm = normalize_weight(w)
        # Add numerical stability check
        if w_norm >= 0.99:  # Prevent division by numbers too close to 1
            weight = 100.0  # Cap the maximum weight
        else:
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
    if len(Ci) == 0 or len(Cj) == 0:
        return 0.0
        
    same_cluster = Ci == Cj
    numerator = 0.0
    
    # Calculate total possible connections for normalization
    total_connections = len(Ci) * len(Cj) if not same_cluster else (len(Ci) * (len(Ci) - 1)) // 2
    if total_connections == 0:
        return 0.0
    
    # For each node in Ci
    for u in Ci:
        u_edges = adj_dict.get(u, {})
        if not u_edges:
            continue
            
        # Get total weight for normalization
        w_u = sum(u_edges.values())
        if w_u < 1e-10:
            continue
            
        # For same cluster, only count each pair once and normalize by total possible connections
        if same_cluster:
            for v in Cj:
                if v > u and v in u_edges:  # Only count pairs once
                    numerator += (u_edges[v] / w_u) / total_connections
        else:
            # Sum normalized weights to nodes in Cj
            u_to_Cj = sum(weight for v, weight in u_edges.items() if v in Cj)
            numerator += (u_to_Cj / w_u) / (2.0 * total_connections)
    
    # For different clusters, also count edges from Cj to Ci
    if not same_cluster:
        for v in Cj:
            v_edges = adj_dict.get(v, {})
            if not v_edges:
                continue
                
            w_v = sum(v_edges.values())
            if w_v < 1e-10:
                continue
                
            v_to_Ci = sum(weight for u, weight in v_edges.items() if u in Ci)
            numerator += (v_to_Ci / w_v) / (2.0 * total_connections)
    
    return numerator

def compute_AVI_dprime(Ci, adj_dict, n_nodes):
    """Compute AVI'' using adjacency dictionary for O(E) complexity with numerical stability."""
    sum_intra = 0.0
    sum_out = 0.0
    
    for u in Ci:
        u_edges = adj_dict.get(u, {})
        w_u = sum(u_edges.values())
        if w_u < 1e-10:  # Use small epsilon instead of exact 0
            continue
            
        # Compute intra-cluster sum
        for v in Ci:
            if v in u_edges:
                sum_intra += u_edges[v]
        
        # Compute out-cluster sum
        out_sum = sum(weight for v, weight in u_edges.items() if v not in Ci)
        # Use epsilon to prevent division by very small numbers
        sum_out += out_sum / w_u if w_u >= 1e-10 else 0
    
    denom = sum_intra + sum_out
    # Use epsilon comparison instead of exact 0
    return sum_intra / denom if denom >= 1e-10 else 0.0

def process_avu_batch(start_idx, end_idx, clusterIDs, clusters, adj_dict):
    """Process a batch of AVU calculations."""
    batch_values = {}
    
    # Force garbage collection before processing each batch
    import gc
    gc.collect()
    
    # Calculate total pairs for progress reporting
    total_pairs = len(clusterIDs) - start_idx
    processed_pairs = 0
    
    for i in range(start_idx, min(end_idx, len(clusterIDs))):
        if processed_pairs % 10 == 0:  # Print progress every 10 pairs
            print(f"        Progress: {processed_pairs}/{total_pairs} pairs processed", end='\r')
        # Compute diagonal value
        Ci = clusters[clusterIDs[i]]
        val = compute_AVU_dprime(Ci, Ci, adj_dict)
        batch_values[(clusterIDs[i], clusterIDs[i])] = val
        
        # Compute pairs with previous clusters
        for j in range(i + 1, len(clusterIDs)):
            Cj = clusters[clusterIDs[j]]
            val = compute_AVU_dprime(Ci, Cj, adj_dict)
            batch_values[(clusterIDs[i], clusterIDs[j])] = val
    return batch_values

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
    num_clusters = len(clusterIDs)
    print(f"    Found {num_clusters} clusters.")
    
    print("[4/7] Building adjacency dictionary...")
    adj_dict = compute_delta_dprime_matrix(edges, n_nodes)
    
    print("[5/7] Computing all AVU'' and AVI''...")
    AVU_values = {}
    AVI_values = {}
    
    # Process AVU calculations in batches
    # Dynamically adjust batch size based on number of clusters
    BATCH_SIZE = max(10, min(100, 10000 // num_clusters))  # Smaller batches for larger cluster counts
    total_batches = (num_clusters + BATCH_SIZE - 1) // BATCH_SIZE
    
    for batch in range(total_batches):
        start_idx = batch * BATCH_SIZE
        end_idx = min((batch + 1) * BATCH_SIZE, num_clusters)
        print(f"    Processing AVU batch {batch + 1}/{total_batches} (clusters {start_idx}-{end_idx})...")
        
        batch_values = process_avu_batch(start_idx, end_idx, clusterIDs, clusters, adj_dict)
        AVU_values.update(batch_values)
    
    # Compute AVI'' for each cluster in batches
    print("    Computing AVI values...")
    for i in range(0, num_clusters, BATCH_SIZE):
        end_idx = min(i + BATCH_SIZE, num_clusters)
        for j in range(i, end_idx):
            cID = clusterIDs[j]
            Ci = clusters[cID]
            val = compute_AVI_dprime(Ci, adj_dict, n_nodes)
            AVI_values[cID] = val
    
    print("[6/7] Computing final metrics...")
    
    if num_clusters == 1:
        cID = clusterIDs[0]
        finalAVI = AVI_values[cID]
        finalAVU = AVU_values[(cID, cID)]
    else:
        finalAVI = np.mean(list(AVI_values.values()))
        
        # Average AVU over distinct pairs without generating all combinations at once
        avu_sum = 0.0
        pair_count = 0
        for i in range(num_clusters):
            for j in range(i + 1, num_clusters):
                avu_sum += AVU_values[(clusterIDs[i], clusterIDs[j])]
                pair_count += 1
        finalAVU = avu_sum / pair_count if pair_count > 0 else 0.0
    
    if finalAVU == 0.0:
        finalQANUI = float("inf")
    else:
        finalQANUI = ((2.0 * finalAVI * (1.0 / finalAVU)) /
                      (finalAVI + (1.0 / finalAVU))) / 2
    
    print("[7/7] Results:\n")
    
    # For large cluster counts, only print summary statistics to avoid memory issues
    if len(clusterIDs) > 1000:
        print("===== METRICS SUMMARY (abbreviated for large cluster count) =====")
        print(f"Number of clusters: {len(clusterIDs)}")
        print(f"Sample of AVU values (first 5 pairs):")
        for i in range(min(5, len(clusterIDs))):
            for j in range(i, min(i+2, len(clusterIDs))):
                val = AVU_values[(clusterIDs[i], clusterIDs[j])]
                print(f"  AVU''({clusterIDs[i]}, {clusterIDs[j]}) = {val:.5f}")
        print("\nSample of AVI values (first 5 clusters):")
        for i in range(min(5, len(clusterIDs))):
            val = AVI_values[clusterIDs[i]]
            print(f"  AVI''({clusterIDs[i]}) = {val:.5f}")
        print()
    else:
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

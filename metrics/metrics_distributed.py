#!/usr/bin/env python3
import os
import argparse
import sys
import numpy as np
import mmap
from collections import defaultdict
import json
import tempfile
from pathlib import Path
import gc
import shutil
import signal
import atexit

class MetricsComputer:
    def __init__(self, graph_path, clusters_path, output_dir):
        self.graph_path = graph_path
        self.clusters_path = clusters_path
        self.output_dir = Path(output_dir)
        self.temp_dir = Path(tempfile.gettempdir()) / "metrics_temp"
        self.temp_dir.mkdir(exist_ok=True)
        
        # Checkpoint files
        self.nodes_checkpoint = self.temp_dir / "nodes.json"
        self.edges_checkpoint = self.temp_dir / "edges.npy"
        self.clusters_checkpoint = self.temp_dir / "clusters.json"
        self.avu_checkpoint = self.temp_dir / "avu_values.npy"
        self.avi_checkpoint = self.temp_dir / "avi_values.npy"
        self.progress_checkpoint = self.temp_dir / "progress.json"
        
        # Register cleanup handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        atexit.register(self.cleanup)
        
        # Initialize progress tracking
        self.load_or_init_progress()

    def signal_handler(self, signum, frame):
        """Handle interruption signals by cleaning up before exit."""
        print(f"\nReceived signal {signum}. Cleaning up...")
        self.cleanup()
        sys.exit(1)

    def cleanup(self):
        """Clean up all temporary files and resources."""
        cleanup_errors = []

        # Clean up memory-mapped arrays
        print("\nCleaning up memory-mapped arrays...")
        try:
            if hasattr(self, 'edges') and isinstance(self.edges, np.memmap):
                self.edges._mmap.close()
                del self.edges
        except Exception as e:
            cleanup_errors.append(f"Failed to close edges memmap: {e}")

        # List of checkpoint files to clean up
        checkpoint_files = [
            self.nodes_checkpoint,
            self.edges_checkpoint,
            self.clusters_checkpoint,
            self.avu_checkpoint,
            self.avi_checkpoint,
            self.progress_checkpoint
        ]

        # Clean up checkpoint files
        print("Cleaning up checkpoint files...")
        for checkpoint in checkpoint_files:
            if checkpoint.exists():
                try:
                    checkpoint.unlink()
                    print(f"Successfully deleted: {checkpoint}")
                except Exception as e:
                    cleanup_errors.append(f"Could not delete {checkpoint}: {e}")

        # Clean up temp directory with force
        if self.temp_dir.exists():
            try:
                shutil.rmtree(self.temp_dir, ignore_errors=True)
                print(f"Successfully removed temp directory: {self.temp_dir}")
            except Exception as e:
                cleanup_errors.append(f"Could not remove temp directory {self.temp_dir}: {e}")

        # Force garbage collection
        gc.collect()

        # Report any cleanup errors
        if cleanup_errors:
            print("\nWarning: Some cleanup operations failed:")
            for error in cleanup_errors:
                print(f"- {error}")
        else:
            print("\nCleanup completed successfully.")

    def verify_cleanup(self):
        """Verify that all temporary files and directories have been cleaned up."""
        verification_errors = []
        
        # Check if checkpoint files still exist
        for checkpoint in [
            self.nodes_checkpoint,
            self.edges_checkpoint,
            self.clusters_checkpoint,
            self.avu_checkpoint,
            self.avi_checkpoint,
            self.progress_checkpoint
        ]:
            if checkpoint.exists():
                verification_errors.append(f"Checkpoint file still exists: {checkpoint}")

        # Check if temp directory still exists
        if self.temp_dir.exists():
            verification_errors.append(f"Temp directory still exists: {self.temp_dir}")

        if verification_errors:
            print("\nWarning: Cleanup verification failed:")
            for error in verification_errors:
                print(f"- {error}")
            return False
        return True

    def load_or_init_progress(self):
        """Load or initialize progress tracking."""
        if self.progress_checkpoint.exists():
            with open(self.progress_checkpoint, 'r') as f:
                self.progress = json.load(f)
        else:
            self.progress = {
                'graph_processed': False,
                'clusters_processed': False,
                'adj_dict_built': False,
                'avu_current_batch': 0,
                'avi_current_idx': 0
            }
            self.save_progress()

    def save_progress(self):
        """Save current progress to checkpoint file."""
        with open(self.progress_checkpoint, 'w') as f:
            json.dump(self.progress, f)

    def read_graph_lazy(self):
        """Memory-efficient graph reading using memory mapping."""
        if self.progress['graph_processed'] and self.nodes_checkpoint.exists():
            print("Loading graph data from checkpoint...")
            with open(self.nodes_checkpoint, 'r') as f:
                nodes_data = json.load(f)
                self.nodes_list = nodes_data['nodes_list']
                self.node_to_idx = {node: idx for idx, node in enumerate(self.nodes_list)}
            self.edges = np.load(self.edges_checkpoint, mmap_mode='r', allow_pickle=True)
            return

        print("Processing graph...")
        nodes = set()
        edge_count = 0
        
        # First pass: count edges and collect nodes
        with open(self.graph_path, "r") as f:
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
        
        # Save nodes data
        self.nodes_list = sorted(nodes)
        self.node_to_idx = {node: idx for idx, node in enumerate(self.nodes_list)}
        with open(self.nodes_checkpoint, 'w') as f:
            json.dump({
                'nodes_list': self.nodes_list,
                'node_count': len(nodes)
            }, f)
        
        # Create memory-mapped array for edges
        self.edges = np.memmap(self.edges_checkpoint, dtype=np.float64, 
                             mode='w+', shape=(edge_count, 3))
        
        # Second pass: fill edges array
        idx = 0
        with open(self.graph_path, "r") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            for line in iter(mm.readline, b""):
                line = line.decode().strip()
                if not line or line.startswith("#"):
                    continue
                u, v, w = line.split()
                self.edges[idx] = [self.node_to_idx[u], self.node_to_idx[v], float(w)]
                idx += 1
            mm.close()
        
        self.edges.flush()
        self.progress['graph_processed'] = True
        self.save_progress()
        gc.collect()

    def read_clusters(self):
        """Read clusters with checkpointing."""
        if self.progress['clusters_processed'] and self.clusters_checkpoint.exists():
            print("Loading clusters from checkpoint...")
            with open(self.clusters_checkpoint, 'r') as f:
                clusters_data = json.load(f)
                self.clusters = defaultdict(set)
                for cID, members in clusters_data.items():
                    self.clusters[cID] = set(members)
            return

        print("Processing clusters...")
        self.clusters = defaultdict(set)
        with open(self.clusters_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                cID, member = line.split()
                if member in self.node_to_idx:
                    self.clusters[cID].add(self.node_to_idx[member])
        
        # Save clusters checkpoint
        with open(self.clusters_checkpoint, 'w') as f:
            clusters_data = {cID: list(members) for cID, members in self.clusters.items()}
            json.dump(clusters_data, f)
        
        self.progress['clusters_processed'] = True
        self.save_progress()
        gc.collect()

    def normalize_weight(self, w):
        """Normalize weight with numerical stability."""
        w_clipped = np.clip(w, -100, 100)
        return 0.98 * (1 / (1 + np.exp(-w_clipped)))

    def compute_avu_dprime(self, Ci, Cj, adj_dict):
        """Compute AVU'' for a pair of clusters."""
        if len(Ci) == 0 or len(Cj) == 0:
            return 0.0
            
        same_cluster = Ci == Cj
        numerator = 0.0
        
        total_connections = len(Ci) * len(Cj) if not same_cluster else (len(Ci) * (len(Ci) - 1)) // 2
        if total_connections == 0:
            return 0.0
        
        for u in Ci:
            u_edges = adj_dict.get(u, {})
            if not u_edges:
                continue
                
            w_u = sum(u_edges.values())
            if w_u < 1e-10:
                continue
                
            if same_cluster:
                for v in Cj:
                    if v > u and v in u_edges:
                        numerator += (u_edges[v] / w_u) / total_connections
            else:
                u_to_Cj = sum(weight for v, weight in u_edges.items() if v in Cj)
                numerator += (u_to_Cj / w_u) / (2.0 * total_connections)
        
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

    def compute_avi_dprime(self, Ci, adj_dict):
        """Compute AVI'' for a cluster."""
        sum_intra = 0.0
        sum_out = 0.0
        
        for u in Ci:
            u_edges = adj_dict.get(u, {})
            w_u = sum(u_edges.values())
            if w_u < 1e-10:
                continue
                
            for v in Ci:
                if v in u_edges:
                    sum_intra += u_edges[v]
            
            out_sum = sum(weight for v, weight in u_edges.items() if v not in Ci)
            sum_out += out_sum / w_u if w_u >= 1e-10 else 0
        
        denom = sum_intra + sum_out
        return sum_intra / denom if denom >= 1e-10 else 0.0

    def process_metrics(self):
        """Process metrics with checkpointing and memory management."""
        try:
            print("Starting metrics computation...")
            
            # Read input data
            self.read_graph_lazy()
            self.read_clusters()
            
            # Sort cluster IDs for consistent ordering
            cluster_ids = sorted(self.clusters.keys())
            n_clusters = len(cluster_ids)
            print(f"Processing {n_clusters} clusters...")
            
            # Create or load memory-mapped arrays for results
            if not self.avu_checkpoint.exists():
                avu_values = np.memmap(self.avu_checkpoint, dtype=np.float64,
                                     mode='w+', shape=(n_clusters, n_clusters))
                avu_values.fill(0)
            else:
                avu_values = np.memmap(self.avu_checkpoint, dtype=np.float64,
                                     mode='r+', shape=(n_clusters, n_clusters))
            
            if not self.avi_checkpoint.exists():
                avi_values = np.memmap(self.avi_checkpoint, dtype=np.float64,
                                     mode='w+', shape=(n_clusters,))
                avi_values.fill(0)
            else:
                avi_values = np.memmap(self.avi_checkpoint, dtype=np.float64,
                                     mode='r+', shape=(n_clusters,))
            
            # Build adjacency dictionary in batches
            EDGE_BATCH_SIZE = 1000000
            adj_dict = defaultdict(dict)
            
            for i in range(0, len(self.edges), EDGE_BATCH_SIZE):
                batch = self.edges[i:i + EDGE_BATCH_SIZE]
                for u_idx, v_idx, w in batch:
                    w_norm = self.normalize_weight(w)
                    weight = 100.0 if w_norm >= 0.99 else 1.0 / (1.0 - w_norm)
                    u_idx, v_idx = int(u_idx), int(v_idx)
                    adj_dict[u_idx][v_idx] = weight
                    adj_dict[v_idx][u_idx] = weight
                
                if i % (EDGE_BATCH_SIZE * 10) == 0:
                    print(f"Processed {i}/{len(self.edges)} edges...")
            
            # Process AVU values in batches
            CLUSTER_BATCH_SIZE = 100
            current_batch = self.progress['avu_current_batch']
            
            while current_batch * CLUSTER_BATCH_SIZE < n_clusters:
                start_idx = current_batch * CLUSTER_BATCH_SIZE
                end_idx = min((current_batch + 1) * CLUSTER_BATCH_SIZE, n_clusters)
                
                print(f"Processing AVU batch {current_batch + 1}/{(n_clusters + CLUSTER_BATCH_SIZE - 1) // CLUSTER_BATCH_SIZE}")
                
                for i in range(start_idx, end_idx):
                    Ci = self.clusters[cluster_ids[i]]
                    # Compute diagonal
                    val = self.compute_avu_dprime(Ci, Ci, adj_dict)
                    avu_values[i, i] = val
                    
                    # Compute pairs
                    for j in range(i + 1, n_clusters):
                        Cj = self.clusters[cluster_ids[j]]
                        val = self.compute_avu_dprime(Ci, Cj, adj_dict)
                        avu_values[i, j] = val
                        avu_values[j, i] = val  # Symmetric
                    
                    if i % 10 == 0:
                        avu_values.flush()
                        self.progress['avu_current_batch'] = current_batch
                        self.save_progress()
                
                current_batch += 1
                gc.collect()
            
            # Process AVI values
            current_idx = self.progress['avi_current_idx']
            
            while current_idx < n_clusters:
                print(f"Processing AVI values: {current_idx}/{n_clusters}")
                
                for i in range(current_idx, min(current_idx + CLUSTER_BATCH_SIZE, n_clusters)):
                    Ci = self.clusters[cluster_ids[i]]
                    val = self.compute_avi_dprime(Ci, adj_dict)
                    avi_values[i] = val
                    
                    if i % 10 == 0:
                        avi_values.flush()
                        self.progress['avi_current_idx'] = i
                        self.save_progress()
                
                current_idx += CLUSTER_BATCH_SIZE
                gc.collect()
            
            # Compute final metrics
            print("Computing final metrics...")
            
            if n_clusters == 1:
                finalAVI = float(avi_values[0])
                finalAVU = float(avu_values[0, 0])
            else:
                finalAVI = float(np.mean(avi_values))
                
                # Compute mean of upper triangle for AVU
                mask = np.triu(np.ones((n_clusters, n_clusters), dtype=bool), k=1)
                finalAVU = float(np.mean(avu_values[mask]))
            
            if finalAVU == 0.0:
                finalQANUI = float("inf")
            else:
                finalQANUI = ((2.0 * finalAVI * (1.0 / finalAVU)) /
                             (finalAVI + (1.0 / finalAVU))) / 2
            
            # Print results
            print("\n===== FINAL METRICS =====")
            print(f"Final AVI'' = {finalAVI:.5f}")
            print(f"Final AVU'' = {finalAVU:.5f}" if finalAVU != float('inf')
                  else "Final AVU'' = Inf")
            if finalQANUI == float("inf"):
                print("Final QANUI' = Infinity (division by zero in 1/AVU)")
            else:
                print(f"Final QANUI' = {finalQANUI:.5f}")
            
            # Save final results to metrics.log
            metrics_log = (
                "===== FINAL METRICS =====\n"
                f"Final AVI'' = {finalAVI:.5f}\n"
                f"Final AVU'' = {'Inf' if finalAVU == float('inf') else f'{finalAVU:.5f}'}\n"
                f"Final QANUI' = {'Infinity (division by zero in 1/AVU)' if finalQANUI == float('inf') else f'{finalQANUI:.5f}'}\n"
            )
            
            with open(self.output_dir / "metrics.log", 'w') as f:
                f.write(metrics_log)
            
            print("\nResults saved to:", self.output_dir / "metrics.log")

        except Exception as e:
            print(f"\nError during metrics computation: {e}")
            raise
        finally:
            # Always attempt cleanup
            self.cleanup()
            # Verify cleanup was successful
            self.verify_cleanup()

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Process metrics for graph and cluster data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python metrics_distributed.py /path/to/clusters.txt /path/to/output/dir
    python metrics_distributed.py --graph /path/to/graph.txt /path/to/clusters.txt /path/to/output/dir
        """
    )
    
    # Add arguments
    parser.add_argument(
        "clusters_path",
        type=str,
        help="Path to the clusters file (clusters.txt)"
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Directory where metrics.log will be saved"
    )
    parser.add_argument(
        "--graph",
        type=str,
        help="Path to the graph file (default: graph.txt in script directory)",
        default=None
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Validate file paths
    if not os.path.exists(args.clusters_path):
        print(f"Error: Clusters file not found: {args.clusters_path}")
        sys.exit(1)
        
    # Set graph path (maintain backward compatibility)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    graph_path = args.graph if args.graph else os.path.join(script_dir, "graph.txt")
    
    if not os.path.exists(graph_path):
        print(f"Error: Graph file not found: {graph_path}")
        sys.exit(1)
        
    # Validate output directory
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"Creating output directory: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
    elif not output_dir.is_dir():
        print(f"Error: Output path exists but is not a directory: {output_dir}")
        sys.exit(1)
        
    # Initialize and run metrics computer
    computer = MetricsComputer(graph_path, args.clusters_path, args.output_dir)
    computer.process_metrics()


if __name__ == "__main__":
    main()

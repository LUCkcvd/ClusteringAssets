import sys
from collections import defaultdict

def main():
    print("[1/7] Starting QANUI calculation...")

    #####################################################################
    # 1. READ THE GRAPH
    #####################################################################
    print("[2/7] Reading 'graph.txt'...")

    weight_dict = {}
    nodes = set()

    with open("graph.txt", "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Expect: node1 node2 weight
            u, v, w_str = line.split()
            w = float(w_str)
            # Store in dictionary for undirected access
            weight_dict[(u, v)] = w
            weight_dict[(v, u)] = w
            nodes.add(u)
            nodes.add(v)

    nodes = sorted(nodes)
    print(f"    Found {len(nodes)} total nodes.")

    #####################################################################
    # 2. READ THE CLUSTERS
    #####################################################################
    print("[3/7] Reading 'clusters.txt'...")

    clusters = defaultdict(set)
    with open("clusters.txt", "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Expect: clusterID clusterMember
            cID, member = line.split()
            clusters[cID].add(member)

    clusterIDs = sorted(clusters.keys())
    print(f"    Found {len(clusterIDs)} clusters.")
    
    #####################################################################
    # 3. DEFINE δʺ(u,v) = 1 / (1 - weight(u,v))
    #    If (u,v) not in weight_dict, assume weight=0 => δʺ(u,v)=1/(1-0)=1
    #####################################################################
    def normalize_weight(w):
        """Normalize weight to [0,1) range to avoid edge cases."""
        # Using sigmoid-like function to map weights to (0,1)
        # This ensures weights never reach 1 while preserving relative strengths
        return 0.99 * (1 / (1 + pow(2.71828, -w)))

    def delta_dprime(u, v):
        w_uv = weight_dict.get((u, v), 0.0)
        # Normalize weight to avoid edge cases
        w_uv = normalize_weight(w_uv)
        return 1.0 / (1.0 - w_uv)

    #####################################################################
    # 4. PRECOMPUTE w(u) = Σᵥ δʺ(u,v)
    #####################################################################
    print("[4/7] Computing w(u) for each node...")

    w_of_u = {}
    for i, u in enumerate(nodes):
        # Simple status update: e.g., every 25% done
        if i % max(1, (len(nodes)//4)) == 0:
            frac = 100.0*i/len(nodes)
            print(f"    ... {frac:.1f}% done computing w(u).")

        total = 0.0
        for v in nodes:
            total += delta_dprime(u, v)
        w_of_u[u] = total

    print("    Finished computing w(u).")

    #####################################################################
    # 5. DEFINE THE RE-MODIFIED AVU'' and AVI''.
    #
    #    AVU''(C_i, C_j)
    #       = [ Σ_{u in C_i, v in C_j} δʺ(u,v) ]
    #         / [ max(1, Σ_{u in C_i} w(u)) + max(1, Σ_{v in C_j} w(v)) ]
    #
    #    AVI''(C_i)
    #       = [ Σ_{u,v in C_i} δʺ(u,v) ]
    #         / [ Σ_{u,v in C_i} δʺ(u,v)  +  Σ_{u in C_i, v not in C_i} (δʺ(u,v)/w(u)) ]
    #####################################################################
    def compute_AVU_dprime(Ci, Cj):
        numerator = 0.0
        for u in Ci:
            for v in Cj:
                numerator += delta_dprime(u, v)

        sum_w_Ci = sum(w_of_u[u] for u in Ci)
        sum_w_Cj = sum(w_of_u[v] for v in Cj)

        denom = max(1.0, sum_w_Ci) + max(1.0, sum_w_Cj)
        if denom == 0.0:
            return 0.0
        return numerator / denom

    def compute_AVI_dprime(Ci):
        # sum_intra = Σ_{u,v in Ci} δʺ(u,v)
        sum_intra = 0.0
        Ci_list = list(Ci)
        for i1 in range(len(Ci_list)):
            for i2 in range(len(Ci_list)):
                u = Ci_list[i1]
                v = Ci_list[i2]
                sum_intra += delta_dprime(u, v)

        # sum_out = Σ_{u in Ci, v not in Ci} δʺ(u,v)/w(u)
        sum_out = 0.0
        notCi = set(nodes) - Ci
        for u in Ci:
            w_u = w_of_u[u]
            if w_u == 0.0:
                continue
            for v in notCi:
                sum_out += delta_dprime(u, v) / w_u

        denom = sum_intra + sum_out
        if denom == 0.0:
            return 0.0
        return sum_intra / denom

    #####################################################################
    # 6. COMPUTE AND PRINT PARTIAL RESULTS:
    #    - AVU''(C_i, C_j) for all pairs
    #    - AVI''(C_i) for each cluster
    #####################################################################
    print("[5/7] Computing all AVU'' and AVI''...")

    # Storing them in dictionaries for reuse in QANUI computation
    AVU_values = {}
    AVI_values = {}

    # Compute AVU'' for all pairs (i,j) with i <= j just as a demonstration
    for i in range(len(clusterIDs)):
        Ci = clusters[clusterIDs[i]]
        for j in range(i, len(clusterIDs)):
            Cj = clusters[clusterIDs[j]]
            val = compute_AVU_dprime(Ci, Cj)
            AVU_values[(clusterIDs[i], clusterIDs[j])] = val
            AVU_values[(clusterIDs[j], clusterIDs[i])] = val  # symmetrical

    # Compute AVI''(C_i) for each cluster
    for cID in clusterIDs:
        Ci = clusters[cID]
        val = compute_AVI_dprime(Ci)
        AVI_values[cID] = val

    print("    Done computing AVU'' and AVI'' for each cluster / cluster-pair.")

    #####################################################################
    # 7. DERIVE "FINAL" AVU'' AND AVI'' (by averaging), THEN QANUI
    #
    #    QANUI' = [ 2 × AVI' × (1 / AVU') ] / [ AVI' + (1 / AVU') ]
    #
    #   finalAVI = average of AVI''(C_i) over all clusters
    #   finalAVU = average of AVU''(C_i,C_j) over all distinct i<j
    #####################################################################
    print("[6/7] Computing final AVI, AVU, and QANUI...")

    # If there's only one cluster, define finalAVU = AVU''(C_1,C_1).
    if len(clusterIDs) == 1:
        cID = clusterIDs[0]
        finalAVI = AVI_values[cID]
        finalAVU = AVU_values[(cID, cID)]  # same cluster
    else:
        # Average AVI
        sumAVI = 0.0
        for cID in clusterIDs:
            sumAVI += AVI_values[cID]
        finalAVI = sumAVI / len(clusterIDs)

        # Average AVU over distinct pairs
        sumAVU = 0.0
        pair_count = 0
        for i in range(len(clusterIDs)):
            for j in range(i+1, len(clusterIDs)):
                sumAVU += AVU_values[(clusterIDs[i], clusterIDs[j])]
                pair_count += 1
        finalAVU = (sumAVU / pair_count) if pair_count > 0 else 0.0

    # QANUI' = 2 * finalAVI * (1/finalAVU) / [ finalAVI + (1/finalAVU) ]
    if finalAVU == 0.0:
        # Edge case if finalAVU=0 => 1/AVU => infinite => handle gracefully
        finalQANUI = float("inf")  
    else:
        finalQANUI = (2.0 * finalAVI * (1.0 / finalAVU)) / (finalAVI + (1.0 / finalAVU))

    #####################################################################
    # 8. PRINT RESULTS
    #####################################################################
    print("[7/7] Finished!  Here are the main outputs:\n")

    # 8.1. Print approximation progress for user satisfaction
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
    print(f"  Final AVU'' = {finalAVU:.5f}" if finalAVU != float('inf') else f"  Final AVU'' = Inf")
    if finalQANUI == float("inf"):
        print("  Final QANUI' = Infinity (division by zero in 1/AVU)")
    else:
        print(f"  Final QANUI' = {finalQANUI:.5f}")

    print("\nDone.")

if __name__ == "__main__":
    main()

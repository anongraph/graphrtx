import pygraph_rtx

def main():
    g = pygraph_rtx.Graph(device=0)
    print("Created GraphRTX instance")

    graph_path = "../data/graph.mtx"
    N = g.load_graph(graph_path)
    print(f"Loaded graph with {N} nodes")

    stats = g.prepare(max_seg_len=1024, num_partitions=0, num_dummy_nodes=0)
    print("Preparation complete:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    mem = g.device_memory_info_gb()
    print(f"GPU memory: {mem['free_gb']:.2f} GB free / {mem['total_gb']:.2f} GB total")

    print("\n--- BFS ---")
    bfs_res = g.run_bfs(src=0)
    for k, v in bfs_res.items():
        print(f"  {k}: {v}")

    print("\n--- PageRank ---")
    pr_res = g.run_pr(iters=20, damp=0.85)
    for k, v in pr_res.items():
        print(f"  {k}: {v}")

    print("\n--- SSSP ---")
    sssp_res = g.run_sssp(src=0)
    for k, v in sssp_res.items():
        print(f"  {k}: {v}")

    print("\n--- BC ---")
    bc_res = g.run_bc()
    print(bc_res)

    print("\n--- TC ---")
    tc_res = g.run_tc()
    print(tc_res)

    all_res = g.run_all(src=0, pr_iters=20, pr_damp=0.85)
    print("\n========== ALL RESULTS ==========")
    for r in all_res["results"]:
        algo = r["algo"]
        print(f"{algo:<6}: {r['ms']:.3f} ms")

if __name__ == "__main__":
    main()

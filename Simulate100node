"""
simulate100.py
- Tính giá trị trung bình đại diện cho trường hợp 100 node
- So sánh: Dijkstra vs GA vs Hybrid PSO–GA
"""

import pickle
import numpy as np
import os
import networkx as nx

# ==============================
# CONFIG
# ==============================
GRAPH_PATH = "./out/iot_graph100.gpickle"

FILES = {
    "Dijkstra": "./out/dijkstra_paths100.pkl",            # dict
    "GA": "./out/ga_pareto_paths100.pkl",            # list[path]
    "Hybrid_PSO_GA": "./out/hybrid_pso_ga_paths100.pkl"  # list[path]
}

OUTPUT_FILE = "./out/simulation100_results.pkl"


# ==============================
# LOAD GRAPH
# ==============================
def load_graph():
    with open(GRAPH_PATH, "rb") as f:
        return pickle.load(f)


# ==============================
# METRICS
# ==============================
def path_metrics(G, path):
    delay = energy = 0.0
    pdr = 1.0

    for i in range(len(path) - 1):
        e = G[path[i]][path[i + 1]]
        delay += e["delay"]
        energy += e["energy"]
        pdr *= e["pdr"]

    return {
        "delay": delay,
        "energy": energy,
        "pdr": pdr
    }


def compute_average(metrics_list):
    return {
        "avg_delay": float(np.mean([m["delay"] for m in metrics_list])),
        "avg_energy": float(np.mean([m["energy"] for m in metrics_list])),
        "avg_pdr": float(np.mean([m["pdr"] for m in metrics_list])),
        "num_runs": len(metrics_list)
    }


# ==============================
# MAIN
# ==============================
def main():
    print("Simulating 100-node results (AVERAGE METRICS)\n")

    G = load_graph()

    summary = {
        "nodes": 100,
        "methods": {}
    }

    for method, filepath in FILES.items():
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Missing result file: {filepath}")

        with open(filepath, "rb") as f:
            data = pickle.load(f)

        metrics_list = []

        # --------------------------
        # Dijkstra: đã là metrics
        # --------------------------
        if method == "Dijkstra":
            metrics_list.append({
                "delay": data["delay"],
                "energy": data["energy"],
                "pdr": data["pdr"]
            })

        # --------------------------
        # GA / Hybrid: list path
        # --------------------------
        else:
            for path in data:
                metrics_list.append(path_metrics(G, path))

        avg = compute_average(metrics_list)
        summary["methods"][method] = avg

        print(f"{method}")
        print(f"  Runs   : {avg['num_runs']}")
        print(f"  Delay  : {avg['avg_delay']:.4f}")
        print(f"  Energy : {avg['avg_energy']:.4f}")
        print(f"  PDR    : {avg['avg_pdr']:.4f}\n")

    # ==========================
    # SAVE RESULT
    # ==========================
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(summary, f)

    print(f"Saved averaged 100-node results to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

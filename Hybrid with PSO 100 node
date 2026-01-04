import random
import pickle
import networkx as nx
import copy

# ================== CONFIG ==================
GRAPH_PATH = "./out/iot_graph100.gpickle"
GA_PARETO_PATH = "./out/ga_pareto_paths100.pkl"

N_PARTICLES = 20
N_ITER = 30

W = 0.7
C1 = 1.5
C2 = 1.5

ALPHA = 0.4   # delay
BETA = 0.3    # energy
GAMMA = 0.3   # packet loss

PENALTY_LAMBDA = 1000
# ===========================================


# ---------- LOAD GRAPH ----------
def load_graph(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------- PATH METRICS ----------
def path_metrics(G, path):
    delay, energy, pdr = 0.0, 0.0, 1.0
    for i in range(len(path) - 1):
        if not G.has_edge(path[i], path[i+1]):
            return 1e6, 1e6, 0
        e = G[path[i]][path[i+1]]
        delay += e["delay"]
        energy += e["energy"]
        pdr *= e["pdr"]
    return delay, energy, pdr


# ---------- FITNESS WITH PENALTY ----------
def fitness(G, path):
    d, e, pdr = path_metrics(G, path)

    # QoS thresholds (example)
    D_MAX = 80
    E_MAX = 20
    P_MIN = 0.6

    penalty = 0
    if d > D_MAX:
        penalty += (d - D_MAX)
    if e > E_MAX:
        penalty += (e - E_MAX)
    if pdr < P_MIN:
        penalty += (P_MIN - pdr) * 100

    cost = ALPHA * d + BETA * e + GAMMA * (1 - pdr)
    return cost + PENALTY_LAMBDA * penalty


# ---------- PATH OPERATORS ----------
def mutate_path(G, path):
    """ GA mutation """
    if len(path) < 4:
        return path

    i = random.randint(1, len(path) - 3)
    neighbors = list(G.neighbors(path[i]))
    random.shuffle(neighbors)

    for n in neighbors:
        if n in path:
            continue
        try:
            tail = nx.shortest_path(G, n, path[-1], weight="delay")
            new_path = path[:i] + [n] + tail[1:]
            if len(new_path) == len(set(new_path)):
                return new_path
        except:
            pass
    return path


def pso_move(G, current, pbest, gbest):
    """ PSO-inspired move operator """
    path = current[:]

    if random.random() < 0.5:
        path = mutate_path(G, pbest)

    if random.random() < 0.5:
        path = mutate_path(G, gbest)

    return path


# ================= MAIN =================
if __name__ == "__main__":

    print("Hybrid PSO–GA Running ...")

    G = load_graph(GRAPH_PATH)

    with open(GA_PARETO_PATH, "rb") as f:
        swarm = pickle.load(f)

    swarm = swarm[:N_PARTICLES]
    pbest = copy.deepcopy(swarm)
    pbest_fit = [fitness(G, p) for p in pbest]

    archive = []

    for it in range(N_ITER):
        print(f"Iteration {it:02d}")

        for i in range(len(swarm)):
            # PSO movement
            new_path = pso_move(G, swarm[i], pbest[i], random.choice(pbest))

            f_new = fitness(G, new_path)
            f_old = fitness(G, swarm[i])

            if f_new < f_old:
                swarm[i] = new_path

            # Update pBest
            if f_new < pbest_fit[i]:
                pbest[i] = new_path
                pbest_fit[i] = f_new

        # Update archive (Pareto)
        for p in swarm:
            f = fitness(G, p)
            dominated = False
            for q in archive:
                if fitness(G, q) <= f:
                    dominated = True
                    break
            if not dominated:
                archive.append(p)

        # limit archive
        archive = archive[:10]

        print(f"  Archive size = {len(archive)}")

    # ====== OUTPUT ======
    print("\n===== FINAL HYBRID PSO-GA RESULTS =====")
    for i, p in enumerate(archive):
        d, e, pr = path_metrics(G, p)
        print(f"\nSolution {i+1}")
        print("Path:", p)
        print(f"Delay={d:.2f}, Energy={e:.2f}, PDR={pr:.4f}")

    with open("./out/hybrid_pso_ga_paths100.pkl", "wb") as f:
        pickle.dump(archive, f)

    print("\nSaved hybrid PSO-GA results.")

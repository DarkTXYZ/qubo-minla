import logging
from pyqubo import Array
import networkx as nx
import neal
from tqdm import tqdm
import pandas as pd
from dwave.samplers import TabuSampler
from scipy.sparse.linalg import eigsh
from scipy.sparse import csgraph
import numpy as np
import math
import time
import threading

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
logger = logging.getLogger(__name__)

def _format_seconds(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

class LiveTimer:
    def __init__(self, label="Elapsed", disable=False, position=0, refresh=1.0):
        self.label = label
        self.disable = disable
        self.position = position
        self.refresh = refresh
        self._stop = threading.Event()
        self._thread = None
        self._bar = None
        self.start = None

    def __enter__(self):
        if self.disable:
            return self
        self.start = time.perf_counter()
        # Simple line that we update with elapsed time
        self._bar = tqdm(
            total=0,
            bar_format="{desc}",
            position=self.position,
            leave=True,
            dynamic_ncols=True
        )
        def run():
            while not self._stop.is_set():
                elapsed = time.perf_counter() - self.start
                self._bar.set_description_str(f"{self.label}: {_format_seconds(elapsed)}")
                time.sleep(self.refresh)
            # Final update
            elapsed = time.perf_counter() - self.start
            self._bar.set_description_str(f"{self.label}: {_format_seconds(elapsed)} (done)")
        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.disable:
            return
        self._stop.set()
        self._thread.join()
        # Keep the last line (leave=True), then close
        self._bar.close()

def param_for_general_graph(n, m):
    naive = m * (n - 1)
    complete = n * (n - 1) * (n + 1) / 6

    k = math.ceil(n + 1/2 - 1/2 * math.sqrt(8*m + 1))
    f = (n - k) * (n - k + 1) / 2
    edges_method = (m - f) * (k - 1) + (n - k) * (n**2 + (n + 3) * k - 2*k**2 - 1) / 6

    logger.debug(f"n={n}, m={m}, naive={naive}, complete={complete}, edges_method={edges_method}")
    return min(naive, complete, edges_method)

def generate_qubo_for_MBSP(G):
    V = G.number_of_nodes()
    E = list(G.edges())
    binary = Array.create('B', (V, V), 'BINARY')

    def thermometer_encoding(param):
        return param * sum(
            sum((1 - binary[u][k]) * binary[u][k + 1] for k in range(V - 1))
            for u in range(V)
        )

    def bijective(param):
        return param * sum(
            (sum(binary[u][k] for u in range(V)) - (V - k)) ** 2
            for k in range(V)
        )

    objective = sum(
        sum(binary[u][k] + binary[v][k] - 2 * binary[u][k] * binary[v][k] for k in range(V))
        for u, v in E
    )

    _lambda = param_for_general_graph(V, len(E))
    H = objective + thermometer_encoding(_lambda) + bijective(_lambda)
    model = H.compile()
    bqm = model.to_bqm()
    logger.debug(f"QUBO generated for graph with {V} nodes and {len(E)} edges.")
    return model, bqm, _lambda

def get_best_sample_from_sampleset(sampleset, V):
    lowest = sampleset.lowest()
    rec = lowest.first
    sample = rec.sample
    energy = float(rec.energy)

    # Build dense matrix once for vectorized checks
    B = np.fromiter(
        (sample[f'B[{u}][{k}]'] for u in range(V) for k in range(V)),
        dtype=np.int8, count=V*V
    ).reshape(V, V)

    # Row violations: count 01 transitions
    row_violations = int(np.sum((1 - B[:, :-1]) * B[:, 1:]))

    # Column violations: sums must equal V-k
    col_sums = B.sum(axis=0)
    target = np.arange(V, 0, -1)  # V, V-1, ..., 1
    col_violations = int(np.sum(col_sums != target))

    feasible = (row_violations == 0 and col_violations == 0)
    values = B.sum(axis=1).tolist()
    return feasible, energy, values

def generate_instance_facebook():
    # Faster loader via pandas
    df = pd.read_csv(
        "dataset/facebook_dataset.edges",
        sep=r"\s+",
        header=None,
        names=["u", "v"],
        engine="c",
        dtype=np.int32
    )
    G = nx.from_pandas_edgelist(df, "u", "v", create_using=nx.Graph())
    G = nx.convert_node_labels_to_integers(G, first_label=0)
    return ("facebook", G.number_of_nodes(), G)

def generate_instances():
    sizes = [100, 200, 300]  # Reduced sizes for quicker tests
    logger.info(f"Generating instances for sizes: {sizes}")
    return [("random", size, nx.gnp_random_graph(size, p=0.5, seed=42)) for size in sizes]

def MBSP_greedy(G):
    V = G.number_of_nodes()
    degrees = dict(G.degree())
    ordered = sorted(G.nodes(), key=degrees.get)
    assigned = {node: i + 1 for i, node in enumerate(ordered)}
    val = sum(abs(assigned[u] - assigned[v]) for u, v in G.edges())
    logger.debug(f"Greedy heuristic value: {val}")
    return val

def spectral_minla(G):
    # Faster Laplacian construction
    A = nx.to_scipy_sparse_array(G, format='csr', dtype=np.float64)
    L = csgraph.laplacian(A, normed=False)
    vals, vecs = eigsh(L, k=2, which='SM')  # second smallest eigenvector
    fiedler = vecs[:, 1]
    node_order = [n for _, n in sorted(zip(fiedler, G.nodes()))]
    label_map = {node: idx + 1 for idx, node in enumerate(node_order)}
    val = sum(abs(label_map[u] - label_map[v]) for u, v in G.edges())
    logger.debug(f"Spectral heuristic value: {val}")
    return val, node_order

def run_heuristics(G, n, bqm, model, verbose=False):
    val_greedy = MBSP_greedy(G)
    val_spectral_order, spectral_order = spectral_minla(G)
    num_simulate = 20

    # Build warm-start initial state from spectral_order
    label_map = {node: idx + 1 for idx, node in enumerate(spectral_order)}
    warm_state = {}
    for u in range(n):
        L = label_map[u]
        # B[u][k] = 1 if L > k else 0
        for k in range(n):
            warm_state[f'B[{u}][{k}]'] = 1 if L > k else 0

    SA_solver = neal.SimulatedAnnealingSampler()
    TB_solver = TabuSampler()

    SA_results = []
    TB_results = []

    # Keep SA/TB bars below the top timer line
    pos_base = 1 if verbose else 0

    sa_iter = tqdm(
        range(num_simulate),
        desc="running SA",
        unit="its",
        leave=False,
        disable=not verbose,
        position=pos_base,
        dynamic_ncols=True
    )
    for _ in sa_iter:
        sampleset = SA_solver.sample(
            bqm,
            num_sweeps=1000,
            initial_states=[warm_state],
            initial_states_generator='tile'
        )
        feasible_SA, val_SA, _ = get_best_sample_from_sampleset(sampleset, n)
        SA_results.append((feasible_SA, val_SA))
        if verbose:
            sa_iter.set_postfix({"feasible": feasible_SA, "val": val_SA})

    tb_iter = tqdm(
        range(num_simulate),
        desc="running TB",
        unit="its",
        leave=False,
        disable=not verbose,
        position=pos_base + 1,
        dynamic_ncols=True
    )
    for _ in tb_iter:
        sampleset = TB_solver.sample(
            bqm,
            tenure=10,
            initial_states=[warm_state],
            initial_states_generator='tile'
        )
        feasible_tabu, val_Tabu, _ = get_best_sample_from_sampleset(sampleset, n)
        TB_results.append((feasible_tabu, val_Tabu))
        if verbose:
            tb_iter.set_postfix({"feasible": feasible_tabu, "val": val_Tabu})

    SA_values = np.array([val for feasible, val in SA_results if feasible])
    TB_values = np.array([val for feasible, val in TB_results if feasible])

    def stats(arr):
        if arr.size == 0:
            return 0, 0, (0, 0)
        return float(np.mean(arr)), float(np.std(arr)), (int(np.min(arr)), int(np.max(arr)))

    avg_SA, std_SA, range_SA = stats(SA_values)
    avg_TB, std_TB, range_TB = stats(TB_values)

    SA_feasibility = float(np.mean([feasible for feasible, _ in SA_results])) if SA_results else 0.0
    TB_feasibility = float(np.mean([feasible for feasible, _ in TB_results])) if TB_results else 0.0

    logger.info(f"SA (warm-start): feasibility={SA_feasibility}, avg={avg_SA}, std={std_SA}, range={range_SA}")
    logger.info(f"TB (warm-start): feasibility={TB_feasibility}, avg={avg_TB}, std={avg_TB}, range={range_TB}")

    return {
        "SA_warm": {"feasibility": SA_feasibility, "average": avg_SA, "std": std_SA, "range": range_SA},
        "TB_warm": {"feasibility": TB_feasibility, "average": avg_TB, "std": avg_TB, "range": range_TB},
        "Greedy": val_greedy,
        "Spectral": val_spectral_order
    }

def experiment(verbose=True):
    start_time = time.perf_counter()
    instances = generate_instances()
    # instance = generate_instance_facebook()
    df_rows = []

    # Place experiment bar below the timer if verbose
    inst_iter = tqdm(
        instances,
        desc="Experiment 2 Warm Start",
        unit="inst",
        disable=not verbose,
        leave=False,
        position=3 if verbose else 0,
        dynamic_ncols=True
    )

    print("Generate instance successful")

    with LiveTimer(label="Total time", disable=not verbose, position=0):
        for _, n, G in inst_iter:
            m = G.number_of_edges()
            logger.info(f"Processing instance with n={n}, m={m}")
            model, bqm, penalty_param = generate_qubo_for_MBSP(G)
            heuristics = run_heuristics(G, n, bqm, model, verbose=verbose)

            df_rows.append({
                "heuristics": "Greedy",
                "n": n,
                "m": m,
                "val": heuristics["Greedy"]
            })
            df_rows.append({
                "heuristics": "Spectral",
                "n": n,
                "m": m,
                "val": heuristics["Spectral"]
            })
            df_rows.append({
                "heuristics": "SA_warm",
                "n": n,
                "m": m,
                "penalty_param": penalty_param,
                "feasible_rate": heuristics["SA_warm"]["feasibility"],
                "average": heuristics["SA_warm"]["average"],
                "std": heuristics["SA_warm"]["std"],
                "range": heuristics["SA_warm"]["range"]
            })
            df_rows.append({
                "heuristics": "TB_warm",
                "n": n,
                "m": m,
                "penalty_param": penalty_param,
                "feasible_rate": heuristics["TB_warm"]["feasibility"],
                "average": heuristics["TB_warm"]["average"],
                "std": heuristics["TB_warm"]["std"],
                "range": heuristics["TB_warm"]["range"]
            })

    df = pd.DataFrame(df_rows)
    df.to_csv("exp2_warm_results.csv", index=False)
    elapsed = time.perf_counter() - start_time
    logger.info(f"Experiment completed in {_format_seconds(elapsed)} and results saved to exp2_warm_results.csv")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging and tqdm progress bars")
    args = parser.parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    experiment(verbose=args.verbose)

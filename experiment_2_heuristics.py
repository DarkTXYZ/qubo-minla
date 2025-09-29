import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional

from pyqubo import Array
import networkx as nx
import neal
from ortools.sat.python import cp_model  # noqa: F401 (kept if used elsewhere)
from tqdm import tqdm
import pandas as pd
from dwave.samplers import TabuSampler
from scipy.sparse.linalg import eigsh
import numpy as np
import math
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ExperimentConfig:
    # Instance generation
    instance_sizes: List[int] = field(default_factory=lambda: [10, 50, 100, 200, 300])
    random_graph_p: float = 0.2
    random_graph_seed: int = 42

    # Sampling control
    simulate_iters: int = 10  # number of outer iterations for SA/TB loops
    methods: List[str] = field(default_factory=lambda: ["TB"])  # which samplers to run

    # Simulated annealing params
    sa_num_sweeps: int = 50000
    sa_num_reads: int = 1
    sa_seed: Optional[int] = None

    # Tabu search params
    tb_tenure: int = 20
    tb_num_reads: int = 1
    tb_seed: Optional[int] = None

    # Output
    results_path: str = "results/exp2_results.csv"


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

    t0 = time.perf_counter()
    objective = sum(
        sum(binary[u][k] + binary[v][k] - 2 * binary[u][k] * binary[v][k] for k in range(V))
        for u, v in E
    )
    t1 = time.perf_counter()

    _lambda = param_for_general_graph(V, len(E))
    H = objective + thermometer_encoding(_lambda) + bijective(_lambda)
    t2 = time.perf_counter()

    model = H.compile()
    t3 = time.perf_counter()

    bqm = model.to_bqm()
    t4 = time.perf_counter()

    logger.info(
        f"QUBO build timings: objective={t1 - t0:.3f}s, penalties={t2 - t1:.3f}s, "
        f"compile={t3 - t2:.3f}s, to_bqm={t4 - t3:.3f}s, total={t4 - t0:.3f}s"
    )
    logger.debug(f"QUBO generated for graph with {V} nodes and {len(E)} edges.")
    return model, bqm, _lambda

def get_best_sample(decoded_samples, V):
    best_sample = min(decoded_samples, key=lambda s: s.energy)
    data = best_sample.sample

    row_violations = sum(
        1
        for u in range(V)
        for k in range(V - 1)
        if data[f'B[{u}][{k}]'] == 0 and data[f'B[{u}][{k+1}]'] == 1
    )
    col_violations = sum(
        1
        for k in range(V)
        if sum(data[f'B[{u}][{k}]'] for u in range(V)) != (V - k)
    )
    feasible = (row_violations == 0 and col_violations == 0)
    values = [sum(data[f'B[{u}][{k}]'] for k in range(V)) for u in range(V)]
    logger.debug(f"Best sample energy: {best_sample.energy}, feasible: {feasible}")
    return feasible, best_sample.energy, values

def generate_instances(config: ExperimentConfig):
    sizes = config.instance_sizes
    logger.info(f"Generating instances for sizes: {sizes}, p={config.random_graph_p}, seed={config.random_graph_seed}")
    t0 = time.perf_counter()
    instances = [("random", size, nx.gnp_random_graph(size, p=config.random_graph_p, seed=config.random_graph_seed)) for size in sizes]
    logger.info(f"Generate instances took {time.perf_counter() - t0:.3f}s")
    return instances

def MBSP_greedy(G):
    V = G.number_of_nodes()
    labels = list(range(1, V + 1))
    degrees = dict(G.degree())
    assigned = {}
    unassigned = set(G.nodes())
    for label in labels:
        min_deg_node = min(unassigned, key=lambda u: degrees[u])
        assigned[min_deg_node] = label
        unassigned.remove(min_deg_node)
    val = sum(abs(assigned[u] - assigned[v]) for u, v in G.edges())
    logger.debug(f"Greedy heuristic value: {val}")
    return val

def spectral_minla(G):
    L = nx.laplacian_matrix(G).astype(float)
    vals, vecs = eigsh(L, k=2, which='SM')
    fiedler = vecs[:, 1]
    node_order = [n for _, n in sorted(zip(fiedler, G.nodes()))]
    label_map = {node: idx + 1 for idx, node in enumerate(node_order)}
    val = sum(abs(label_map[u] - label_map[v]) for u, v in G.edges())
    logger.debug(f"Spectral heuristic value: {val}")
    return val

def run_heuristics(G, n, bqm, model, config: ExperimentConfig, verbose=False):
    t0 = time.perf_counter()
    val_greedy = MBSP_greedy(G)
    t1 = time.perf_counter()
    logger.info(f"Greedy heuristic took {t1 - t0:.3f}s")

    val_spectral_order = spectral_minla(G)
    t2 = time.perf_counter()
    logger.info(f"Spectral heuristic took {t2 - t1:.3f}s")

    num_simulate = config.simulate_iters

    methods = [m.upper() for m in (config.methods or [])]
    do_sa = "SA" in methods
    do_tb = "TB" in methods

    SA_results = []
    TB_results = []

    if do_sa:
        SA_solver = neal.SimulatedAnnealingSampler()
        sa_start = time.perf_counter()
        sa_iter = tqdm(range(num_simulate), desc="running SA", unit="its", leave=False, disable=not verbose)
        sa_kwargs = {"num_sweeps": config.sa_num_sweeps, "num_reads": config.sa_num_reads}
        if config.sa_seed is not None:
            sa_kwargs["seed"] = config.sa_seed
        for _ in sa_iter:
            sampleset = SA_solver.sample(bqm, **sa_kwargs)
            decoded_samples = model.decode_sampleset(sampleset)
            feasible_SA, val_SA, _ = get_best_sample(decoded_samples, n)
            SA_results.append((feasible_SA, val_SA))
            if verbose:
                sa_iter.set_postfix({"feasible": feasible_SA, "val": val_SA})
        sa_total = time.perf_counter() - sa_start
        logger.info(f"Simulated Annealing loop took {sa_total:.3f}s (~{sa_total/num_simulate:.3f}s/iter)")
    else:
        logger.info("Simulated Annealing disabled by config.")

    if do_tb:
        TB_solver = TabuSampler()
        tb_start = time.perf_counter()
        tb_iter = tqdm(range(num_simulate), desc="running TB", unit="its", leave=False, disable=not verbose)
        tb_kwargs = {"tenure": config.tb_tenure, "num_reads": config.tb_num_reads}
        if config.tb_seed is not None:
            tb_kwargs["seed"] = config.tb_seed
        for _ in tb_iter:
            sampleset = TB_solver.sample(bqm, **tb_kwargs)
            decoded_samples = model.decode_sampleset(sampleset)
            feasible_tabu, val_Tabu, _ = get_best_sample(decoded_samples, n)
            TB_results.append((feasible_tabu, val_Tabu))
            if verbose:
                tb_iter.set_postfix({"feasible": feasible_tabu, "val": val_Tabu})
        tb_total = time.perf_counter() - tb_start
        logger.info(f"Tabu Search loop took {tb_total:.3f}s (~{tb_total/num_simulate:.3f}s/iter)")
    else:
        logger.info("Tabu Search disabled by config.")

    # Use numpy for statistics
    def stats(arr):
        if arr.size == 0:
            return 0, 0, (0, 0)
        return float(np.mean(arr)), float(np.std(arr)), (int(np.min(arr)), int(np.max(arr)))

    results = {
        "Greedy": val_greedy,
        "Spectral": val_spectral_order
    }

    if do_sa:
        SA_values = np.array([val for feasible, val in SA_results if feasible])
        avg_SA, std_SA, range_SA = stats(SA_values)
        SA_feasibility = float(np.mean([feasible for feasible, _ in SA_results])) if SA_results else 0.0
        logger.info(f"SA: feasibility={SA_feasibility}, avg={avg_SA}, std={std_SA}, range={range_SA}")
        results["SA"] = {"feasibility": SA_feasibility, "average": avg_SA, "std": std_SA, "range": range_SA}

    if do_tb:
        TB_values = np.array([val for feasible, val in TB_results if feasible])
        avg_TB, std_TB, range_TB = stats(TB_values)
        TB_feasibility = float(np.mean([feasible for feasible, _ in TB_results])) if TB_results else 0.0
        logger.info(f"TB: feasibility={TB_feasibility}, avg={avg_TB}, std={std_TB}, range={range_TB}")
        results["TB"] = {"feasibility": TB_feasibility, "average": avg_TB, "std": std_TB, "range": range_TB}

    return results

def experiment(config: ExperimentConfig, verbose=True):
    logger.info(f"Running with config: {config}")
    overall_start = time.perf_counter()
    instances = generate_instances(config)
    df_rows = []
    inst_iter = tqdm(instances, desc="Experiment 2", unit="inst", disable=not verbose)
    for _, n, G in inst_iter:
        m = G.number_of_edges()
        logger.info(f"Processing instance with n={n}, m={m}")

        t_qubo_start = time.perf_counter()
        model, bqm, penalty_param = generate_qubo_for_MBSP(G)
        logger.info(f"QUBO generation for n={n}, m={m} took {time.perf_counter() - t_qubo_start:.3f}s")

        t_heur_start = time.perf_counter()
        heuristics = run_heuristics(G, n, bqm, model, config=config, verbose=verbose)
        logger.info(f"Heuristics for n={n}, m={m} took {time.perf_counter() - t_heur_start:.3f}s")

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

        if "SA" in heuristics:
            df_rows.append({
                "heuristics": "SA",
                "n": n,
                "m": m,
                "penalty_param": penalty_param,
                "feasible_rate": heuristics["SA"]["feasibility"],
                "average": heuristics["SA"]["average"],
                "std": heuristics["SA"]["std"],
                "range": heuristics["SA"]["range"]
            })

        if "TB" in heuristics:
            df_rows.append({
                "heuristics": "TB",
                "n": n,
                "m": m,
                "penalty_param": penalty_param,
                "feasible_rate": heuristics["TB"]["feasibility"],
                "average": heuristics["TB"]["average"],
                "std": heuristics["TB"]["std"],
                "range": heuristics["TB"]["range"]
            })

    t_save = time.perf_counter()
    df = pd.DataFrame(df_rows)
    os.makedirs(os.path.dirname(config.results_path), exist_ok=True) if os.path.dirname(config.results_path) else None
    df.to_csv(config.results_path, index=False)
    logger.info(f"Saving results took {time.perf_counter() - t_save:.3f}s")
    logger.info(f"Experiment completed in {time.perf_counter() - overall_start:.3f}s and results saved to {config.results_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging and tqdm progress bars")

    # Config overrides
    parser.add_argument("--sizes", type=int, nargs="+", help="Instance sizes (e.g., --sizes 10 50 100)")
    parser.add_argument("--p", type=float, help="Erdos-Renyi edge prob p")
    parser.add_argument("--seed", type=int, help="Graph generation seed")

    parser.add_argument("--simulate-iters", type=int, help="Number of outer iterations for SA/TB loops")
    parser.add_argument("--methods", nargs="+", choices=["SA", "TB"], help="Heuristic samplers to run (e.g., --methods SA or --methods SA TB)")

    parser.add_argument("--sa-num-sweeps", type=int, help="SA num_sweeps")
    parser.add_argument("--sa-num-reads", type=int, help="SA num_reads")
    parser.add_argument("--sa-seed", type=int, help="SA seed")

    parser.add_argument("--tb-tenure", type=int, help="Tabu tenure")
    parser.add_argument("--tb-num-reads", type=int, help="Tabu num_reads")
    parser.add_argument("--tb-seed", type=int, help="Tabu seed")

    parser.add_argument("--results", type=str, help="Path to results CSV")

    args = parser.parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    default_cfg = ExperimentConfig()
    cfg = ExperimentConfig(
        instance_sizes=args.sizes if args.sizes else default_cfg.instance_sizes,
        random_graph_p=args.p if args.p is not None else default_cfg.random_graph_p,
        random_graph_seed=args.seed if args.seed is not None else default_cfg.random_graph_seed,
        simulate_iters=args.simulate_iters if args.simulate_iters is not None else default_cfg.simulate_iters,
        methods=args.methods if args.methods is not None else default_cfg.methods,
        sa_num_sweeps=args.sa_num_sweeps if args.sa_num_sweeps is not None else default_cfg.sa_num_sweeps,
        sa_num_reads=args.sa_num_reads if args.sa_num_reads is not None else default_cfg.sa_num_reads,
        sa_seed=args.sa_seed if args.sa_seed is not None else default_cfg.sa_seed,
        tb_tenure=args.tb_tenure if args.tb_tenure is not None else default_cfg.tb_tenure,
        tb_num_reads=args.tb_num_reads if args.tb_num_reads is not None else default_cfg.tb_num_reads,
        tb_seed=args.tb_seed if args.tb_seed is not None else default_cfg.tb_seed,
        results_path=args.results if args.results is not None else default_cfg.results_path,
    )

    experiment(config=cfg, verbose=args.verbose)

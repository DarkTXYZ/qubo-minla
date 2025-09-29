import os
import networkx as nx
from pyqubo import Array
import math
from tqdm import tqdm
import numpy as np
import pandas as pd
import neal
from dwave.samplers import TabuSampler
from scipy.sparse.linalg import eigsh
import time
import logging


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiment_log.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def read_graph_from_edge_list(file_path: str) -> nx.Graph:
    start_time = time.time()
    G = nx.Graph()
    with open(file_path, 'r') as f:
        for line in f:
            u, v = map(int, line.strip().split())
            G.add_edge(u, v)
    elapsed = time.time() - start_time
    logger.info(f"Graph loaded from {file_path} in {elapsed:.3f}s")
    return G


def read_datasets():
    start_time = time.time()
    logger.info("Starting dataset reading...")
    
    datasets = {}
    dataset_dirs = {
        "synthetic": "synthetic-dataset",
        "real_world": "real-world-dataset"
    }
    
    for key, dir_path in dataset_dirs.items():
        if not os.path.exists(dir_path):
            logger.warning(f"Directory {dir_path} not found")
            continue
        datasets[key] = {}
        for filename in os.listdir(dir_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(dir_path, filename)
                logger.info(f"Reading dataset: {file_path}")
                G = read_graph_from_edge_list(file_path)
                datasets[key][filename] = G
                logger.info(f"  Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    total_elapsed = time.time() - start_time
    logger.info(f"Dataset reading completed in {total_elapsed:.3f}s")
    return datasets


def generate_qubo_for_MBSP(G):
    start_time = time.time()
    logger.debug(f"Generating QUBO for graph with {G.number_of_nodes()} nodes")
    
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
    
    elapsed = time.time() - start_time
    logger.debug(f"QUBO generation completed in {elapsed:.3f}s")
    return model, bqm, _lambda


def param_for_general_graph(n, m):
    naive = m * (n - 1)
    complete = n * (n - 1) * (n + 1) / 6
    k = math.ceil(n + 1/2 - 1/2 * math.sqrt(8*m + 1))
    f = (n - k) * (n - k + 1) / 2
    edges_method = (m - f) * (k - 1) + (n - k) * (n**2 + (n + 3) * k - 2*k**2 - 1) / 6
    return min(naive, complete, edges_method)


def get_best_sample(decoded_samples, V):
    if not decoded_samples:
        return False, float('inf'), []
        
    best_sample = min(decoded_samples, key=lambda s: s.energy)
    data = best_sample.sample
    
    row_violations = sum(
        1 for u in range(V) for k in range(V - 1)
        if data.get(f'B[{u}][{k}]', 0) == 0 and data.get(f'B[{u}][{k+1}]', 0) == 1
    )
    
    col_violations = sum(
        1 for k in range(V)
        if sum(data.get(f'B[{u}][{k}]', 0) for u in range(V)) != (V - k)
    )
    
    feasible = (row_violations == 0 and col_violations == 0)
    values = [sum(data.get(f'B[{u}][{k}]', 0) for k in range(V)) for u in range(V)]
    
    return feasible, best_sample.energy, values


def greedy_minla(G):
    start_time = time.time()
    
    V = G.number_of_nodes()
    labels = list(range(1, V + 1))
    degrees = dict(G.degree())
    assigned = {}
    unassigned = set(G.nodes())
    
    for label in labels:
        min_deg_node = min(unassigned, key=lambda u: degrees[u])
        assigned[min_deg_node] = label
        unassigned.remove(min_deg_node)
    
    result = sum(abs(assigned[u] - assigned[v]) for u, v in G.edges())
    elapsed = time.time() - start_time
    logger.debug(f"Greedy MinLA completed in {elapsed:.3f}s, result: {result}")
    return result


def spectral_minla(G):
    start_time = time.time()
    
    if G.number_of_nodes() < 2:
        elapsed = time.time() - start_time
        logger.debug(f"Spectral MinLA (trivial case) completed in {elapsed:.3f}s")
        return 0
        
    L = nx.laplacian_matrix(G).astype(float)
    vals, vecs = eigsh(L, k=2, which='SM')
    fiedler = vecs[:, 1]
    node_order = [n for _, n in sorted(zip(fiedler, G.nodes()))]
    label_map = {node: idx + 1 for idx, node in enumerate(node_order)}
    
    result = sum(abs(label_map[u] - label_map[v]) for u, v in G.edges())
    elapsed = time.time() - start_time
    logger.debug(f"Spectral MinLA completed in {elapsed:.3f}s, result: {result}")
    return result


def run_solver(solver, bqm, model, n, num_simulate, desc, **solver_params):
    start_time = time.time()
    logger.debug(f"Starting {desc} with {num_simulate} simulations")
    
    results = []
    solver_iter = tqdm(range(num_simulate), desc=desc, unit="its", leave=False)
    
    for i in solver_iter:
        iter_start = time.time()
        try:
            sampleset = solver.sample(bqm, **solver_params)
            decoded_samples = model.decode_sampleset(sampleset)
            feasible, val, _ = get_best_sample(decoded_samples, n)
            results.append((feasible, val))
        except Exception as e:
            logger.warning(f"Error in {desc} iteration {i}: {e}")
            results.append((False, float('inf')))
        
        iter_elapsed = time.time() - iter_start
        logger.debug(f"{desc} iteration {i+1}/{num_simulate} completed in {iter_elapsed:.3f}s")
    
    total_elapsed = time.time() - start_time
    logger.info(f"{desc} completed in {total_elapsed:.3f}s")
    return results


def calculate_stats(values):
    if len(values) == 0:
        return 0.0, 0.0, (0, 0)
    values_array = np.array(values)
    return float(np.mean(values_array)), float(np.std(values_array)), (int(np.min(values_array)), int(np.max(values_array)))


def run_heuristics(G, n, bqm, model, num_simulate=10, verbose=False):
    start_time = time.time()
    graph_id = f"n{n}_m{G.number_of_edges()}"
    logger.info(f"Running heuristics for graph {graph_id}")
    
    # Greedy
    greedy_start = time.time()
    val_greedy = greedy_minla(G)
    greedy_time = time.time() - greedy_start
    logger.info(f"Greedy completed in {greedy_time:.3f}s")
    
    # Spectral
    spectral_start = time.time()
    val_spectral = spectral_minla(G)
    spectral_time = time.time() - spectral_start
    logger.info(f"Spectral completed in {spectral_time:.3f}s")
    
    # SA Solver
    sa_start = time.time()
    SA_solver = neal.SimulatedAnnealingSampler()
    SA_results = run_solver(SA_solver, bqm, model, n, num_simulate, "running SA", 
                           num_sweeps=1000, num_reads=1)
    sa_time = time.time() - sa_start
    logger.info(f"SA solver completed in {sa_time:.3f}s")
    
    # TB Solver
    tb_start = time.time()
    TB_solver = TabuSampler()
    TB_results = run_solver(TB_solver, bqm, model, n, num_simulate, "running TB",
                           tenure=20, num_reads=1)
    tb_time = time.time() - tb_start
    logger.info(f"TB solver completed in {tb_time:.3f}s")
    
    results = {
        "Greedy": val_greedy,
        "Spectral": val_spectral
    }
    
    for solver_name, solver_results in [("SA", SA_results), ("TB", TB_results)]:
        feasible_results = [val for feasible, val in solver_results if feasible]
        feasibility_rate = float(np.mean([feasible for feasible, _ in solver_results]))
        avg, std, range_vals = calculate_stats(feasible_results)
        
        results[solver_name] = {
            "feasibility": feasibility_rate,
            "average": avg,
            "std": std,
            "range": range_vals
        }
        
        logger.info(f"{solver_name}: feasibility={feasibility_rate:.3f}, avg={avg:.2f}")
    
    total_elapsed = time.time() - start_time
    logger.info(f"All heuristics for graph {graph_id} completed in {total_elapsed:.3f}s")
    return results


def create_result_row(graph_id, heuristic_name, n, m, penalty_param=None, **kwargs):
    row = {
        "id": graph_id,
        "heuristics": heuristic_name,
        "n": n,
        "m": m
    }
    if penalty_param is not None:
        row["penalty_param"] = penalty_param
    row.update(kwargs)
    return row


def experiment(results_path="results.csv", verbose=False):
    experiment_start = time.time()
    logger.info("Starting experiment")
    
    # Read datasets
    datasets = read_datasets()
    df_rows = []
    
    all_graphs = []
    for dataset_type, graphs in datasets.items():
        for filename, graph in graphs.items():
            graph_id = f"{dataset_type}_{filename}"
            all_graphs.append((graph_id, graph))
    
    logger.info(f"Total graphs to process: {len(all_graphs)}")
    
    for i, (graph_id, G) in enumerate(all_graphs):
        graph_start = time.time()
        logger.info(f"Processing graph {i+1}/{len(all_graphs)}: {graph_id}")
        
        n = G.number_of_nodes()
        m = G.number_of_edges()
        
        if n == 0:
            logger.warning(f"Skipping graph {graph_id}: empty graph")
            continue
            
        try:
            # Generate QUBO
            qubo_start = time.time()
            model, bqm, penalty_param = generate_qubo_for_MBSP(G)
            qubo_time = time.time() - qubo_start
            logger.info(f"QUBO generation for {graph_id} completed in {qubo_time:.3f}s")
            
            # Run heuristics
            heuristics_start = time.time()
            heuristics = run_heuristics(G, n, bqm, model, verbose=verbose)
            heuristics_time = time.time() - heuristics_start
            logger.info(f"Heuristics for {graph_id} completed in {heuristics_time:.3f}s")
            
            # Create result rows
            df_rows.append(create_result_row(graph_id, "Greedy", n, m, val=heuristics["Greedy"]))
            df_rows.append(create_result_row(graph_id, "Spectral", n, m, val=heuristics["Spectral"]))
            
            for solver_name in ["SA", "TB"]:
                df_rows.append(create_result_row(
                    graph_id, solver_name, n, m, penalty_param,
                    feasible_rate=heuristics[solver_name]["feasibility"],
                    average=heuristics[solver_name]["average"],
                    std=heuristics[solver_name]["std"],
                    range=heuristics[solver_name]["range"]
                ))
            
            graph_time = time.time() - graph_start
            logger.info(f"Graph {graph_id} processing completed in {graph_time:.3f}s")
                
        except Exception as e:
            logger.error(f"Error processing {graph_id}: {e}")
            continue
    
    # Save results
    save_start = time.time()
    df = pd.DataFrame(df_rows)
    os.makedirs(os.path.dirname(results_path), exist_ok=True) if os.path.dirname(results_path) else None
    df.to_csv(results_path, index=False)
    save_time = time.time() - save_start
    logger.info(f"Results saved to {results_path} in {save_time:.3f}s")
    
    total_time = time.time() - experiment_start
    logger.info(f"Experiment completed in {total_time:.3f}s")


if __name__ == "__main__":
    experiment(verbose=True)

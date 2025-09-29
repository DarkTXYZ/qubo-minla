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
import argparse
import json


def create_config():
    """Create configuration from command line arguments"""
    parser = argparse.ArgumentParser(description='QUBO MinLA Experiment Configuration')
    
    # Dataset configuration
    parser.add_argument('--synthetic-dir', type=str, default='synthetic-dataset',
                        help='Path to synthetic dataset directory')
    parser.add_argument('--real-world-dir', type=str, default='real-world-dataset',
                        help='Path to real-world dataset directory')
    
    # Solver configuration
    parser.add_argument('--num-simulate', type=int, default=10,
                        help='Number of simulation runs for SA and TB solvers')
    parser.add_argument('--sa-sweeps', type=int, default=1000,
                        help='Number of sweeps for Simulated Annealing')
    parser.add_argument('--sa-reads', type=int, default=1,
                        help='Number of reads for Simulated Annealing')
    parser.add_argument('--tb-tenure', type=int, default=20,
                        help='Tenure parameter for Tabu Search')
    
    # Output configuration
    parser.add_argument('--results-path', type=str, default='results.csv',
                        help='Output path for results CSV file')
    parser.add_argument('--log-file', type=str, default='experiment_log.txt',
                        help='Log file path')
    
    # Logging configuration
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    
    # Processing configuration
    parser.add_argument('--skip-datasets', type=str, nargs='*', default=[],
                        help='Dataset types to skip (synthetic, real_world)')
    parser.add_argument('--max-nodes', type=int, default=None,
                        help='Maximum number of nodes to process (skip larger graphs)')
    parser.add_argument('--skip-heuristics', type=str, nargs='*', default=[],
                        choices=['Greedy', 'Spectral', 'SA', 'TB'],
                        help='Heuristics to skip')
    
    # Configuration file support
    parser.add_argument('--config', type=str, default=None,
                        help='Path to JSON configuration file')
    
    args = parser.parse_args()
    
    # Load configuration from file if provided
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
            # Update args with config file values (command line takes precedence)
            for key, value in config_dict.items():
                if not hasattr(args, key) or getattr(args, key) == parser.get_default(key):
                    setattr(args, key, value)
    
    return args


def setup_logging(config):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config.log_file),
            logging.StreamHandler()
        ],
        force=True  # Override existing configuration
    )
    return logging.getLogger(__name__)


def read_graph_from_edge_list(file_path: str) -> nx.Graph:
    start_time = time.time()
    G = nx.Graph()
    with open(file_path, 'r') as f:
        for line in f:
            u, v = map(int, line.strip().split())
            G.add_edge(u, v)
    elapsed = time.time() - start_time
    logging.info(f"Graph loaded from {file_path} in {elapsed:.3f}s")
    return G


def read_datasets(config):
    start_time = time.time()
    logging.info("Starting dataset reading...")
    
    datasets = {}
    dataset_dirs = {
        "synthetic": config.synthetic_dir,
        "real_world": config.real_world_dir
    }
    
    for key, dir_path in dataset_dirs.items():
        if key in config.skip_datasets:
            logging.info(f"Skipping dataset type: {key}")
            continue
            
        if not os.path.exists(dir_path):
            logging.warning(f"Directory {dir_path} not found")
            continue
            
        datasets[key] = {}
        for filename in os.listdir(dir_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(dir_path, filename)
                logging.info(f"Reading dataset: {file_path}")
                G = read_graph_from_edge_list(file_path)
                
                # Skip graphs that are too large if max_nodes is specified
                if config.max_nodes and G.number_of_nodes() > config.max_nodes:
                    logging.info(f"Skipping {filename}: {G.number_of_nodes()} nodes > {config.max_nodes}")
                    continue
                    
                datasets[key][filename] = G
                logging.info(f"  Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    total_elapsed = time.time() - start_time
    logging.info(f"Dataset reading completed in {total_elapsed:.3f}s")
    return datasets


def generate_qubo_for_MBSP(G):
    start_time = time.time()
    logging.debug(f"Generating QUBO for graph with {G.number_of_nodes()} nodes")
    
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
    logging.debug(f"QUBO generation completed in {elapsed:.3f}s")
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
    logging.debug(f"Greedy MinLA completed in {elapsed:.3f}s, result: {result}")
    return result


def spectral_minla(G):
    start_time = time.time()
    
    if G.number_of_nodes() < 2:
        elapsed = time.time() - start_time
        logging.debug(f"Spectral MinLA (trivial case) completed in {elapsed:.3f}s")
        return 0
        
    L = nx.laplacian_matrix(G).astype(float)
    vals, vecs = eigsh(L, k=2, which='SM')
    fiedler = vecs[:, 1]
    node_order = [n for _, n in sorted(zip(fiedler, G.nodes()))]
    label_map = {node: idx + 1 for idx, node in enumerate(node_order)}
    
    result = sum(abs(label_map[u] - label_map[v]) for u, v in G.edges())
    elapsed = time.time() - start_time
    logging.debug(f"Spectral MinLA completed in {elapsed:.3f}s, result: {result}")
    return result


def run_solver(solver, bqm, model, n, num_simulate, desc, **solver_params):
    start_time = time.time()
    logging.debug(f"Starting {desc} with {num_simulate} simulations")
    
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
            logging.warning(f"Error in {desc} iteration {i}: {e}")
            results.append((False, float('inf')))
        
        iter_elapsed = time.time() - iter_start
        logging.debug(f"{desc} iteration {i+1}/{num_simulate} completed in {iter_elapsed:.3f}s")
    
    total_elapsed = time.time() - start_time
    logging.info(f"{desc} completed in {total_elapsed:.3f}s")
    return results


def calculate_stats(values):
    if len(values) == 0:
        return 0.0, 0.0, (0, 0)
    values_array = np.array(values)
    return float(np.mean(values_array)), float(np.std(values_array)), (int(np.min(values_array)), int(np.max(values_array)))


def run_heuristics(G, n, bqm, model, config):
    start_time = time.time()
    graph_id = f"n{n}_m{G.number_of_edges()}"
    logging.info(f"Running heuristics for graph {graph_id}")
    
    results = {}
    
    # Greedy
    if "Greedy" not in config.skip_heuristics:
        greedy_start = time.time()
        val_greedy = greedy_minla(G)
        greedy_time = time.time() - greedy_start
        results["Greedy"] = val_greedy
        logging.info(f"Greedy completed in {greedy_time:.3f}s")
    
    # Spectral
    if "Spectral" not in config.skip_heuristics:
        spectral_start = time.time()
        val_spectral = spectral_minla(G)
        spectral_time = time.time() - spectral_start
        results["Spectral"] = val_spectral
        logging.info(f"Spectral completed in {spectral_time:.3f}s")
    
    # SA Solver
    if "SA" not in config.skip_heuristics:
        sa_start = time.time()
        SA_solver = neal.SimulatedAnnealingSampler()
        SA_results = run_solver(SA_solver, bqm, model, n, config.num_simulate, "running SA", 
                               num_sweeps=config.sa_sweeps, num_reads=config.sa_reads)
        sa_time = time.time() - sa_start
        logging.info(f"SA solver completed in {sa_time:.3f}s")
        
        feasible_results = [val for feasible, val in SA_results if feasible]
        feasibility_rate = float(np.mean([feasible for feasible, _ in SA_results]))
        avg, std, range_vals = calculate_stats(feasible_results)
        
        results["SA"] = {
            "feasibility": feasibility_rate,
            "average": avg,
            "std": std,
            "range": range_vals
        }
        logging.info(f"SA: feasibility={feasibility_rate:.3f}, avg={avg:.2f}")
    
    # TB Solver
    if "TB" not in config.skip_heuristics:
        tb_start = time.time()
        TB_solver = TabuSampler()
        TB_results = run_solver(TB_solver, bqm, model, n, config.num_simulate, "running TB",
                               tenure=config.tb_tenure, num_reads=1)
        tb_time = time.time() - tb_start
        logging.info(f"TB solver completed in {tb_time:.3f}s")
        
        feasible_results = [val for feasible, val in TB_results if feasible]
        feasibility_rate = float(np.mean([feasible for feasible, _ in TB_results]))
        avg, std, range_vals = calculate_stats(feasible_results)
        
        results["TB"] = {
            "feasibility": feasibility_rate,
            "average": avg,
            "std": std,
            "range": range_vals
        }
        logging.info(f"TB: feasibility={feasibility_rate:.3f}, avg={avg:.2f}")
    
    total_elapsed = time.time() - start_time
    logging.info(f"All heuristics for graph {graph_id} completed in {total_elapsed:.3f}s")
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


def experiment(config):
    experiment_start = time.time()
    logger = setup_logging(config)
    logger.info("Starting experiment with configuration:")
    logger.info(f"  Results path: {config.results_path}")
    logger.info(f"  Number of simulations: {config.num_simulate}")
    logger.info(f"  SA sweeps: {config.sa_sweeps}")
    logger.info(f"  SA reads: {config.sa_reads}")
    logger.info(f"  TB tenure: {config.tb_tenure}")
    logger.info(f"  Skip datasets: {config.skip_datasets}")
    logger.info(f"  Skip heuristics: {config.skip_heuristics}")
    if config.max_nodes:
        logger.info(f"  Max nodes: {config.max_nodes}")
    
    # Read datasets
    datasets = read_datasets(config)
    
    # sort datasets by number of nodes
    for dataset_type in datasets:
        datasets[dataset_type] = dict(sorted(datasets[dataset_type].items(), key=lambda item: item[1].number_of_nodes()))
        
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
            heuristics = run_heuristics(G, n, bqm, model, config)
            heuristics_time = time.time() - heuristics_start
            logger.info(f"Heuristics for {graph_id} completed in {heuristics_time:.3f}s")
            
            # Create result rows
            if "Greedy" in heuristics:
                df_rows.append(create_result_row(graph_id, "Greedy", n, m, val=heuristics["Greedy"]))
            if "Spectral" in heuristics:
                df_rows.append(create_result_row(graph_id, "Spectral", n, m, val=heuristics["Spectral"]))
            
            for solver_name in ["SA", "TB"]:
                if solver_name in heuristics:
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
    os.makedirs(os.path.dirname(config.results_path), exist_ok=True) if os.path.dirname(config.results_path) else None
    df.to_csv(config.results_path, index=False)
    save_time = time.time() - save_start
    logger.info(f"Results saved to {config.results_path} in {save_time:.3f}s")
    
    total_time = time.time() - experiment_start
    logger.info(f"Experiment completed in {total_time:.3f}s")


if __name__ == "__main__":
    config = create_config()
    experiment(config)

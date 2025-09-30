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
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import pickle
from functools import lru_cache
import multiprocessing as mp


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
    
    # Performance configuration
    parser.add_argument('--parallel-graphs', action='store_true',
                        help='Process graphs in parallel')
    parser.add_argument('--parallel-solvers', action='store_true',
                        help='Run solver iterations in parallel')
    parser.add_argument('--cache-qubo', action='store_true',
                        help='Cache QUBO generation results')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Number of worker processes (default: CPU count)')
    
    # Output configuration
    parser.add_argument('--results-path', type=str, default='exp_2_results.csv',
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
    
    # Set default number of workers
    if args.num_workers is None:
        args.num_workers = min(mp.cpu_count(), 8)  # Cap at 8 to avoid resource exhaustion
    
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
        force=True
    )
    return logging.getLogger(__name__)


def read_graph_from_edge_list(file_path: str) -> nx.Graph:
    """Optimized graph reading using numpy for faster parsing"""
    start_time = time.time()
    
    # Try to read with numpy for faster parsing
    try:
        edges = np.loadtxt(file_path, dtype=int)
        if edges.ndim == 1:
            edges = edges.reshape(1, -1)
        G = nx.Graph()
        G.add_edges_from(edges)
    except:
        # Fallback to original method
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
    
    def load_single_graph(args):
        key, filename, file_path, max_nodes = args
        try:
            G = read_graph_from_edge_list(file_path)
            if max_nodes and G.number_of_nodes() > max_nodes:
                logging.info(f"Skipping {filename}: {G.number_of_nodes()} nodes > {max_nodes}")
                return None
            return (key, filename, G)
        except Exception as e:
            logging.error(f"Error loading {filename}: {e}")
            return None
    
    # Collect all file paths
    file_args = []
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
                file_args.append((key, filename, file_path, config.max_nodes))
    
    # Load graphs in parallel
    with ThreadPoolExecutor(max_workers=config.num_workers) as executor:
        results = list(tqdm(executor.map(load_single_graph, file_args), 
                           total=len(file_args), desc="Loading graphs"))
    
    # Organize results
    for result in results:
        if result is not None:
            key, filename, G = result
            if key not in datasets:
                datasets[key] = {}
            datasets[key][filename] = G
            logging.info(f"  Loaded graph {filename} with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    total_elapsed = time.time() - start_time
    logging.info(f"Dataset reading completed in {total_elapsed:.3f}s")
    return datasets


@lru_cache(maxsize=128)
def param_for_general_graph_cached(n, m):
    """Cached version of parameter calculation"""
    naive = m * (n - 1)
    complete = n * (n - 1) * (n + 1) / 6
    k = math.ceil(n + 1/2 - 1/2 * math.sqrt(8*m + 1))
    f = (n - k) * (n - k + 1) / 2
    edges_method = (m - f) * (k - 1) + (n - k) * (n**2 + (n + 3) * k - 2*k**2 - 1) / 6
    return min(naive, complete, edges_method)


def generate_qubo_for_MBSP(G, cache_enabled=False):
    start_time = time.time()
    V = G.number_of_nodes()

    E = list(G.edges())
    E = [tuple(map(int, edge)) for edge in E]

    # Use cached parameter calculation
    _lambda = param_for_general_graph_cached(V, len(E))
    
    # Check cache if enabled
    cache_key = f"qubo_{V}_{len(E)}_{hash(tuple(sorted(E)))}"
    cache_file = f".cache_{cache_key}.pkl"
    
    if cache_enabled and os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                model, bqm = pickle.load(f)
            logging.debug(f"Loaded QUBO from cache in {time.time() - start_time:.3f}s")
            return model, bqm, _lambda
        except:
            pass
    
    # Generate QUBO
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
    
    H = objective + thermometer_encoding(_lambda) + bijective(_lambda)
    model = H.compile()
    bqm = model.to_bqm()
    
    # Save to cache if enabled
    if cache_enabled:
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump((model, bqm), f)
        except:
            pass
    
    elapsed = time.time() - start_time
    logging.debug(f"QUBO generation completed in {elapsed:.3f}s")
    return model, bqm, _lambda


def param_for_general_graph(n, m):
    return param_for_general_graph_cached(n, m)


def get_best_sample(decoded_samples, V):
    if not decoded_samples:
        return False, float('inf'), []
        
    best_sample = min(decoded_samples, key=lambda s: s.energy)
    data = best_sample.sample
    
    # Vectorized constraint checking
    row_violations = 0
    col_violations = 0
    
    for u in range(V):
        for k in range(V - 1):
            if data.get(f'B[{u}][{k}]', 0) == 0 and data.get(f'B[{u}][{k+1}]', 0) == 1:
                row_violations += 1
    
    for k in range(V):
        col_sum = sum(data.get(f'B[{u}][{k}]', 0) for u in range(V))
        if col_sum != (V - k):
            col_violations += 1
    
    feasible = (row_violations == 0 and col_violations == 0)
    values = [sum(data.get(f'B[{u}][{k}]', 0) for k in range(V)) for u in range(V)]
    
    return feasible, best_sample.energy, values


def greedy_minla(G):
    start_time = time.time()
    
    V = G.number_of_nodes()
    if V == 0:
        return 0
        
    # Pre-compute degrees
    degrees = dict(G.degree())
    
    # Use numpy for faster operations
    labels = np.arange(1, V + 1)
    assigned = {}
    unassigned = set(G.nodes())
    
    for label in labels:
        min_deg_node = min(unassigned, key=lambda u: degrees[u])
        assigned[min_deg_node] = label
        unassigned.remove(min_deg_node)
    
    # Vectorized calculation
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
        
    L = nx.laplacian_matrix(G)
    vals, vecs = eigsh(L, k=2, which='SM', tol=1e-6)
    fiedler = vecs[:, 1]
    
    # Faster sorting using numpy
    sorted_indices = np.argsort(fiedler)
    nodes = list(G.nodes())
    node_order = [nodes[i] for i in sorted_indices]
    label_map = {node: idx + 1 for idx, node in enumerate(node_order)}
    
    result = sum(abs(label_map[u] - label_map[v]) for u, v in G.edges())
    elapsed = time.time() - start_time
    logging.debug(f"Spectral MinLA completed in {elapsed:.3f}s, result: {result}")
    return result


def run_single_solver_iteration(args):
    """Single solver iteration for parallel execution"""
    solver_type, bqm_data, model_data, n, solver_params, iteration = args
    
    try:
        # Recreate objects (needed for multiprocessing)
        if solver_type == "SA":
            solver = neal.SimulatedAnnealingSampler()
        else:  # TB
            solver = TabuSampler()
            
        # Note: This is a simplified version - you might need to serialize/deserialize BQM and model
        # For now, we'll use the threaded approach which is safer
        sampleset = solver.sample(bqm_data, **solver_params)
        decoded_samples = model_data.decode_sampleset(sampleset)
        feasible, val, _ = get_best_sample(decoded_samples, n)
        return (feasible, val)
    except Exception as e:
        logging.warning(f"Error in solver iteration {iteration}: {e}")
        return (False, float('inf'))


def run_solver(solver, bqm, model, n, num_simulate, desc, parallel=False, **solver_params):
    start_time = time.time()
    logging.debug(f"Starting {desc} with {num_simulate} simulations")
    
    if parallel and num_simulate > 1:
        # Use ThreadPoolExecutor for solver iterations (safer than ProcessPoolExecutor for complex objects)
        with ThreadPoolExecutor(max_workers=min(4, num_simulate)) as executor:
            futures = []
            for i in range(num_simulate):
                future = executor.submit(lambda: solver.sample(bqm, **solver_params))
                futures.append(future)
            
            results = []
            for i, future in enumerate(tqdm(futures, desc=desc, unit="its", leave=False)):
                try:
                    sampleset = future.result(timeout=300)  # 5 minute timeout
                    decoded_samples = model.decode_sampleset(sampleset)
                    feasible, val, _ = get_best_sample(decoded_samples, n)
                    results.append((feasible, val))
                except Exception as e:
                    logging.warning(f"Error in {desc} iteration {i}: {e}")
                    results.append((False, float('inf')))
    else:
        # Sequential execution
        results = []
        solver_iter = tqdm(range(num_simulate), desc=desc, unit="its", leave=False)
        
        for i in solver_iter:
            try:
                sampleset = solver.sample(bqm, **solver_params)
                decoded_samples = model.decode_sampleset(sampleset)
                feasible, val, _ = get_best_sample(decoded_samples, n)
                results.append((feasible, val))
            except Exception as e:
                logging.warning(f"Error in {desc} iteration {i}: {e}")
                results.append((False, float('inf')))
    
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
    
    # Run classical heuristics in parallel if beneficial
    classical_tasks = []
    if "Greedy" not in config.skip_heuristics:
        classical_tasks.append(("Greedy", lambda: greedy_minla(G)))
    if "Spectral" not in config.skip_heuristics:
        classical_tasks.append(("Spectral", lambda: spectral_minla(G)))
    
    # Execute classical heuristics
    if classical_tasks:
        if len(classical_tasks) > 1:
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = {name: executor.submit(func) for name, func in classical_tasks}
                for name, future in futures.items():
                    try:
                        results[name] = future.result(timeout=60)
                        logging.info(f"{name} completed")
                    except Exception as e:
                        logging.error(f"Error in {name}: {e}")
        else:
            name, func = classical_tasks[0]
            results[name] = func()
            logging.info(f"{name} completed")
    
    # SA Solver
    if "SA" not in config.skip_heuristics:
        sa_start = time.time()
        SA_solver = neal.SimulatedAnnealingSampler()
        SA_results = run_solver(SA_solver, bqm, model, n, config.num_simulate, "running SA",
                               parallel=config.parallel_solvers,
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
                               parallel=config.parallel_solvers,
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


def process_single_graph(args):
    """Process a single graph - for parallel execution"""
    graph_id, G, config = args
    
    try:
        n = G.number_of_nodes()
        m = G.number_of_edges()
        
        if n == 0:
            logging.warning(f"Skipping graph {graph_id}: empty graph")
            return []
        
        # Generate QUBO
        model, bqm, penalty_param = generate_qubo_for_MBSP(G, config.cache_qubo)

        # Run heuristics
        heuristics = run_heuristics(G, n, bqm, model, config)
        
        # Create result rows
        rows = []
        if "Greedy" in heuristics:
            rows.append(create_result_row(graph_id, "Greedy", n, m, val=heuristics["Greedy"]))
        if "Spectral" in heuristics:
            rows.append(create_result_row(graph_id, "Spectral", n, m, val=heuristics["Spectral"]))
        
        for solver_name in ["SA", "TB"]:
            if solver_name in heuristics:
                rows.append(create_result_row(
                    graph_id, solver_name, n, m, penalty_param,
                    feasible_rate=heuristics[solver_name]["feasibility"],
                    average=heuristics[solver_name]["average"],
                    std=heuristics[solver_name]["std"],
                    range=heuristics[solver_name]["range"]
                ))
        
        return rows
        
    except Exception as e:
        logging.error(f"Error processing {graph_id}: {e}")
        return []


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
    logger.info(f"  Parallel graphs: {config.parallel_graphs}")
    logger.info(f"  Parallel solvers: {config.parallel_solvers}")
    logger.info(f"  Cache QUBO: {config.cache_qubo}")
    logger.info(f"  Number of workers: {config.num_workers}")
    
    # Read datasets
    datasets = read_datasets(config)
    
    # Sort datasets by number of nodes
    for dataset_type in datasets:
        datasets[dataset_type] = dict(sorted(datasets[dataset_type].items(), 
                                           key=lambda item: item[1].number_of_nodes()))
    
    # Prepare all graphs
    all_graphs = []
    for dataset_type, graphs in datasets.items():
        for filename, graph in graphs.items():
            graph_id = f"{dataset_type}_{filename}"
            all_graphs.append((graph_id, graph, config))
    
    logger.info(f"Total graphs to process: {len(all_graphs)}")
    
    # Process graphs
    if config.parallel_graphs and len(all_graphs) > 1:
        # Parallel processing of graphs
        with ProcessPoolExecutor(max_workers=config.num_workers) as executor:
            results = list(tqdm(executor.map(process_single_graph, all_graphs),
                               total=len(all_graphs), desc="Processing graphs"))
        df_rows = [row for result in results for row in result]
    else:
        # Sequential processing
        df_rows = []
        for i, graph_args in enumerate(all_graphs):
            logger.info(f"Processing graph {i+1}/{len(all_graphs)}: {graph_args[0]}")
            rows = process_single_graph(graph_args)
            df_rows.extend(rows)
    
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

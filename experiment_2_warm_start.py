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
from scipy.sparse import csgraph
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
    parser = argparse.ArgumentParser(description='QUBO MinLA Experiment Configuration (Warm Start)')
    
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
    parser.add_argument('--results-path', type=str, default='exp_2_warm_start_results.csv',
                        help='Output path for results CSV file')
    parser.add_argument('--log-file', type=str, default='experiment_warm_start_log.txt',
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
                        choices=['Greedy', 'Spectral', 'SA_warm', 'TB_warm'],
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


def get_best_sample_from_sampleset(sampleset, V):
    """Get best sample from sampleset using the warm start method"""
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


def greedy_minla(G):
    """MBSP_greedy from warm start implementation"""
    start_time = time.time()
    
    V = G.number_of_nodes()
    if V == 0:
        return 0
        
    degrees = dict(G.degree())
    ordered = sorted(G.nodes(), key=degrees.get)
    assigned = {node: i + 1 for i, node in enumerate(ordered)}
    val = sum(abs(assigned[u] - assigned[v]) for u, v in G.edges())
    
    elapsed = time.time() - start_time
    logging.debug(f"Greedy MinLA completed in {elapsed:.3f}s, result: {val}")
    return val


def spectral_minla(G):
    """Enhanced spectral_minla that returns both value and order"""
    start_time = time.time()
    
    if G.number_of_nodes() < 2:
        elapsed = time.time() - start_time
        logging.debug(f"Spectral MinLA (trivial case) completed in {elapsed:.3f}s")
        return 0, list(G.nodes())
    
    # Faster Laplacian construction
    A = nx.to_scipy_sparse_array(G, format='csr', dtype=np.float64)
    L = csgraph.laplacian(A, normed=False)
    vals, vecs = eigsh(L, k=2, which='SM')  # second smallest eigenvector
    fiedler = vecs[:, 1]
    node_order = [n for _, n in sorted(zip(fiedler, G.nodes()))]
    label_map = {node: idx + 1 for idx, node in enumerate(node_order)}
    val = sum(abs(label_map[u] - label_map[v]) for u, v in G.edges())
    
    elapsed = time.time() - start_time
    logging.debug(f"Spectral MinLA completed in {elapsed:.3f}s, result: {val}")
    return val, node_order


def generate_warm_state(G, spectral_order):
    """Generate warm start state from spectral ordering"""
    n = G.number_of_nodes()
    
    # Build warm-start initial state from spectral_order
    label_map = {node: idx + 1 for idx, node in enumerate(spectral_order)}
    warm_state = {}
    for u in range(n):
        L = label_map[u]
        # B[u][k] = 1 if L > k else 0
        for k in range(n):
            warm_state[f'B[{u}][{k}]'] = 1 if L > k else 0
    
    return warm_state


def run_solver_with_warm_start(solver, bqm, model, n, num_simulate, desc, warm_state, parallel=False, **solver_params):
    """Run solver with warm start initial states"""
    start_time = time.time()
    logging.debug(f"Starting {desc} with {num_simulate} simulations (warm start)")
    
    # Add warm start parameters to solver params
    solver_params['initial_states'] = [warm_state]
    solver_params['initial_states_generator'] = 'tile'
    
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
                    feasible, val, _ = get_best_sample_from_sampleset(sampleset, n)
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
                feasible, val, _ = get_best_sample_from_sampleset(sampleset, n)
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
    
    # Run classical heuristics
    if "Greedy" not in config.skip_heuristics:
        val_greedy = greedy_minla(G)
        results["Greedy"] = val_greedy
        logging.info(f"Greedy completed: {val_greedy}")
    
    if "Spectral" not in config.skip_heuristics:
        val_spectral_order, spectral_order = spectral_minla(G)
        results["Spectral"] = val_spectral_order
        logging.info(f"Spectral completed: {val_spectral_order}")
    else:
        # Still need spectral order for warm start even if we skip reporting it
        _, spectral_order = spectral_minla(G)
    
    # Generate warm start state
    warm_state = generate_warm_state(G, spectral_order)
    
    # SA Solver with warm start
    if "SA_warm" not in config.skip_heuristics:
        sa_start = time.time()
        SA_solver = neal.SimulatedAnnealingSampler()
        SA_results = run_solver_with_warm_start(
            SA_solver, bqm, model, n, config.num_simulate, "running SA (warm)",
            warm_state, parallel=config.parallel_solvers,
            num_sweeps=config.sa_sweeps, num_reads=config.sa_reads
        )
        sa_time = time.time() - sa_start
        logging.info(f"SA warm solver completed in {sa_time:.3f}s")
        
        feasible_results = [val for feasible, val in SA_results if feasible]
        feasibility_rate = float(np.mean([feasible for feasible, _ in SA_results]))
        avg, std, range_vals = calculate_stats(feasible_results)
        
        results["SA_warm"] = {
            "feasibility": feasibility_rate,
            "average": avg,
            "std": std,
            "range": range_vals
        }
        logging.info(f"SA (warm-start): feasibility={feasibility_rate:.3f}, avg={avg:.2f}")
    
    # TB Solver with warm start
    if "TB_warm" not in config.skip_heuristics:
        tb_start = time.time()
        TB_solver = TabuSampler()
        TB_results = run_solver_with_warm_start(
            TB_solver, bqm, model, n, config.num_simulate, "running TB (warm)",
            warm_state, parallel=config.parallel_solvers,
            tenure=config.tb_tenure, num_reads=1
        )
        tb_time = time.time() - tb_start
        logging.info(f"TB warm solver completed in {tb_time:.3f}s")
        
        feasible_results = [val for feasible, val in TB_results if feasible]
        feasibility_rate = float(np.mean([feasible for feasible, _ in TB_results]))
        avg, std, range_vals = calculate_stats(feasible_results)
        
        results["TB_warm"] = {
            "feasibility": feasibility_rate,
            "average": avg,
            "std": std,
            "range": range_vals
        }
        logging.info(f"TB (warm-start): feasibility={feasibility_rate:.3f}, avg={avg:.2f}")
    
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
        
        for solver_name in ["SA_warm", "TB_warm"]:
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
    logger.info("Starting warm start experiment with configuration:")
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
    logger.info(f"Warm start experiment completed in {total_time:.3f}s")


if __name__ == "__main__":
    config = create_config()
    experiment(config)

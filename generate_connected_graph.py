# generate connected graph with V = [10, 50, 100, 200, 300] with edge prob = 0.3 and seed = 42.
# the output graph should be in text file and in edge list format e.g. 0 1 \n 0 2 \n 1 0
# the first line should contain number of vertices and number of edges

import networkx as nx
import os
import random

def generate_connected_graph(n, p=0.3, seed=42, max_attempts=100):
    """
    Generate a connected random graph using Erdős–Rényi model.
    If the generated graph is not connected, retry until we get a connected one.
    
    Args:
        n (int): Number of vertices
        p (float): Edge probability
        seed (int): Random seed
        max_attempts (int): Maximum attempts to generate connected graph
    
    Returns:
        networkx.Graph: Connected graph
    """
    random.seed(seed)
    
    for attempt in range(max_attempts):
        # Generate random graph
        G = nx.gnp_random_graph(n, p, seed=seed + attempt)
        
        # Check if connected
        if nx.is_connected(G):
            return G
    
    # If we couldn't generate a connected graph, create one manually
    print(f"Warning: Could not generate connected graph with p={p} after {max_attempts} attempts.")
    print(f"Generating connected graph by adding minimum edges to spanning tree...")
    
    # Start with a random spanning tree to ensure connectivity
    G = nx.random_tree(n, seed=seed)
    
    # Add additional random edges based on probability
    for u in range(n):
        for v in range(u + 1, n):
            if not G.has_edge(u, v) and random.random() < p:
                G.add_edge(u, v)
    
    return G

def save_graph_to_file(G, filename):
    """
    Save graph to file in the specified format.
    First line: number of vertices and number of edges
    Subsequent lines: edge list format
    
    Args:
        G (networkx.Graph): Graph to save
        filename (str): Output filename
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()
    
    with open(filename, 'w') as f:
        # First line: number of vertices and edges
        f.write(f"{n} {m}\n")
        
        # Edge list
        for u, v in G.edges():
            f.write(f"{u} {v}\n")

def generate_connected_graphs(node_sizes=[10, 50, 100, 200, 300], edge_prob=0.3, seed=42):
    """
    Generate connected random graphs for different node sizes.
    
    Args:
        node_sizes (list): List of node counts for each graph
        edge_prob (float): Edge probability for Erdős–Rényi model  
        seed (int): Random seed for reproducibility
    """
    # Create output directory if it doesn't exist
    output_dir = "synthetic-datasets"
    os.makedirs(output_dir, exist_ok=True)
    
    for i, n in enumerate(node_sizes):
        print(f"Generating connected graph with {n} vertices and edge probability {edge_prob}")
        
        # Use different seed for each graph to ensure variety
        graph_seed = seed + i * 1000
        
        # Generate connected graph
        G = generate_connected_graph(n, edge_prob, graph_seed)
        
        # Get number of edges
        m = G.number_of_edges()
        
        # Verify connectivity
        if not nx.is_connected(G):
            print(f"Error: Generated graph with {n} nodes is not connected!")
            continue
            
        # Define output filename
        filename = f"{output_dir}/random_graph_n{n}_p{edge_prob}_m{m}.txt"
        
        # Save graph to file
        save_graph_to_file(G, filename)
        
        print(f"  Saved: {filename} ({m} edges, connected: {nx.is_connected(G)})")

def main():
    """Main function to generate all connected graph datasets"""
    print("Generating connected graph datasets...")
    print("=" * 50)
    
    generate_connected_graphs()
    
    print("=" * 50)
    print("Connected graph generation complete!")
    
    # List generated files
    output_dir = "synthetic-dataset"
    if os.path.exists(output_dir):
        print(f"\nGenerated files in '{output_dir}':")
        for filename in sorted(os.listdir(output_dir)):
            if filename.endswith('.txt'):
                filepath = os.path.join(output_dir, filename)
                with open(filepath, 'r') as f:
                    first_line = f.readline().strip()
                print(f"  {filename} - {first_line}")

if __name__ == "__main__":
    main()

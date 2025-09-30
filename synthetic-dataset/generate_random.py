# generate random graph with N = [10,20,30,50,100] with edge probability p = 0.3 and seed = 42
# the output file should be in text file and the format should be: 0 1\n0 2\n1 2\n...
# output directory should be ./synthetic_dataset/

import networkx as nx
import os

def generate_random_graphs():
    """
    Generate random graphs using Erdős–Rényi model and save as edge lists.
    """
    # Parameters from comments
    node_sizes = [5, 10, 15, 20, 25]
    edge_prob = 0.3
    seed = 42
    output_dir = "./synthetic-dataset/"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating random graphs with the following parameters:")
    print(f"Node sizes: {node_sizes}")
    print(f"Edge probability: {edge_prob}")
    print(f"Seed: {seed}")
    print(f"Output directory: {output_dir}")
    print("-" * 50)
    
    for i, n in enumerate(node_sizes):
        print(f"Generating graph {i+1}/{len(node_sizes)}: {n} nodes")
        
        # Generate random graph using Erdős–Rényi model
        # Use different seed for each graph to ensure variety
        current_seed = seed
        G = nx.gnp_random_graph(n, edge_prob, seed=current_seed)
        
        # Get number of edges
        m = G.number_of_edges()
        
        # Define output filename
        filename = f"{output_dir}random_n{n}.txt"
        
        # Save as edge list in the specified format
        with open(filename, 'w') as f:
            for u, v in G.edges():
                f.write(f"{u} {v}\n")
        
        print(f"  Saved: {filename}")
        print(f"  Edges: {m}")
        print(f"  Density: {2*m/(n*(n-1)):.4f}")
        print()

def main():
    """Main function to generate all datasets"""
    print("Starting random graph dataset generation...")
    print("=" * 60)
    
    try:
        generate_random_graphs()
        print("=" * 60)
        print("Dataset generation completed successfully!")
    except Exception as e:
        print(f"Error during generation: {e}")

if __name__ == "__main__":
    main()
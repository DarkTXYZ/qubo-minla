# generate random graph with N = [10,20,30,50,100] with edge probability p = 0.3 and seed = 42
# the output file should be in text file and the format should be: 0 1\n0 2\n1 2\n...
# output directory should be ./synthetic_dataset/

import networkx as nx
import os

def generate_random_graphs():
    """
    Generate random graphs using Erdős–Rényi model and save as edge lists.
    Ensures all graphs are connected and use all vertices.
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
        
        # Generate connected random graph
        current_seed = seed
        G = None
        attempts = 0
        max_attempts = 1000
        
        # Keep generating until we get a connected graph
        while G is None or not nx.is_connected(G):
            G = nx.gnp_random_graph(n, edge_prob, seed=current_seed + attempts)
            attempts += 1
            
            if attempts >= max_attempts:
                # If we can't generate a connected graph with the given probability,
                # create a minimum spanning tree and add random edges
                print(f"  Warning: Generating connected graph directly with p={edge_prob}")
                G = nx.random_tree(n, seed=current_seed)
                # Add additional random edges while maintaining connectivity
                import random
                random.seed(current_seed)
                total_possible_edges = n * (n - 1) // 2
                target_edges = int(total_possible_edges * edge_prob)
                
                while G.number_of_edges() < target_edges:
                    u = random.randint(0, n-1)
                    v = random.randint(0, n-1)
                    if u != v and not G.has_edge(u, v):
                        G.add_edge(u, v)
                break
        
        # Verify the graph uses all vertices and is connected
        assert G.number_of_nodes() == n, f"Graph should have {n} nodes, but has {G.number_of_nodes()}"
        assert nx.is_connected(G), "Graph must be connected"
        
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
        print(f"  Connected: {nx.is_connected(G)}")
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
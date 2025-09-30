# Preprocess real-world datasets in raw data to ensure consistent 0-based indexing
# and proper edge list format

import os
import networkx as nx
import pandas as pd
import re

def preprocess_enzymes_file(input_file, output_file):
    """
    Preprocess ENZYMES dataset files.
    These files appear to be 1-based indexed, convert to 0-based.
    """
    print(f"Processing ENZYMES file: {input_file}")
    
    edges = []
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('%') and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        u, v = int(parts[0]), int(parts[1])
                        # Convert to 0-based indexing
                        edges.append((u-1, v-1))
                    except ValueError:
                        continue
    
    # Remove duplicate edges and self-loops
    edges = list(set(edges))
    edges = [(u, v) for u, v in edges if u != v]
    
    # Relabel nodes to ensure consecutive 0-based indexing
    G = nx.Graph()
    G.add_edges_from(edges)
    G = nx.convert_node_labels_to_integers(G, first_label=0)
    
    # Save preprocessed data
    with open(output_file, 'w') as f:
        for u, v in G.edges():
            f.write(f"{u} {v}\n")
    
    print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    return G.number_of_nodes(), G.number_of_edges()

def preprocess_iscas_file(input_file, output_file):
    """
    Preprocess ISCAS89 circuit files.
    These appear to be already 0-based indexed.
    """
    print(f"Processing ISCAS file: {input_file}")
    
    edges = []
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('%') and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        u, v = int(parts[0]), int(parts[1])
                        edges.append((u, v))
                    except ValueError:
                        continue
    
    # Remove duplicate edges and self-loops
    edges = list(set(edges))
    edges = [(u, v) for u, v in edges if u != v]
    
    # Relabel nodes to ensure consecutive 0-based indexing
    G = nx.Graph()
    G.add_edges_from(edges)
    G = nx.convert_node_labels_to_integers(G, first_label=0)
    
    # Save preprocessed data
    with open(output_file, 'w') as f:
        for u, v in G.edges():
            f.write(f"{u} {v}\n")
    
    print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    return G.number_of_nodes(), G.number_of_edges()

def preprocess_soc_tribes_file(input_file, output_file):
    """
    Preprocess social network (tribes) files.
    These files may have header lines and weight information.
    """
    print(f"Processing SOC-tribes file: {input_file}")
    
    edges = []
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('%') and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        u, v = int(parts[0]), int(parts[1])
                        # Convert to 0-based indexing (assuming 1-based input)
                        edges.append((u-1, v-1))
                    except ValueError:
                        continue
    
    # Remove duplicate edges and self-loops
    edges = list(set(edges))
    edges = [(u, v) for u, v in edges if u != v and u >= 0 and v >= 0]
    
    # Relabel nodes to ensure consecutive 0-based indexing
    G = nx.Graph()
    G.add_edges_from(edges)
    G = nx.convert_node_labels_to_integers(G, first_label=0)
    
    # Save preprocessed data
    with open(output_file, 'w') as f:
        for u, v in G.edges():
            f.write(f"{u} {v}\n")
    
    print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    return G.number_of_nodes(), G.number_of_edges()

def preprocess_generic_file(input_file, output_file):
    """
    Preprocess generic graph files.
    Assumes space-separated edge list format.
    """
    print(f"Processing generic file: {input_file}")
    
    edges = []
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('%') and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        u, v = int(parts[0]), int(parts[1])
                        edges.append((u, v))
                    except ValueError:
                        continue
    
    # Remove duplicate edges and self-loops
    edges = list(set(edges))
    edges = [(u, v) for u, v in edges if u != v]
    
    # Relabel nodes to ensure consecutive 0-based indexing
    G = nx.Graph()
    G.add_edges_from(edges)
    G = nx.convert_node_labels_to_integers(G, first_label=0)
    
    # Save preprocessed data
    with open(output_file, 'w') as f:
        for u, v in G.edges():
            f.write(f"{u} {v}\n")
    
    print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    return G.number_of_nodes(), G.number_of_edges()

def preprocess_all_datasets():
    """
    Main function to preprocess all real-world datasets.
    """
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data_dir = os.path.join(script_dir, "raw_data")
    output_dir = os.path.join(script_dir, "")
    
    # Check if raw_data directory exists
    if not os.path.exists(raw_data_dir):
        print(f"Error: raw_data directory not found at {raw_data_dir}")
        print("Please ensure the raw_data directory exists with dataset files.")
        return None
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("Starting preprocessing of real-world datasets...")
    print(f"Raw data directory: {raw_data_dir}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    results = []
    
    # Get all files in raw_data directory
    raw_files = [f for f in os.listdir(raw_data_dir) if f.endswith(('.edges', '.txt'))]
    
    for filename in sorted(raw_files):
        input_path = os.path.join(raw_data_dir, filename)
        
        # Generate output filename (remove extension and add .txt)
        base_name = os.path.splitext(filename)[0]
        output_path = os.path.join(output_dir, f"{base_name}_processed.txt")
        
        try:
            # Choose preprocessing function based on filename
            if "ENZYMES" in filename:
                nodes, edges = preprocess_enzymes_file(input_path, output_path)
            elif "iscas" in filename.lower():
                nodes, edges = preprocess_iscas_file(input_path, output_path)
            elif "soc-tribes" in filename:
                nodes, edges = preprocess_soc_tribes_file(input_path, output_path)
            else:
                nodes, edges = preprocess_generic_file(input_path, output_path)
            
            results.append({
                'original_file': filename,
                'processed_file': f"{base_name}_processed.txt",
                'nodes': nodes,
                'edges': edges,
                'status': 'success'
            })
            
        except Exception as e:
            print(f"  Error processing {filename}: {e}")
            results.append({
                'original_file': filename,
                'processed_file': 'N/A',
                'nodes': 0,
                'edges': 0,
                'status': f'error: {e}'
            })
        
        print()
    
    # Save summary
    summary_df = pd.DataFrame(results)
    summary_path = os.path.join(output_dir, "preprocessing_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    
    print("=" * 60)
    print("Preprocessing Summary:")
    print(summary_df.to_string(index=False))
    print(f"\nSummary saved to: {summary_path}")
    print(f"Processed files saved to: {output_dir}/")
    
    return results

def main():
    """Main execution function"""
    try:
        results = preprocess_all_datasets()
        print("\nPreprocessing completed successfully!")
        return results
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return None

if __name__ == "__main__":
    main()

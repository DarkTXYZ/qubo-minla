# change karate.txt to start with 0
# Preprocess real-world datasets to ensure consistent 0-based indexing
# and proper edge list format

import os
import networkx as nx

def preprocess_karate():
    """
    Convert karate.txt from 1-based to 0-based indexing
    """
    input_file = "raw_data/karate.txt"
    output_file = "karate.txt"
    
    print(f"Processing {input_file}...")
    
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) == 2:
                    # Convert from 1-based to 0-based indexing
                    u = int(parts[0]) - 1
                    v = int(parts[1]) - 1
                    f_out.write(f"{u} {v}\n")
    
    print(f"  Saved as {output_file} (converted to 0-based indexing)")

def preprocess_iscas89():
    """
    Process iscas89-s27.txt (already 0-based, just copy and verify)
    """
    input_file = "raw_data/iscas89-s27.txt"
    output_file = "iscas89-s27.txt"
    
    print(f"Processing {input_file}...")
    
    # Read and clean the file
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) == 2:
                    u = int(parts[0])
                    v = int(parts[1])
                    f_out.write(f"{u} {v}\n")
    
    print(f"  Saved as {output_file} (cleaned format)")

def preprocess_chesapeake():
    """
    Process chesapeake.mtx file - convert to standard edge list format
    """
    input_file = "raw_data/chesapeake.mtx"
    output_file = "chesapeake.txt"
    
    print(f"Processing {input_file}...")
    
    # Read the matrix file and convert to edge list
    edges = []
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) >= 2:
                    u = int(parts[0])
                    v = int(parts[1])
                    edges.append((u, v))
    
    # Find min node to adjust to 0-based indexing
    if edges:
        min_node = min(min(u, v) for u, v in edges)
        
        # Write edges with 0-based indexing
        with open(output_file, 'w') as f:
            for u, v in edges:
                u_adj = u - min_node
                v_adj = v - min_node
                f.write(f"{u_adj} {v_adj}\n")
        
        print(f"  Saved as {output_file} (converted to 0-based indexing)")
        print(f"  Original range: {min_node} to {max(max(u, v) for u, v in edges)}")
        print(f"  Adjusted range: 0 to {max(max(u, v) for u, v in edges) - min_node}")

def verify_datasets():
    """
    Verify the processed datasets and provide statistics
    """
    print("\nVerifying processed datasets:")
    print("=" * 50)
    
    datasets = ["karate.txt", "iscas89-s27.txt", "chesapeake.txt"]
    
    for dataset in datasets:
        if os.path.exists(dataset):
            print(f"\n{dataset}:")
            
            # Read edges
            edges = []
            with open(dataset, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) == 2:
                            u, v = int(parts[0]), int(parts[1])
                            edges.append((u, v))
            
            if edges:
                nodes = set()
                for u, v in edges:
                    nodes.add(u)
                    nodes.add(v)
                
                print(f"  Nodes: {len(nodes)} (range: {min(nodes)} to {max(nodes)})")
                print(f"  Edges: {len(edges)}")
                print(f"  Density: {2 * len(edges) / (len(nodes) * (len(nodes) - 1)):.4f}")
                
                # Check if 0-based
                if min(nodes) == 0:
                    print("  ✓ 0-based indexing")
                else:
                    print(f"  ⚠ Non-zero based indexing (starts at {min(nodes)})")
        else:
            print(f"  ⚠ {dataset} not found")

def main():
    """
    Main preprocessing function
    """
    print("Real-world Dataset Preprocessing")
    print("=" * 40)
    
    # Change to the real-world-dataset directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    try:
        # Process each dataset
        preprocess_karate()
        preprocess_iscas89()
        preprocess_chesapeake()
        
        # Verify results
        verify_datasets()
        
        print("\n" + "=" * 40)
        print("Preprocessing completed successfully!")
        
    except Exception as e:
        print(f"Error during preprocessing: {e}")

if __name__ == "__main__":
    main()

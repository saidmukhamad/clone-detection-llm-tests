import streamlit as st
import networkx as nx
import os
import hashlib
from typing import List, Tuple
from interactive_graph_component import interactive_graph

def calculate_similarity(file1: str, file2: str) -> float:
    """
    Calculate similarity between two files based on their content.
    This is a simple implementation using hash comparison.
    In a real-world scenario, you'd use more sophisticated methods.
    """
    with open(file1, 'rb') as f1, open(file2, 'rb') as f2:
        hash1 = hashlib.md5(f1.read()).hexdigest()
        hash2 = hashlib.md5(f2.read()).hexdigest()
    
    # Compare the first 8 characters of the hash
    similarity = sum(c1 == c2 for c1, c2 in zip(hash1[:8], hash2[:8])) / 8
    return similarity

def build_similarity_graph(files: List[str], threshold: float) -> nx.Graph:
    G = nx.Graph()
    
    for i, file1 in enumerate(files):
        G.add_node(os.path.basename(file1))
        for j, file2 in enumerate(files[i+1:], start=i+1):
            similarity = calculate_similarity(file1, file2)
            if similarity >= threshold:
                G.add_edge(os.path.basename(file1), os.path.basename(file2), weight=similarity)
    
    return G

def generate_report(G: nx.Graph) -> List[Tuple[str, str, float]]:
    report = []
    for edge in G.edges(data=True):
        file1, file2, data = edge
        report.append((file1, file2, data['weight']))
    return report

def graph_to_d3_format(G: nx.Graph) -> dict:
    return {
        "nodes": [{"id": node} for node in G.nodes()],
        "links": [{"source": u, "target": v, "value": d["weight"]} for u, v, d in G.edges(data=True)]
    }

# Streamlit UI
st.title("File Similarity Graph")

# File upload
uploaded_files = st.file_uploader("Choose files", accept_multiple_files=True)

if uploaded_files:
    # Save uploaded files temporarily
    temp_dir = "temp_files"
    os.makedirs(temp_dir, exist_ok=True)
    file_paths = []
    for file in uploaded_files:
        file_path = os.path.join(temp_dir, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        file_paths.append(file_path)

    # Similarity threshold slider
    threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.5, 0.01)

    if st.button("Generate Similarity Graph"):
        G = build_similarity_graph(file_paths, threshold)
        
        # Visualize graph using the custom component
        graph_data = graph_to_d3_format(G)
        interactive_graph(graph_data)

        # Generate and display report
        report = generate_report(G)
        st.subheader("Similarity Report")
        for file1, file2, similarity in report:
            st.write(f"{file1} - {file2}: {similarity:.2f}")

    # Cleanup temporary files
    for file_path in file_paths:
        os.remove(file_path)
        
    os.rmdir(temp_dir)
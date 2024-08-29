import streamlit as st
import networkx as nx
import plotly.graph_objects as go
import numpy as np
from scipy.spatial.distance import cosine

def calculate_similarity(file1_content, file2_content):
    """
    Calculate similarity between two files based on their content.
    """
    words1 = set(file1_content.lower().split())
    words2 = set(file2_content.lower().split())
    return len(words1.intersection(words2)) / len(words1.union(words2))

def build_similarity_graph(files, threshold):
    G = nx.Graph()
    
    for i, (file1, content1) in enumerate(files.items()):
        G.add_node(i, name=file1)
        for j, (file2, content2) in enumerate(files.items()):
            if i < j:  # Avoid duplicate comparisons
                similarity = calculate_similarity(content1, content2)
                if similarity >= threshold:
                    G.add_edge(i, j, weight=similarity)
    
    return G

def create_plotly_network(G):
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            )
        )
    )

    node_adjacencies = []
    node_text = []
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append(f"File: {G.nodes[node]['name']}<br># of connections: {len(adjacencies[1])}")

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0,l=0,r=0,t=0),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    ))
    
    return fig

st.title("Interactive File Similarity Graph")

uploaded_files = st.file_uploader("Choose text files", accept_multiple_files=True, type=['py'])

if uploaded_files:
    files = {}
    for file in uploaded_files:
        content = file.read().decode('utf-8')
        files[file.name] = content

    threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.5, 0.01)

    if st.button("Generate Similarity Graph"):
        G = build_similarity_graph(files, threshold)
        
        st.write(f"Number of files: {len(files)}")
        st.write(f"Number of similarities above threshold: {G.number_of_edges()}")
        
        fig = create_plotly_network(G)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Similarity Report")
        for edge in G.edges(data=True):
            file1 = G.nodes[edge[0]]['name']
            file2 = G.nodes[edge[1]]['name']
            similarity = edge[2]['weight']
            st.write(f"{file1} - {file2}: {similarity:.2f}")

st.sidebar.title("About")
st.sidebar.info(
    "This app demonstrates an interactive file similarity graph using Streamlit and Plotly. "
    "Upload text files, set a similarity threshold, and visualize the relationships between files."
)

# Add custom CSS to make the graph larger
st.markdown("""
    <style>
    .js-plotly-plot {
        height: 800px !important;
    }
    </style>
    """, unsafe_allow_html=True)
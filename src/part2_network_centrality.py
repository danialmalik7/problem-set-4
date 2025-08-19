'''
PART 2: NETWORK CENTRALITY METRICS

Using the imbd_movies dataset
- Build a graph and perform some rudimentary graph analysis, extracting centrality metrics from it. 
- Below is some basic code scaffolding that you will need to add to
- Tailor this code scaffolding and its stucture to however works to answer the problem
- Make sure the code is inline with the standards we're using in this class 
'''

import numpy as np
import pandas as pd
import networkx as nx
import json
import os
from datetime import datetime
from urllib.request import urlopen

def download_data():
    """Download the IMDB movie data."""
    url = "https://github.com/cbuntain/umd.inst414/blob/main/data/imdb_movies_2000to2022.prolific.json?raw=true"
    try:
        with urlopen(url) as response:
            content = response.read().decode('utf-8')
            try:
                data = json.loads(content)
                return data
            except json.JSONDecodeError:
                data = []
                for line in content.strip().split('\n'):
                    if line.strip():
                        try:
                            movie = json.loads(line.strip())
                            data.append(movie)
                        except:
                            continue
                return data
    except Exception as e:
        print(f"Error downloading data: {e}")
        return None

# Build the graph
g = nx.Graph()

def build_graph(movies_data):
    """Build the actor network graph."""
    for movie in movies_data:
        actors = movie.get('actors', [])
        
        # Create a node for every actor
        for actor_id, actor_name in actors:
            if not g.has_node(actor_id):
                g.add_node(actor_id, name=actor_name)
        
        # Iterate through the list of actors, generating all pairs
        # Starting with the first actor in the list, generate pairs with all subsequent actors
        # then continue to second actor in the list and repeat
        
        i = 0 #counter
        for left_actor_id, left_actor_name in actors:
            for right_actor_id, right_actor_name in actors[i+1:]:
                # Get the current weight, if it exists
                current_weight = g.get_edge_data(left_actor_id, right_actor_id, {}).get('weight', 0)
                
                # Add an edge for these actors
                g.add_edge(left_actor_id, right_actor_id, weight=current_weight + 1)
            i += 1

def calculate_centrality():
    """Calculate centrality metrics."""
    # Use the largest connected component to speed up centrality calculations
    if g.number_of_nodes() == 0:
        return []
    H = g
    try:
        # If graph is disconnected, use the largest connected component
        if not nx.is_connected(g):
            largest_cc = max(nx.connected_components(g), key=len)
            H = g.subgraph(largest_cc).copy()
    except nx.NetworkXError:
        # Fallback; if connectedness check fails, keep full graph
        H = g

    degree_centrality = nx.degree_centrality(H)

    # Fast, standard approximation: sample k pivot nodes for betweenness
    betweenness_centrality = nx.betweenness_centrality(
        H, k=500, normalized=True, seed=42
    )

    centrality_data = []
    for node in H.nodes():
        node_data = {
            'actor_id': node,
            'actor_name': H.nodes[node].get('name', ''),
            'degree_centrality': degree_centrality.get(node, 0),
            'betweenness_centrality': betweenness_centrality.get(node, 0),
            'degree': H.degree(node)
        }
        centrality_data.append(node_data)
    
    return centrality_data

# Set up your dataframe(s) -> the df that's output to a CSV should include at least the columns 'left_actor_name', '<->', 'right_actor_name'
def create_edges_df():
    """Create edges dataframe."""
    edges_data = []
    
    for edge in g.edges(data=True):
        left_actor_id = edge[0]
        right_actor_id = edge[1]
        weight = edge[2].get('weight', 1)
        
        left_actor_name = g.nodes[left_actor_id].get('name', '')
        right_actor_name = g.nodes[right_actor_id].get('name', '')
        
        edge_record = {
            'left_actor_name': left_actor_name,
            '<->': '<->',
            'right_actor_name': right_actor_name,
            'weight': weight
        }
        edges_data.append(edge_record)
    
    return pd.DataFrame(edges_data)

def save_results(centrality_df, edges_df):
    """Save results to CSV."""
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Output the final dataframe to a CSV named 'network_centrality_{current_datetime}.csv' to `/data`
    centrality_filename = f"network_centrality_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    centrality_path = os.path.join(data_dir, centrality_filename)
    centrality_df.to_csv(centrality_path, index=False)
    
    edges_filename = f"network_edges_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    edges_path = os.path.join(data_dir, edges_filename)
    edges_df.to_csv(edges_path, index=False)

def nc():
    """Main function for network centrality analysis."""
    print("Starting Network Centrality Analysis")
    
    movies_data = download_data()
    if not movies_data:
        return
    
    print(f"Downloaded {len(movies_data)} movies")
    
    build_graph(movies_data)
    
    # Print the info below
    print("Nodes:", len(g.nodes))
    
    # Print the 10 the most central nodes
    centrality_data = calculate_centrality()
    centrality_df = pd.DataFrame(centrality_data)
    
    top_central = centrality_df.nlargest(10, 'degree_centrality')
    print("\nTop 10 most central actors:")
    for _, row in top_central.iterrows():
        print(f"{row['actor_name']}: {row['degree_centrality']:.4f}")
    
    edges_df = create_edges_df()
    save_results(centrality_df, edges_df)
    
    print("Network centrality analysis completed")

if __name__ == "__main__":
    nc()

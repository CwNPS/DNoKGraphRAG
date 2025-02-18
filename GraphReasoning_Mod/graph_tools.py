"""
graph_tools.py

Based on https://github.com/lamm-mit/GraphReasoning/assets/101393859/3baa3752-8222-4857-a64c-c046693d6315

This module provides various tools for analyzing and visualizing large graphs.
It includes functions for calculating graph statistics, plotting degree distributions,
and detecting communities using the Louvain method.

Dependencies:
- networkx
- matplotlib
- community (python-louvain)
- tqdm
- sklearn
- numpy
- seaborn
- pandas
"""

import networkx as nx
import matplotlib.pyplot as plt
from community import community_louvain
from tqdm import tqdm
from copy import deepcopy
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import copy
import seaborn as sns
palette = "hls"
import random
import pandas as pd
import heapq
from scipy.spatial.distance import cosine
from pyvis.network import Network

def graph_statistics_and_plots_for_large_graphs (G, data_dir='./', include_centrality=False,
                                                 make_graph_plot=False,root='graph', log_scale=True, 
                                                 log_hist_scale=True,density_opt=False, bins=50,
                                                ):
    # Basic statistics
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    degrees = [degree for node, degree in G.degree()]
    log_degrees = np.log1p(degrees)  # Using log1p for a better handle on zero degrees
    #degree_distribution = np.bincount(degrees)
    average_degree = np.mean(degrees)
    density = nx.density(G)
    connected_components = nx.number_connected_components(G)
    
    # Centrality measures
    if include_centrality:
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)
        closeness_centrality = nx.closeness_centrality(G)
        eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
    
    # Community detection with Louvain method
    partition = community_louvain.best_partition(G)
    num_communities = len(set(partition.values()))

    # # Plotting
    # # Degree Distribution on a log-log scale
    # plt.figure(figsize=(10, 6))
     
    # if log_scale:
    #     counts, bins, patches = plt.hist(log_degrees, bins=bins, alpha=0.75, color='blue', log=log_hist_scale, density=density_opt)
    
    #     plt.xscale('log')
    #     plt.yscale('log')
    #     xlab_0='Log(1 + Degree)'
    #     if density_opt:
    #         ylab_0='Probability Distribution'
    #     else: 
    #         ylab_0='Probability Distribution'
    #     ylab_0=ylab_0 + log_hist_scale*' (log)'    
        
        
    #     plt_title='Histogram of Log-Transformed Node Degrees with Log-Log Scale'
        
    # else:
    #     counts, bins, patches = plt.hist(degrees, bins=bins, alpha=0.75, color='blue', log=log_hist_scale, density=density_opt)
    #     xlab_0='Degree'
    #     if density_opt:
    #         ylab_0='Probability Distribution'
    #     else: 
    #         ylab_0='Probability Distribution'
    #     ylab_0=ylab_0 + log_hist_scale*' (log)'     
    #     plt_title='Histogram of Node Degrees'

    
    
    # if make_graph_plot:
    #     plt.title(plt_title)
    #     plt.xlabel(xlab_0)
    #     plt.ylabel(ylab_0)
    #     plt.savefig(f'{data_dir}/{plt_title}_{root}.svg')
    #     plt.show()

    #     # Additional Plots
    #     # Plot community structure
    #     plt.figure(figsize=(10, 6))
    #     pos = nx.spring_layout(G)  # for better visualization
    #     cmap = plt.get_cmap('viridis')
    #     nx.draw_networkx(G, pos, node_color=list(partition.values()), node_size=20, cmap=cmap, with_labels=False)
    #     plt.title('Community Structure')
    #     plt.savefig(f'{data_dir}/community_structure_{root}.svg')
    #     plt.show()
    #     plt.close()

    # Save statistics
    statistics = {
        'Number of Nodes': num_nodes,
        'Number of Edges': num_edges,
        'Average Degree': average_degree,
        'Density': density,
        'Connected Components': connected_components,
        'Number of Communities': num_communities,
        # Centrality measures could be added here as well, but they are often better analyzed separately due to their detailed nature
    }
    if include_centrality:
        centrality = {
            'degree_centrality': degree_centrality,
            'betweenness_centrality': betweenness_centrality,
            'closeness_centrality': closeness_centrality,
            'eigenvector_centrality': eigenvector_centrality,
        }
    else:
        centrality=None
 
    
    return statistics, include_centrality

def remove_small_fragents (G_new, size_threshold):
    if size_threshold >0:
        
        # Find all connected components, returned as sets of nodes
        components = list(nx.connected_components(G_new))
        
        # Iterate through components and remove those smaller than the threshold
        for component in components:
            if len(component) < size_threshold:
                # Remove the nodes in small components
                G_new.remove_nodes_from(component)
    return G_new

def colors2Community(communities) -> pd.DataFrame:
    """
    Assigns unique colors to communities for visualization purposes.

    Args:
        communities (list): A list of communities, where each community is a list of nodes.

    Returns:
        pd.DataFrame: A DataFrame mapping nodes to colors and groups.
    """
    p = sns.color_palette(palette, len(communities)).as_hex()
    random.shuffle(p)
    rows = []
    group = 0
    for community in communities:
        color = p.pop()
        group += 1
        for node in community:
            rows += [{"node": node, "color": color, "group": group}]
    df_colors = pd.DataFrame(rows)
    return df_colors

def graph_Louvain (G, 
                  graph_GraphML=None, palette = "hls", verbose=False):
    # Assuming G is your graph and data_dir is defined
    
    # Compute the best partition using the Louvain algorithm
    partition = community_louvain.best_partition(G)
    
    # Organize nodes into communities based on the Louvain partition
    communities = {}
    for node, comm_id in partition.items():
        communities.setdefault(comm_id, []).append(node)
    
    communities_list = list(communities.values())
    if verbose: 
        print("Number of Communities =", len(communities_list))
        print("Communities: ", communities_list)
    
    # Assuming colors2Community can work with the communities_list format
    colors = colors2Community(communities_list)
    if verbose: print("Colors: ", colors)
    
    # Assign attributes to nodes based on their community membership
    for index, row in colors.iterrows():
        node = row['node']
        G.nodes[node]['group'] = row['group']
        G.nodes[node]['color'] = row['color']
        G.nodes[node]['size'] = G.degree[node]
    
    if verbose: print("Done, assigned colors and groups...")
    
    # Write the graph with community information to a GraphML file
    if graph_GraphML != None:
        try:
            nx.write_graphml(G, graph_GraphML)
    
            print("Written GraphML.")

        except:
            print ("Error saving GraphML file.")
    return G

def simplify_graph(graph_, node_embeddings, embd, llm=None, similarity_threshold=0.95, use_llm=False,
                   data_dir_output='./', graph_root='simple_graph', verbatim=False, max_tokens=2048, 
                   temperature=0.3, generate=None):
    """
    Simplifies a graph by merging similar nodes and optionally renaming them using a language model.
    """

    graph = graph_.copy()
    
    nodes = list(node_embeddings.keys())
    embeddings_matrix = np.array([np.array(node_embeddings[node]).flatten() for node in nodes])

    similarity_matrix = cosine_similarity(embeddings_matrix)
    to_merge = np.where(similarity_matrix > similarity_threshold)

    node_mapping = {}
    nodes_to_recalculate = set()
    merged_nodes = set()  # Keep track of nodes that have been merged
    if verbatim:
        print("Start...")
    #for i, j in tqdm(zip(*to_merge), total=len(to_merge[0])) #if tdqm is desired
    for i, j in zip(*to_merge):
        if i != j and nodes[i] not in merged_nodes and nodes[j] not in merged_nodes:  # Check for duplicates
            node_i, node_j = nodes[i], nodes[j]
            
            try:
                if graph.degree(node_i) >= graph.degree(node_j):
                #if graph.degree[node_i] >= graph.degree[node_j]:
                    node_to_keep, node_to_merge = node_i, node_j
                else:
                    node_to_keep, node_to_merge = node_j, node_i
    
                if verbatim:
                    print("Node to keep and merge:", node_to_keep, "<--", node_to_merge)
    
                #if use_llm and node_to_keep in nodes_to_recalculate:
                #    node_to_keep = simplify_node_name_with_llm(node_to_keep, max_tokens=max_tokens, temperature=temperature)
    
                node_mapping[node_to_merge] = node_to_keep
                nodes_to_recalculate.add(node_to_keep)
                merged_nodes.add(node_to_merge)  # Mark the merged node to avoid duplicate handling
            except:
                print (end="")
    if verbatim:
        print ("Now relabel. ")
    # Create the simplified graph by relabeling nodes. removes adds edges from to_merge to to_keep and removes to_merge.
    new_graph = nx.relabel_nodes(graph, node_mapping, copy=True)
    if verbatim:
        print ("New graph generated, nodes relabled. ")
    # Recalculate embeddings for nodes that have been merged or renamed.
    recalculated_embeddings = regenerate_node_embeddings(new_graph, nodes_to_recalculate, embd)
    if verbatim:
        print ("Relcaulated embeddings... ")
    # Update the embeddings dictionary with the recalculated embeddings.
    updated_embeddings = {**node_embeddings, **recalculated_embeddings}

    # Remove embeddings for nodes that no longer exist in the graph.
    for node in merged_nodes:
        updated_embeddings.pop(node, None)
    if verbatim:
        print ("Now save graph... ")

    # Save the simplified graph to a file.
    graph_path = f'{data_dir_output}/{graph_root}_graphML_simplified.graphml'
    nx.write_graphml(new_graph, graph_path)

    if verbatim:
        print(f"Graph simplified and saved to {graph_path}")

    return new_graph, updated_embeddings

#Modified to use HuggingFaceEmbeddings 
def regenerate_node_embeddings(graph, nodes_to_recalculate, embd, verbatim=False):
    """
    Regenerate embeddings for specific nodes using a HuggingFace embeddings model.

    Args:
        graph (nx.Graph): The graph containing the nodes.
        nodes_to_recalculate (list): A list of node names to recalculate embeddings for.
        embd (HuggingFaceBgeEmbeddings): The HuggingFace embeddings model.

    Returns:
        dict: A dictionary mapping node names to their new embeddings.
    """
    new_embeddings = {}
    # for node in tqdm(nodes_to_recalculate): #if tdqm is desired
    iterator = tqdm(nodes_to_recalculate, desc="Recalculating embeddings") if verbatim else nodes_to_recalculate
    for node in iterator:
        embedding = embd.embed_query(node)
        new_embeddings[node] = embedding
    return new_embeddings

#Modified to use HuggingFaceEmbeddings 
def update_node_embeddings(embeddings, graph_new, embd, remove_embeddings_for_nodes_no_longer_in_graph=True,
                           verbatim=False):
    """
    Update embeddings for new nodes in an updated graph, ensuring that the original embeddings are not altered.

    Args:
        embeddings (dict): Existing node embeddings.
        graph_new: The updated graph object.
        embd (HuggingFaceBgeEmbeddings): The HuggingFace embeddings model.
        remove_embeddings_for_nodes_no_longer_in_graph (bool): Whether to remove embeddings for nodes no longer in the graph.
        verbatim (bool): If True, print intermediate information for debugging.

    Returns:
        dict: Updated embeddings dictionary with embeddings for new nodes, without altering the original embeddings.
    """
    # Create a deep copy of the original embeddings
    embeddings_updated = copy.deepcopy(embeddings)

    # Iterate through new graph nodes
    # for node in tqdm(graph_new.nodes()): #if tdqm is desired
    for node in graph_new.nodes():
        # Check if the node already has an embedding in the copied dictionary
        if node not in embeddings_updated:
            if verbatim:
                print(f"Generating embedding for new node: {node}")
            embedding = embd.embed_query(node)
            embeddings_updated[node] = embedding

    if remove_embeddings_for_nodes_no_longer_in_graph:
        # Remove embeddings for nodes that no longer exist in the graph from the copied dictionary
        nodes_in_graph = set(graph_new.nodes())
        for node in list(embeddings_updated):
            if node not in nodes_in_graph:
                if verbatim:
                    print(f"Removing embedding for node no longer in graph: {node}")
                del embeddings_updated[node]

    return embeddings_updated

def simplify_node_name_with_llm(node_name, generate, max_tokens=2048, temperature=0.3):
    # Generate a prompt for the LLM to simplify or describe the node name
    system_prompt='You are an ontological graph maker. You carefully rename nodes in complex networks.'
    prompt = f"Provide a simplified, more descriptive name for a network node named '{node_name}' that reflects its importance or role within a network."
   
    # Assuming 'generate' is a function that calls the LLM with the given prompt
    #simplified_name = generate(system_prompt=system_prompt, prompt)
    simplified_name = generate(system_prompt=system_prompt, prompt=prompt, max_tokens=max_tokens, temperature=temperature)
   
    return simplified_name

def find_best_fitting_node_list(keyword, embeddings, embedding_object, N_samples=5):
    """
    Find the top N_samples nodes with the highest similarity to the keyword.

    Parameters:
    - keyword: str, the input keyword to find similar nodes for.
    - embeddings: dict, a dictionary where keys are nodes and values are their embeddings.
    - embedding_object: HuggingFaceEmbeddings.
    - N_samples: int, number of top similar nodes to return.

    Returns:
    - List of tuples [(node, similarity), ...] in descending order of similarity.
    """

    # Generate embedding for the keyword using the embedding endpoint
    keyword_embedding = embedding_object.embed_query(keyword)
    
    # Initialize a min-heap
    min_heap = []
    heapq.heapify(min_heap)
    
    for node, embedding in embeddings.items():
        embedding = np.array(embedding)  # Ensure embedding is a numpy array
        if embedding.ndim > 1:
            embedding = embedding.flatten()  # Flatten only if not already 1-D
        similarity = 1 - cosine(keyword_embedding, embedding)  # Cosine similarity
        
        # If the heap is smaller than N_samples, just add the current node and similarity
        if len(min_heap) < N_samples:
            heapq.heappush(min_heap, (similarity, node))
        else:
            # If the current similarity is greater than the smallest similarity in the heap
            if similarity > min_heap[0][0]:
                heapq.heappop(min_heap)  # Remove the smallest
                heapq.heappush(min_heap, (similarity, node))  # Add the current node and similarity
                
    # Convert the min-heap to a sorted list in descending order of similarity
    best_nodes = sorted(min_heap, key=lambda x: -x[0])

    assert len(best_nodes) >0 , f"Error in graph_tools.find_best_fitting_node_list(): No nodes found for keyword {keyword}."
    
    # Return a list of tuples (node, similarity)
    return [(node, similarity) for similarity, node in best_nodes]

def save_graph_without_text(G_or, data_dir='./', graph_name='my_graph.graphml'):
    G = deepcopy(G_or)

    # Process nodes: remove 'texts' attribute and convert others to string
    for _, data in tqdm(G.nodes(data=True), desc="Processing nodes"):
        if 'texts' in data:
            del data['texts']  # Remove the 'texts' attribute
        # Convert all other attributes to strings
        for key in data:
            data[key] = str(data[key])

    # Process edges: similar approach, remove 'texts' and convert attributes
    for i, (_, _, data) in enumerate(tqdm(G.edges(data=True), desc="Processing edges")):
    #for _, _, data in tqdm(G.edges(data=True), desc="Processing edges"):
        data['id'] = str(i)  # Assign a unique ID
        if 'texts' in data:
            del data['texts']  # Remove the 'texts' attribute
        # Convert all other attributes to strings
        for key in data:
            data[key] = str(data[key])
    
    # Ensure correct directory path and file name handling
    fname = os.path.join(data_dir, graph_name)
    
    # Save the graph to a GraphML file
    nx.write_graphml(G, fname, edge_id_from_attribute='id')
    return fname

def make_HTML (G,data_dir='./', graph_root='graph_root'):

    net = Network(
            #notebook=False,
            notebook=True,
            # bgcolor="#1a1a1a",
            cdn_resources="remote",
            height="900px",
            width="100%",
            select_menu=True,
            # font_color="#cccccc",
            filter_menu=False,
        )
        
    net.from_nx(G)
    # net.repulsion(node_distance=150, spring_length=400)
    net.force_atlas_2based(central_gravity=0.015, gravity=-31)
    # net.barnes_hut(gravity=-18100, central_gravity=5.05, spring_length=380)
    
    #net.show_buttons(filter_=["physics"])
    net.show_buttons()
    
    #net.show(graph_output_directory, notebook=False)
    graph_HTML= f'{data_dir}/{graph_root}_graphHTML.html'
    
    net.show(graph_HTML, #notebook=True
            )

    return graph_HTML

# Function to generate embeddings
def generate_node_embeddings(graph, embedding_object):
    embeddings = {}
    for node in tqdm(graph.nodes()):
        embedding = embedding_object.embed_query(node)
        embeddings[node] = embedding
    return embeddings
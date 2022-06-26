from tqdm_utils import tqdm
import time
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import networkx_utils as xu
def shortest_path_distance_samples(
    G,
    n_samples = 10000,
    nodes = None,
    edge_weight = None,
    nodes_weight = None,
    plot = True,
    title_prefix = None,#"Minnie65 (E only)",
    log_scale = True,
    verbose = True,
    #ignore_no_path = True,
    ):
    """
    Purpose: To compute the shortest path distance
    between nodes in a graph (may just need to sample the path)
    
    Pseudocode: 
    1) Generate x samples of nodes in the graph as the sources and as the targets
    2) Make sure the two nodes are not the same
    3) Compute the shortest path and add to the list
    4) Compute the mean and standard deviation
    5) Plot a histogram if requested
    """
    
    if not xu.is_graph(G):
        G = nx.Graph(G)
    
    if nodes is None:
        nodes= np.array(G.nodes())
        
    source_names = np.random.choice(nodes,int(n_samples*2),p=nodes_weight)
    target_names = np.random.choice(nodes,int(n_samples*2),p=nodes_weight)
    
    path_lengths = []
    no_paths = 0
    #while len(path_lengths) < n_samples:
    for i in tqdm(range(n_samples)):
#         if verbose:
#             if counter % 1000 == 0:
#                 print(f"Checking Path #{counter}")
            
        s = source_names[i]
        t = target_names[i]
        
        try:
            path = nx.shortest_path(G,s,t,weight = edge_weight)
            path_lengths.append(len(path)-1)
        except:
            no_paths += 1
        
        #counter += 1 
    
    #4) Compute the mean and standard deviation
    if verbose:
        print(f"Path Lengths: Mean = {np.round(np.mean(path_lengths),2)}, Std Dev = {np.round(np.std(path_lengths),2)}, Edge Weight = {edge_weight}, No paths = {no_paths}/{n_samples}")
        
    
    if plot:
        title = f"Shortest Path Distance\n{n_samples} Samples\nEdge Weight = {edge_weight}\nNumber of Samples with No Path = {no_paths}/{n_samples}"
        if title_prefix is not None:
            title = f"{title_prefix}\n{title}"
        
        fig,ax = plt.subplots(1,1)
        ax.hist(path_lengths,bins = 100)
        ax.set_xlabel(f"Shortest Path Distance (Edge Weight = {edge_weight})")
        ax.set_ylabel("Count")
        if log_scale:
            ax.set_yscale('log')
        ax.set_title(title)
        
    return path_lengths
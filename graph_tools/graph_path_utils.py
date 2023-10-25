
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import time
def shortest_path_distance_samples(
    G,
    n_samples = 10000,
    nodes = None,
    source_nodes = None,
    target_nodes = None,
    edge_weight = None,
    nodes_weight = None,
    plot = False,
    title_prefix = None,#"Minnie65 (E only)",
    log_scale = True,
    verbose = True,
    verbose_time = False,
    undirected = False,
    path_nodes = None, #for restricting the nodes to a certain path
    return_no_path_perc = False,
    #ignore_no_path = True,
    **kwargs
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
    if verbose_time:
        tqdmu.turn_on_tqdm()
    st = time.time()
    if not xu.is_graph_any(G):
        G = nx.Graph(G)
    
    if undirected and xu.is_digraph(G):
        G = nx.Graph(G)
        
    
    if verbose_time:
        print(f"Time for graph setup :{time.time() - st}")
        st = time.time()
    
    if nodes is None:
        nodes= np.array(G.nodes())
        
    if verbose_time:
        print(f"Time for node setup :{time.time() - st}")
        st = time.time()
        
    if verbose_time:
        print(f"Time for node and graph setup :{time.time() - st}")
        st = time.time()
        
    if source_nodes is None:
        source_nodes = np.random.choice(nodes,int(n_samples*2),p=nodes_weight,replace=True,)
    else:
        source_nodes = np.random.choice(source_nodes,int(n_samples*2),p=nodes_weight,replace=True,)
        
    if target_nodes is None:
        target_nodes = np.random.choice(nodes,int(n_samples*2),p=nodes_weight,replace=True,)
    else:
        target_nodes = np.random.choice(target_nodes,int(n_samples*2),p=nodes_weight,replace=True,)
        
    if verbose_time:
        print(f"Time for source and target nodes :{time.time() - st}")
        st = time.time()
    
    path_lengths = []
    no_paths = 0
    #while len(path_lengths) < n_samples:
    for i in tqdm(range(n_samples)):
#         if verbose:
#             if counter % 1000 == 0:
#                 print(f"Checking Path #{counter}")
            
        s = source_nodes[i]
        t = target_nodes[i]
        
        
        try:
            if path_nodes is None:
                path = nx.shortest_path(G,s,t,weight = edge_weight)
            else:
                path = xu.shortest_path_along_node_subset(
                    G,
                    start = s,
                    end = t,
                    node_subset = path_nodes,
                    weight = edge_weight,
                    #verbose = False,
                    #verbose_time = False,
                )
            path_lengths.append(len(path)-1)
        except:
            no_paths += 1
        #counter += 1 
    
    if verbose_time:
        print(f"Time for main loop :{time.time() - st}")
        st = time.time()
    
    
    no_path_perc = no_paths/n_samples
    
    #4) Compute the mean and standard deviation
    if verbose:
        print(f"Path Lengths: Mean = {np.round(np.mean(path_lengths),2)}, Std Dev = {np.round(np.std(path_lengths),2)}, Edge Weight = {edge_weight}, No paths = {no_path_perc}")
        
    
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
        
    if return_no_path_perc:
        return path_lengths,no_path_perc
    else:
        return path_lengths

def shortest_path_distance_samples_stat(
    G,
    stat,
    n_samples = 10000,
    nodes = None,
    source_nodes = None,
    target_nodes = None,
    edge_weight = None,
    nodes_weight = None,
    plot = False,
    title_prefix = None,#"Minnie65 (E only)",
    log_scale = True,
    verbose = False,
    return_no_path_perc = False,
    #ignore_no_path = True,
    **kwargs
    ):
    
    path_lengths,no_path_perc = shortest_path_distance_samples(
        G,
        n_samples = n_samples,
        nodes = nodes,
        source_nodes = source_nodes,
        target_nodes = target_nodes,
        edge_weight = edge_weight,
        nodes_weight = nodes_weight,
        plot = plot,
        title_prefix = title_prefix,#"Minnie65 (E only)",
        log_scale = log_scale,
        verbose = verbose,
        return_no_path_perc = True,
        #ignore_no_path = True,
        **kwargs
    )
    
    if type(stat) == str:
        stat = getattr(np,stat)
        
    value = stat(path_lengths)
    
    if return_no_path_perc:
        return value,no_path_perc
    else:
        return value
    
    
def shortest_path_distance_samples_mean(
    G,
    nodes = None,
    source_nodes = None,
    target_nodes = None,
    n_samples = 40_000,
    return_no_path_perc = False,
    **kwargs
    ):
    
    return shortest_path_distance_samples_stat(
        G,
        stat = "mean",
        nodes = nodes,
        source_nodes = source_nodes,
        target_nodes = target_nodes,
        n_samples=n_samples,
        return_no_path_perc = return_no_path_perc,
        **kwargs
    )

    
def shortest_path_distance_samples_mean_undirected(
    G,
    nodes = None,
    source_nodes = None,
    target_nodes = None,
    n_samples = 40_000,
    return_no_path_perc = False,
    **kwargs
    ):
    
    return shortest_path_distance_samples_stat(
        G,
        stat = "mean",
        nodes = nodes,
        source_nodes = source_nodes,
        target_nodes = target_nodes,
        n_samples=n_samples,
        undirected=True,
        return_no_path_perc = return_no_path_perc,
        **kwargs
    )


def shortest_path_distance_samples_mean_from_source(
    G,
    nodes = None,
    n_samples = 40_000,
    return_no_path_perc = False,
    **kwargs
    ):
    
    return shortest_path_distance_samples_stat(
        G,
        stat = "mean",
        nodes = None,
        source_nodes = nodes,
        target_nodes = None,
        n_samples=n_samples,
        return_no_path_perc = return_no_path_perc,
        **kwargs
    )

    
def shortest_path_distance_samples_mean_from_source_undirected(
    G,
    nodes = None,
    n_samples = 40_000,
    return_no_path_perc = False,
    **kwargs
    ):
    
    return shortest_path_distance_samples_stat(
        G,
        stat = "mean",
        nodes = None,
        source_nodes = nodes,
        target_nodes = None,
        n_samples=n_samples,
        undirected = True,
        return_no_path_perc = return_no_path_perc,
        **kwargs
    )


def perc_k(array,k):
    if len(array) == 0:
        return np.nan
    return np.percentile(array,k)
def perc_95(array):
    return perc_k(array,k=95)
def shortest_path_distance_samples_perc_95_from_source(
    G,
    nodes = None,
    n_samples = 40_000,
    return_no_path_perc = False,
    **kwargs
    ):
    
    return shortest_path_distance_samples_stat(
        G,
        stat = perc_95,
        nodes = None,
        source_nodes = nodes,
        target_nodes = None,
        n_samples=n_samples,
        return_no_path_perc = return_no_path_perc,
        **kwargs
    )

    
def shortest_path_distance_samples_perc_95_from_source_undirected(
    G,
    nodes = None,
    n_samples = 40_000,
    return_no_path_perc = False,
    **kwargs
    ):
    
    return shortest_path_distance_samples_stat(
        G,
        stat = perc_95,
        nodes = None,
        source_nodes = nodes,
        target_nodes = None,
        n_samples=n_samples,
        undirected = True,
        return_no_path_perc = return_no_path_perc,
        **kwargs
    )


def example():
    print(f"ex func")



#--- from datasci_tools ---
from datasci_tools import networkx_utils as xu
from datasci_tools import tqdm_utils as tqdmu
from datasci_tools.tqdm_utils import tqdm

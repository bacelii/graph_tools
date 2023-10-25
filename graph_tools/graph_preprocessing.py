
import numpy as np

# def largest_connected_component(
#     G,
#     verbose = False):
#     conn_comp = list(nx.connected_components(G))
#     largest_idx = np.argmax([len(k) for k in conn_comp])
#     if verbose:
#         print(f"Largest connected component size = {len(conn_comp[largest_idx])}")
        
#     return G.subgraph(list(conn_comp[largest_idx]))



def random_subgraph(
    G,
    size = None,
    p = None,
    verbose = False,
    with_replacement = False,
    return_largest_component=False,):
    """
    Purpose: To produce a random sample of graphs
    """
    node_list = np.random.choice(list(G.nodes()),
                     size = size,
                     replace = with_replacement,
                     p = p)
    sub_G = G.subgraph(node_list)
    
    if verbose:
        print(f"After sampling: ")
        xu.print_node_edges_counts(sub_G)
        
    if return_largest_component:
        sub_G = gpre.largest_connected_component(sub_G,verbose = verbose)
        
    return sub_G










#--- from datasci_tools ---
from datasci_tools import networkx_utils as nx
from datasci_tools import networkx_utils as xu

largest_connected_component = xu.largest_connected_component

from . import graph_preprocessing as gpre
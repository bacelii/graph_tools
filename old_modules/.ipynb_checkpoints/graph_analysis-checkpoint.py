import numpy as np
import networkx_utils as xu
import networkx as nx
import graph_visualizations as gviz



import numpy as np
def degree_distribution(G,nodes=None,percentile=None,
                        degree_type="in_and_out",**kwargs):
    """
    Purpose: To find the degree distribution of a graph
    (and optionally apply a node mask or restrict to a percentile of nodes)
    
    """
    if nodes is None:
        nodes = list(G.nodes())
    
    dist = np.array(xu.get_node_degree(G,nodes,degree_type=degree_type))
    
    if percentile is not None:
        dist = dist[dist<np.percentile(dist,percentile)]
        
    return dist

def connected_components(G):
    return list(nx.connected_components(G))
def connected_component_sizes(G):
    """
    Purpose: To return the size of the connected components
    """
    conn_comp = list(nx.connected_components(G))
    conn_comp_size = [len(k) for k in conn_comp]
    return conn_comp_size

def largest_connected_component_size(G):
    """
    Purpose: Returns the size of the largest connected component
    """
    return np.max(connected_component_sizes(G))

def largest_connected_component(G):
    conn_comp = list(nx.connected_components(G))
    largest_idx = np.argmax([len(k) for k in conn_comp])
    return G.subgraph(list(conn_comp[largest_idx]))

def degree_distribution_analysis(G,
                                 graph_title="Graph (No Title)",
                    degree_type_list = ["in_and_out","in","out"],
                    percentile = 99.5,
                    verbose=True,
                    plot_distributions=True):
    """
    Purpose: Will get statistics and possibly 
    plot degree distribution data for a graph


    """


    degree_dist_stats = dict()

    for k in degree_type_list:
        try:
            curr_degree_distr = ga.degree_distribution(G,
                                                       degree_type=k,
                                                        percentile=percentile)
        except:
            if plot_distributions:
                print(f"{graph_title} {k} distribution can't be graphed")
            degree_dist_stats[f"{k}_mean"] = None
            degree_dist_stats[f"{k}_median"] = None
            continue

        curr_degree_distr_mean = np.mean(curr_degree_distr)
        curr_degree_distr_median = np.median(curr_degree_distr)

        if verbose:
            import numpy_utils as nu
            print(f"{graph_title} {k} degree distribution mean = {nu.comma_str(curr_degree_distr_mean)},\n"
                  f"{graph_title} {k} degree distribution median = {nu.comma_str(curr_degree_distr_median)}")

        degree_dist_stats[f"{k}_mean"] = curr_degree_distr_mean
        degree_dist_stats[f"{k}_median"] = curr_degree_distr_median
                                   
        if plot_distributions:
            gviz.plot_degree_distribution(G,
                                         degree_type=k,
                                         title_append=graph_title,
                                          degree_distribution=curr_degree_distr,
                                         print_degree_distribution_stats=False)
    return degree_dist_stats

def graph_analysis_basic(G,
                        graph_title = "Graph (No Name)",
                        verbose = True,
                        plot_distributions = True,
                        degree_distribution_percentile=99.5):

    """
    Purpose: To apply a lot of the graph analysis to a certain graph
    - both stats and visuals

    Pseudocode: 
    1) Print ands save the n_node and n_edges
    For all possible degree distributions
    2) Find statistics and plot degree distribution info
    3) Find the largest component and the stats
    4) for largest compoennt: Find statistics and plot degree distribution info
    5) Assmeble the dictionary to return the statistics
    
    Example: 
    ga.graph_analysis_basic(G=G_orig_multi,
                        graph_title = "Original Graph (Multi-Edge, Undirectional)",
                        verbose = True,
                        plot_distributions = True,
                        degree_distribution_percentile=99.5)
    """
    graph_stats_dict = dict()

    #1) Print ands save the n_node and n_edges
    n_nodes = len(G.nodes())
    n_edges = len(G.edges())
    if verbose:
        import numpy_utils as nu
        print(f"Stats for {graph_title}")
        print(f"Number of Nodes = {nu.comma_str(n_nodes)}")
        print(f"Number of Edges = {nu.comma_str(n_edges)}")

    graph_stats_dict["n_nodes"] = n_nodes
    graph_stats_dict["n_edges"] = n_edges

    #For all degree distributions
    #2) Find the mean and median of the degree distribution
    G_degree_dist_info = ga.degree_distribution_analysis(G,
                            graph_title=graph_title,
                        degree_type_list = ["in_and_out","in","out"],
                        percentile = degree_distribution_percentile,
                        verbose=verbose,
                        plot_distributions=plot_distributions)
    
    graph_stats_dict.update(G_degree_dist_info)

    # Repeating 1 and 2 for the largest component
    return graph_stats_dict



def graph_analysis_basic_with_components(G,
                        graph_title = "Graph (No Name)",
                        verbose = True,
                        plot_distributions = True,
                        degree_distribution_percentile=99.5,
                                        **kwargs):
    """
    Purpose: To run the basic graph analysis for a graph
    and the largest component of that graph
    
    Pseudocode: 
    1) Run graph analysis
    2) Find the number of components and their sizes
    3) Find the largest component
    4) Run graph analysis on largest component
    
    Example: 
    outptu_dict = ga.graph_analysis_basic_with_components(G=G_orig_multi_di,
                        graph_title = "Original Graph (Multi-Edge,Diretional)",
                        verbose = True,
                        plot_distributions = True,
                        degree_distribution_percentile=99.5)
    
    """

    #1) Run graph analysis
    output_dict = dict()
    G_stats = ga.graph_analysis_basic(G=G,
                            graph_title = graph_title,
                            verbose = verbose,
                            plot_distributions = plot_distributions,
                            degree_distribution_percentile=degree_distribution_percentile)
    
    output_dict.update(G_stats)
    
    try:
        output_dict["n_components"] = nx.number_connected_components(G)
    except:
        print("Cant perform connected component analysis")
        output_dict["n_components"] = None
        return output_dict

    if verbose:
        print(f"Working on Largest component")
    G_largest_component = ga.largest_connected_component(G)


    #2) Find the number of components and their sizes
    G_stats_largest_component = ga.graph_analysis_basic(G=G_largest_component,
                            graph_title = graph_title + "\nLargest Component",
                            verbose = verbose,
                            plot_distributions = plot_distributions,
                            degree_distribution_percentile=degree_distribution_percentile)

    new_dict = dict([(f"{k}_largest_comp",v) for k,v in G_stats_largest_component.items()])
    
    output_dict.update(new_dict)
    return output_dict

# ------------ Working with the real graph ---------------
from tqdm_utils import tqdm
import networkx as nx
def direct_conn_df_to_G(df,
                       ):
    """
    Purpose: To convert a graph with direct connetion
    info to a networkx graph
    
    """
    
    edges_df = df[["presyn_nucleus_id","postsyn_nucleus_id","presyn_synapse_id","presyn_skeletal_distance_to_soma","postsyn_skeletal_distance_to_soma"]]
    connectome_edges = edges_df.to_numpy()
    
    G_proof_multi_di = nx.MultiDiGraph()
    #_ = G_proof_multi_di.add_edges_from(connectome_edges[:,:2])
    for u,v,syn_id,presy_dist,postsyn_dist in tqdm(connectome_edges):
        G_proof_multi_di.add_edge(u,v,synapse_id=syn_id,presyn_sk_distance_to_soma=presy_dist,
                                 postsyn_sk_distance_to_soma=postsyn_dist)
    return G_proof_multi_di

def direct_conn_df_to_G_lite(df,
                            presyn_name="presyn",
                            postsyn_name="postsyn"):
    """
    Purpose: To convert a graph with direct connetion
    info to a networkx graph
    
    """
    
    edges_df = df[[presyn_name,postsyn_name]]
    connectome_edges = edges_df.to_numpy()
    
    G_proof_multi_di = nx.MultiDiGraph()
    _ = G_proof_multi_di.add_edges_from(connectome_edges)
    return G_proof_multi_di


def graph_analysis_different_graph_types(G,
                                        title_append=None,
                                        graph_title=None):
    """
    Purpose: Will run the graph analysis for 
    different graph types and return the results

    """


    def return_same(x):
        return x
    graph_func = [return_same,nx.MultiGraph,nx.DiGraph,nx.Graph]
    graph_names = ["multi_di","multi","di","simple"]
    graph_str = ["Multi-Edge,Directional",
                "Multi-Edge,Undirectional",
                 "Unique-Edge,Directional",
                 "Unique-Edge,Undirectional"]
    output_dict = dict()
    for g_func,g_name,g_str in zip(graph_func,graph_names,graph_str):

        print(f"\n\n---- Working on graph type {g_name} ----- \n\n")
        curr_G = g_func(G)

        if graph_title is None:
            graph_title_fixed = f"Graph ({g_str})"
        else:
            graph_title_fixed = f"{graph_title.title()} Graph ({g_str})"
        
        if title_append is not None:
            graph_title_fixed = graph_title_fixed + f"\n{title_append}"

        output_dict[g_name] = ga.graph_analysis_basic_with_components(G=curr_G,
                                                                      graph_title=graph_title_fixed
                                                                     )
    return output_dict

# --------- How to query the graph ----------------

"""
Example:
red_fish = set(n for u,v,d in G.edges_iter(data=True)
               if d['color']=='red'
               for n in (u, v)
               if G.node[n]['label']=='fish')
"""

import graph_analysis as ga
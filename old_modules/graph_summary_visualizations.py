import matplotlib.pyplot as plt
import networkx as nx
import graph_analysis as ga
import numpy as np


import matplotlib_utils as mu
import matplotlib
    
    
def d(G,degree_type="in_and_out",
                             density=False,logscale=True,
                             n_bins=50,
                             bin_width=None,
                             bin_max = None,
                             bin_min = None,
                             title=None,
                             title_append=None,
                             degree_distribution=None,
                             print_degree_distribution_stats=True,
                             fontsize_axes=20,
                             fontsize_title = 20,
                             **kwargs,
                            ):
    """
    Purpose: To plot the degree distribution
    
    """
    if degree_distribution is None:
        degree_distribution_filtered = ga.degree_distribution(G,
                                                       degree_type=degree_type,
                                                       **kwargs)
    else:
        degree_distribution_filtered= degree_distribution
    
    
        
    
    if title is None:
        
        if degree_type == "in":
            title = "In-Degree Distribution"
        elif degree_type == "out":
            title = "Out-Degree Distribution"
        else:
            title = "Degree Distribution"
            
    if title_append is not None:
        title += f"\n{title_append}"
            
    ax = mu.histogram(degree_distribution_filtered,
                   n_bins=n_bins,
                   bin_width=bin_width,
                 bin_max = bin_max,
                 bin_min = bin_min,
                   density=density,
                   logscale=logscale,
                   return_fig_ax=True,
                     fontsize_axes=fontsize_axes)        
    
    ax.set_title(title,fontsize=fontsize_title)
        
    ax.set_xlabel("Degree",fontsize=fontsize_axes)
        
    plt.show()
    
    if print_degree_distribution_stats:
        import numpy_utils as nu
        mean_orig_multi_di = np.mean(degree_distribution_filtered)
        median_orig_multi_di = np.median(degree_distribution_filtered)
        print(f"Mean {degree_type} Degree = {nu.comma_str(np.round(mean_orig_multi_di,2))}\n"
            f"Median {degree_type} Degree = {nu.comma_str(np.round(median_orig_multi_di,2))}")
        
        
import matplotlib.pyplot as plt
def graph_to_soma_to_soma_distance(G):
    """
    Assuming that the nodes of graph have the xyz of the soma centers stored inside of them,
    can compute the soma soma distance of the different connections
    
    Pseudocode:
    
    """
#     total_unique_edges = np.array(list(G.edges()))
#     edges_mask = np.all(total_unique_edges>0,axis=1)
#     soma_distances = np.array([[nuc_id_to_center[k],nuc_id_to_center[v]] for k,v in total_unique_edges[edges_mask]])
#     soma_distances_norm = np.linalg.norm(soma_distances[:,0,:] - soma_distances[:,1,:],axis=1)

    
#     fig,ax = plt.subplots(1,1)
#     ax.hist(soma_distances_norm/1000,bins=100,density=False)
#     ax.set_title("Soma to Soma Distance of Unique Direct Connections \n After Auto Proofreading")
#     ax.set_xlabel("Soma to Soma Distance (um)")
#     ax.set_ylabel("Counts")
#     plt.yscale("log")
#     plt.xscale("linear")
#     plt.show()
    
    pass

import pandas_utils as pu
import numpy_utils as nu
import pandas as pd
def graph_stats_dicts_to_plt_table(stats_list,graph_names_list):
    """
    Psuedocode: To assemble all of the statistics in to a table
    that can be printed oout and put in a paper

    Stats want to note of 
    n_nodes
    n_edge
    unique edges
    Largest connected cluster
    Average Degree
    Average In Degree
    Average Out Degree

    Iterate through all of the graph types
    1) Compute a dictionary of the above mentioned stats

    2) Create a Dataframe from all the dictionaries
    3) Export a Nice looking copy of the DataFrame

    """



    dict_list = []
    for j,s in enumerate(stats_list):
        s_name = graph_names_list[j]
        curr_dict = {
            "Graph Type":s_name,
            "# of nodes":nu.comma_str(s["multi_di"]["n_nodes"]),
            "# of edges":nu.comma_str(s["multi_di"]["n_edges"]),
            "unique edges":nu.comma_str(s["di"]["n_edges"]),
            "# of nodes \nlargest cluster":nu.comma_str(s["multi"]["n_nodes_largest_comp"]),
            "# of edges \nlargest cluster":nu.comma_str(s["multi"]["n_edges_largest_comp"]),
            "mean degree \n(unique edges)": np.round(s["di"]["in_and_out_mean"],2),
            "mean in degree \n(unique edges)": np.round(s["di"]["in_mean"],2),
            "mean out degree \n(unique edges)":np.round(s["di"]["out_mean"],2),
        }
        dict_list.append(curr_dict)

    df = pd.DataFrame.from_dict(dict_list)#.set_index("Graph Type")
    return pu.df_to_render_table(df,transpose=True,col_width=3.2,row_height=0.9,
                                fontsize_header=14,
                                font_size=20)
    
import graph_visualizations as gviz
    
    
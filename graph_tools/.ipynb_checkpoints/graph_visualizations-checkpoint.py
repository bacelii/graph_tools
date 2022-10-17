import matplotlib.pyplot as plt
import networkx as nx
import graph_statistics as gs
import numpy as np


import matplotlib_utils as mu
import matplotlib

def plot_degree_distribution_simple(G,bins = 20):
    degree_distr = gs.degree_distribution(G)
    fig,ax = plt.subplots(1,1)
    ax.hist(degree_distr,bins = bins)
    ax.set_title("Degree Disribution")
    plt.show()

def plot_degree_distribution(G,degree_type="in_and_out",
                             density=False,
                             logscale=False,
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
        degree_distribution_filtered = gs.degree_distribution(G,
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

def draw_G_with_color_array(
    G,
    colors,
    pos = None,
    vmin = None,
    vmax = None,
    cmap = plt.cm.coolwarm,
    ):
    """
    Purpose: To graph the colors associated with an array referencing
    the individual nodes of a graph
    
    ex: 
    eigvals,eigvecs = laplacian_eig_vals_vecs(G)
    draw_G_with_color_array(
        G,
        eigvecs[:,1]
    )
    """
    
    colors = np.array(colors).reshape(-1)
    
    if vmin is None:
        vmin = np.min(colors)
    if vmax is None:
        vmax = np.max(colors)
        
    if pos is None:
        pos = nx.spring_layout(G)
        
    nx.draw_networkx(
        G, 
        pos=pos, 
        node_color=colors,
        cmap=cmap, 
        vmin=vmin, 
        vmax=vmax, 
        with_labels=False)
    
    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm)
    plt.show()
    
    
def plot_partition(
    G,
    values,
    nodes=None,
    pos=None,
    class_colors = ("blue","red",),
    ):
    if pos is None:
        pos = nx.spring_layout(G)
    if nodes is None:
        nodes= list(G.nodes())
    
    values= np.array(values).ravel()
    
    # ----------- doing the scalar plotting ----
    print(f"--- Continuous Classification ---")
    vmin = values.min()
    vmin = -1
    vmax = values.max()
    vmax = 1
    cmap = plt.cm.coolwarm
    print(f"values={[np.round(k,4) for k in values]}")
    nx.draw_networkx(
        G, pos=pos, node_color=values,
                 cmap=cmap, vmin=vmin, vmax=vmax, with_labels=False)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    #sm.set_array([])
    cbar = plt.colorbar(sm)
    plt.show()
    
    print(f"--- Binary Classification ---")
    colors = np.array([class_colors[0]]*len(values)).astype('object')
    colors[values >= 0] = class_colors[1]
    #node_colors = {n:c for n,c in zip(nodes,colors)}
    
    nx.draw(
        G,
        node_color = colors,
        pos = pos,
        with_labels = True,
        
    )
    plt.show()

def plot_modularity_vs_spectral_partitioning(
    G,
    ):

    nodelist = list(G.nodes())
    pos = nx.spring_layout(
        G,
    )

    L = nx.laplacian_matrix(
        G,
        nodelist=nodelist
    ).toarray()
    B = nx.modularity_matrix(
        G,
        nodelist=nodelist
    )

    print(f"--- Modularity Spectral Clustering ---")
    eigvals,eigvecs = np.linalg.eigh(B)
    print(f"eigvals = {eigvals}")
    B_eigvec = eigvecs[:,-1]

    plot_partition(
        G,
        values = B_eigvec,
        nodes = nodelist,
        pos = pos
    )

    print(f"--- Spectral Clustering ---")
    eigvals,eigvecs = np.linalg.eigh(L)
    print(f"eigvals = {eigvals}")
    L_eigvec = eigvecs[:,1]

    plot_partition(
        G,
        values = L_eigvec,
        nodes = nodelist,
        pos = pos
    )
    
import graph_visualizations as gviz
        
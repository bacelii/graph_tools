'''





Note: The motifs it finds, if it finds 3 nodes for a motif,
those 3 nodes are not used in double counting of another motif
(NO DOUBLE COUNTING)





'''
import matplotlib.pyplot as plt
import netsci.metrics.motifs as nsm
import netsci.visualization as nsv
import numpy as np
import seaborn as sns



def plot_all_triads(
    order=nsm.triad_order_nn4576,
    o=(1, 0), delta=(1, 0),
    r=.4,
    node_colors = None,
    **kwargs):
    
    return_list = []
    if node_colors is None:
        node_colors = [None]*len(order)
    
    for i,col in zip(range(len(order)),node_colors):
        return_list.append(
            nsv.plot_a_triad(
                order[i], 
                o=np.array(o) + np.array([i * delta[0], np.mod(i, 2) * delta[1]]),
                r=r, 
                node_color = col,
                **kwargs)
        )
            
    plt.axis('equal')
    
    
def example_plot_3_node_motifs():
    
    figsize = (20,10)
    fig,ax = plt.subplots(1,1,figsize=figsize)

    from datasci_tools import matplotlib_utils as mu
    n_colors = 16
    node_color = mu.generate_non_randon_named_color_list(n_colors,colors_to_omit="black")

    plot_all_triads(
        head_width = 0.12,
        ax=ax,
        node_colors=node_color,
        label = False,
        node_size = 400,
        node_alpha = 0.7
        )

    
def example_plot_3_node_motifs_comparison(
    frequency_syn,
    frequency_prox,
    ):
    
    df = pd.DataFrame()
    motif_names = ["synapse"]*len(frequency_syn) + ["proximity"]*len(frequency_syn)
    df["motif_count"] = list(frequency_syn) + list(frequency_prox)
    df["motif_index"] = np.hstack([
        np.arange(len(frequency_syn)),
        np.arange(len(frequency_syn))
    ])
    df["data"] = motif_names

    df = df.query(f"motif_count >= 0")

    figsize = (20,5)
    fig,ax = plt.subplots(1,1,figsize=figsize)

    ax = sns.barplot(
        data=df,
        x = "motif_index",
        y = "motif_count",
        hue = "data",
        ax = ax,
    )

    ax.set_yscale("log")




#--- from datasci_tools ---
from datasci_tools import matplotlib_utils as mu

from . import netsci_utils as nsu
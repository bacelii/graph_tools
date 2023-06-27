
import matplotlib.pyplot as plt
import numpy as np

def simulate_infection_size(
    A,
    pi0,
    B = 0.1,
    y = 0.1,
    T = 5,
    dt = 0.05,
    ensure_probability_between_0_1 = False,
    plot = True,
    plot_pit = False,
    figsize = (6,6),
    return_pit = False,
    title_suffix = None,
    ):
    r"""Compute the average degree connectivity of graph.

    The average degree connectivity is the average nearest neighbor degree of
    nodes with degree k. For weighted graphs, an analogous measure can
    be computed using the weighted average neighbors degree defined in
    [1]_, for a node `i`, as

    .. math::

        k_{nn,i}^{w} = \frac{1}{s_i} \sum_{j \in N(i)} w_{ij} k_j

    where `s_i` is the weighted degree of node `i`,
    `w_{ij}` is the weight of the edge that links `i` and `j`,
    and `N(i)` are the neighbors of node `i`.

    Parameters
    ----------
    G : NetworkX graph

    source :  "in"|"out"|"in+out" (default:"in+out")
       Directed graphs only. Use "in"- or "out"-degree for source node.

    target : "in"|"out"|"in+out" (default:"in+out"
       Directed graphs only. Use "in"- or "out"-degree for target node.

    nodes : list or iterable (optional)
        Compute neighbor connectivity for these nodes. The default is all
        nodes.

    weight : string or None, optional (default=None)
       The edge attribute that holds the numerical value used as a weight.
       If None, then each edge has weight 1.

    Returns
    -------
    d : dict
       A dictionary keyed by degree k with the value of average connectivity.

    Raises
    ------
    NetworkXError
        If either `source` or `target` are not one of 'in',
        'out', or 'in+out'.
        If either `source` or `target` is passed for an undirected graph.

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> G.edges[1, 2]["weight"] = 3
    >>> nx.average_degree_connectivity(G)
    {1: 2.0, 2: 1.5}
    >>> nx.average_degree_connectivity(G, weight="weight")
    {1: 2.0, 2: 1.75}

    See Also
    --------
    average_neighbor_degree

    References
    ----------
    .. [1] A. Barrat, M. Barthélemy, R. Pastor-Satorras, and A. Vespignani,
       "The architecture of complex weighted networks".
       PNAS 101 (11): 3747–3752 (2004).
    """
    
    
    """
    From homework 2 of networks class
    """

    timesteps = np.arange(dt,T+0.01,dt)

    pt = []

    #initializing the pit
    pit = pi0.copy()
    pt.append(np.sum(pit))
    for i,t in enumerate(timesteps):

        pit = pit + (B * (1 - pit) * (A@pit) - (y*pit))*dt
        if ensure_probability_between_0_1:
            pit[pit > 1] = 1
            pit[pit < 0] = 0

        pt_curr = np.sum(pit)
        pt.append(pt_curr)

    parameters_str = (
        fr"$\beta$ = {B}" + "\n" + 
        fr"$\gamma$ = {y}"
    )
    
    if title_suffix is not None:
        parameters_str += f"\n{title_suffix}"
    if plot:
        fig,ax = plt.subplots(1,1,figsize=figsize)
        ax.plot(np.concatenate([[0],timesteps]),pt)
        ax.set_xlabel("Time")
        ax.set_ylabel(f"Expected size of infection (Total Network Size = {len(A)})")
        ax.set_title(
             "Simulated size of infection\n" + parameters_str
        )    
#         ax.set_title(
#             fr"Simulated size of infection\\n$\beta$ = {B}\\n$\gamma$ = {y}"
#         )
        plt.show()
    
    if plot_pit:
        fig,ax = plt.subplots(1,1,figsize=figsize)
        ax.hist(pit,bins = 50)
        ax.set_title(
            f"Final Pi(t = {T}) histogam\n" + parameters_str
        )
        plt.show()
        
    if return_pit:
        return pt,pit
    return pt






from . import epidemics_utils as epu
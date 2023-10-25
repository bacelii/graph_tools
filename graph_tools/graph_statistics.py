'''





Useful python functions main from networkx
for estimating network stats

Notes: most of functions orignially from /old_modules/graph_statistics_and_simulations.py

packages that need to be installed: 
1) ndlib
2) powerlaw





'''
from itertools import chain
from networkx.algorithms import approximation as app
from networkx.algorithms import distance_measures as dist
from tqdm import tqdm
import ndlib
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep
import networkx as nx
import numpy as np
import powerlaw
import scipy
import time






# ------------- Degree Distributions ------------
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


def degree_distribution_analysis(G,
                                 graph_title="Graph (No Title)",
                    degree_type_list = ["in_and_out","in","out"],
                    percentile = 99.5,
                    verbose=True,
                    plot_distributions=True,
                                 **kwargs
                                ):
    """
    Purpose: Will get statistics and possibly 
    plot degree distribution data for a graph


    """


    degree_dist_stats = dict()

    for k in degree_type_list:
        try:
            curr_degree_distr = gs.degree_distribution(G,
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
            from datasci_tools import numpy_utils as nu
            print(f"{graph_title} {k} degree distribution mean = {nu.comma_str(curr_degree_distr_mean)},\n"
                  f"{graph_title} {k} degree distribution median = {nu.comma_str(curr_degree_distr_median)}")

        degree_dist_stats[f"{k}_mean"] = curr_degree_distr_mean
        degree_dist_stats[f"{k}_median"] = curr_degree_distr_median
                                   
        if plot_distributions:
            gviz.plot_degree_distribution(G,
                                         degree_type=k,
                                         title_append=graph_title,
                                          degree_distribution=curr_degree_distr,
                                         print_degree_distribution_stats=False,
                                          **kwargs
                                         )
    return degree_dist_stats

def degree_distribution_stat(
    G,
    nodes=None,
    stat = "mean",
    percentile=None,
    degree_type="in_and_out",
    verbose = False,
    **kwargs):
    
    if type(stat) == str:
        stat = getattr(np,stat)
        
    # print(f"degree_type = {degree_type}")
    value = stat(degree_distribution(
        G,
        nodes=nodes,
        percentile=percentile,
        degree_type=degree_type,
        **kwargs,
    ))
    
    if verbose:
        print(f"{stat.__name__} of {degree_type} = {value}")
        
    return value

def degree_distribution_mean(
    G,
    nodes=None,
    percentile=None,
    degree_type="in_and_out",
    **kwargs):

    return degree_distribution_stat(
        G,
        nodes=nodes,
        stat = "mean",
        percentile=percentile,
        degree_type=degree_type,
        **kwargs
        )

def degree_distribution_median(
    G,
    nodes=None,
    percentile=None,
    degree_type="in_and_out",
    **kwargs):

    return degree_distribution_stat(
        G,
        nodes=nodes,
        stat = "median",
        percentile=percentile,
        degree_type=degree_type,
        **kwargs
        )

def in_degree_mean(
    G,
    nodes = None,
    percentile = None,
    **kwargs
    ):
    
    return degree_distribution_mean(
        G,
        nodes = nodes,
        percentile = percentile,
        degree_type="in",
        **kwargs
        )

def out_degree_mean(
    G,
    nodes = None,
    percentile = None,
    **kwargs
    ):
    
    return degree_distribution_mean(
        G,
        nodes = nodes,
        percentile = percentile,
        degree_type="out",
        **kwargs
        )

def in_degree_median(
    G,
    nodes = None,
    percentile = None,
    **kwargs
    ):
    
    return degree_distribution_median(
        G,
        nodes = nodes,
        percentile = percentile,
        degree_type="in",
        **kwargs
        )

def out_degree_median(
    G,
    nodes = None,
    percentile = None,
    **kwargs
    ):
    
    return degree_distribution_median(
        G,
        nodes = nodes,
        percentile = percentile,
        degree_type="out",
        **kwargs
        )



#-------------- Functions that are available for graph stats ------------------ #
#adding attributes to functions
class run_options:
    def __init__(self, directional = False,multiedge = False):
        self.directional = directional
        self.multiedge = multiedge

    def __call__(self, f):
        f.directional = self.directional
        f.multiedge = self.multiedge
        return f
    
    
# ------------- basic general stats ----------------

@run_options(directional=False,multiedge=False)
def n_triangles(G):
    triangle_dict = nx.triangles(G)
    n_triangles = np.sum(list(triangle_dict.values()))/3
    return n_triangles

@run_options(directional=False,multiedge=False)
def n_edges_empirical(G):
    return len(G.edges())

@run_options(directional=False,multiedge=False)
def longest_shortest_path(G):
    return dist.diameter(G)


@run_options(directional=False,multiedge=False)
def average_shortest_path_length(G):
    return nx.average_shortest_path_length(G)


diameter = longest_shortest_path

@run_options(directional=True,multiedge=False)
def transitivity(G,**kwargs):
    """
    transitivity: Fraction of all possible traingles present in G
    Triad = 2 edges with a shared vertex

    Transitivity = 3* # of triangles/ # of traids

    """
    return nx.transitivity(G)

@run_options(directional=True,multiedge=True)
def node_connectivity(G,**kwargs):
    return app.node_connectivity(G)

@run_options(directional=False,multiedge=True)
def size_maximum_clique(G,**kwargs):
    """
    clique is just subset of vertices group where every
    vertex in group is connected (subgraph induced is complete)

    Maximum clique = clique of the largest size in a graph
    clique number = number of vertices in a maxium clique

    """
    return nx.graph_clique_number(G)

@run_options(directional=False,multiedge=True)
def n_maximal_cliques(G,**kwargs):
    """
    clique is just subset of vertices group where every
    vertex in group is connected (subgraph induced is complete)

    Maximal clique = clique that cannot be extended by including one or more adjacent vertex 
    (aka not subset of larger clique)
    Maximum clique = clique of the largest size in a graph
    clique number = number of vertices in a maxium clique

    """
    return nx.graph_number_of_cliques(G)

@run_options(directional=True,multiedge=True)
def average_degree_connectivity(G,**kwargs):
    """ Returns dictionary that maps nodes with a certain degree to the average degree of the nearest neightbors"""
    return nx.average_degree_connectivity(G)


@run_options(directional=True,multiedge=False)
def average_clustering(G,**kwargs):
    """ 
    local clustering: theoretically the fraction of traingles that actually exist / 
                                                    all possible traingles in its neighborhood
    How it is computed: 
    1) choose random node
    2) choose 2 neighbors at random
    3) check if traingle (if yes increment traingle counter)
    4) Repeat and compute number with triangle_counter/ trials

    """
    return nx.average_clustering(G)

@run_options(directional=True,multiedge=True)
def min_weighted_vertex_cover_len(G,**kwargs):
    """ 
    Returns length of Minimum number of vertices so that all edges are coincident on at least one vertice

    """
    return len(app.min_weighted_vertex_cover(G))


@run_options(directional=False,multiedge=False)
def tree_number(G,**kwargs):
    """ 
    Returns an approximation of the tree width of the graph (aka how tree-like it is):
    The lower the value the more tree-like the graph is

    """
    return app.treewidth_min_degree(G)[0]

# ------ newly added statistics 3/19 ------------ #

@run_options(directional=True,multiedge=True)
def degree_distribution_mean_simple(G,**kwargs):
    sequences = [k for v,k in dict(G.degree).items()]
    return np.mean(sequences)


@run_options(directional=False,multiedge=False)
def number_connected_components(G,**kwargs):
    return nx.number_connected_components(G)

@run_options(directional=False,multiedge=False)
def largest_connected_component_size(G,**kwargs):
    Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
    return len(Gcc[0])


@run_options(directional=False,multiedge=False)
def largest_connected_component_size(G,**kwargs):
    Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
    return len(Gcc[0])

# ------------- New Functions that were added 3/25 ------------------------ #
@run_options(directional=False,multiedge=False)
def inverse_average_shortest_path(G):
    Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
    sp = nx.average_shortest_path_length(nx.subgraph(G,Gcc[0]))
    if sp > 0:
        return 1/sp
    else:
        return None

    
    
    


# - For the percolation - #
def _get_vertex_order(G,selection_type="random"):
    if selection_type == "random":
        return np.random.permutation(list(G.nodes))
    elif selection_type == "degree":
        """ Will organize from highest to lowest degree"""
        degree_dict = dict()
        for k,v in G.degree():
            if v not in degree_dict.keys():
                degree_dict[v] = [k]
            else:
                degree_dict[v].append(k)
        degree_dict

        #get the order of degree
        order_degrees = np.sort(list(degree_dict.keys()))

        node_order = []
        for k in order_degrees:
            node_order += list(np.random.permutation(degree_dict[k]))

        return node_order
    else:
        raise Exception("Invalid Selection Type")


def run_site_percolation(G,vertex_order_type="random",n_iterations=1000):
    total_runs = []

    for y in tqdm(range(0,n_iterations)):
        current_run_results = [0,1]
        """
        1) Start with empty network. Number of clusters, c = 0, currently in network
        Choose at random the order in which vertices will be added to the network
        """

        clusters=dict() #starting out the clusters list as empyt
        vertex_order = _get_vertex_order(G,vertex_order_type)


        """
        2) Add the next vertex in list to the network initially with no edges
        """
        vertex_labels = dict()
        for i,v in enumerate(vertex_order):
            #print(f"Working on vertex {v}")

            """ 2b)
            - increase the cluster count by 1 (because the new vertex is initially a cluster of its own)
            - Make the cluster size of one

            """

            try:
                max_index_plus_1 = np.max(list(clusters.keys())) + 1
                clusters[max_index_plus_1] = 1
                vertex_labels[v] = max_index_plus_1
            except:
                clusters[0] = 1
                vertex_labels[v] = 0
                continue

            """
            3) Go through the edges attached to newly added vertex and add the edges where the other 
            vertex already exists in the network

            4) For each edge added, check if the vertices have the same cluster group number:
            - if yes then do nothing
            - if no, relabel the smaller cluster the same cluster number as the bigger cluster number
            - update the sizes of the 2 clusters from which formed
            """
            already_added_v = set(vertex_order[:i]).intersection(set(G[v].keys()))
            for a_v in already_added_v:
                if vertex_labels[a_v] != vertex_labels[v]:
                    index_max = np.argmax([clusters[vertex_labels[a_v]],clusters[vertex_labels[v]]])
                    if index_max == 0: #need to change all the labels with v
                        replaced_cluster = vertex_labels[v]
                        indexes_to_change = [jj for jj in vertex_labels.keys() if vertex_labels[jj] == vertex_labels[v]]
                        final_cluster = vertex_labels[a_v]
                    else:
                        replaced_cluster = vertex_labels[a_v]
                        indexes_to_change = [jj for jj in vertex_labels.keys() if vertex_labels[jj] == vertex_labels[a_v]]
                        final_cluster = vertex_labels[v]

                    #change the labels
                    for vv in indexes_to_change:
                        vertex_labels[vv] = final_cluster

                    replaced_size = clusters.pop(replaced_cluster)
                    clusters[final_cluster] += replaced_size

            current_run_results.append(np.max([v for v in clusters.values()]))


            #Done adding that vertex and will continue on to next vertex
            #print(f"clusters = {clusters}")

            total_runs.append(current_run_results)
    total_runs = np.array(total_runs)
    
    from scipy.special import comb
    n = len(G.nodes)
    S_r = np.mean(total_runs,axis=0)
    #calculate s_phi : average largest cluster size as a functin of the occupancy probability
    phi = np.arange(0,1.05,0.05)
    r = np.arange(0,n+1,1)
    s_phi = [np.sum([comb(n, r_curr, exact=True)*(phi_curr**r_curr)*((1-phi_curr)**(n- r_curr))*S_r_curr
                        for r_curr,S_r_curr in zip(r,S_r)]) for phi_curr in phi]
    s_phi = np.array(s_phi)/n
    
    return s_phi,phi
    


@run_options(directional=False,multiedge=False)
def random_degree_site_percolation(G,n_iterations=200):
    random_degree_site_percolation.stat_names = ["area_above_identity_random_percol",
                                                "area_below_identity_random_percol",
                                                "area_above_identity_degree_percol",
                                                "area_below_identity_degree_percol"]
    s_phi_barabasi_rand,phi_barabasi_rand= run_site_percolation(G,"random",n_iterations)
    s_phi_barabasi_degree,phi_barabasi_degree= run_site_percolation(G,"degree",n_iterations)

    rand_diff = s_phi_barabasi_rand - phi_barabasi_rand
    degree_diff = s_phi_barabasi_degree - phi_barabasi_degree

    dx = phi_barabasi_rand[1]-phi_barabasi_rand[0]

    rand_diff_positive = np.where(rand_diff>0)[0]
    rand_diff_negative = np.where(rand_diff<= 0)[0]
    degree_diff_positive = np.where(degree_diff>0)[0]
    degree_diff_negative = np.where(degree_diff<=0)[0]

    return (np.trapz(rand_diff[rand_diff_positive],dx=dx),
     np.trapz(rand_diff[rand_diff_negative],dx=dx),
     np.trapz(degree_diff[degree_diff_positive],dx=dx),
     np.trapz(degree_diff[degree_diff_negative],dx=dx))

# - End of Percolation - #

# - Start of Beta Epidemic Stat -- #
    


def pandemic_beta_average(
                                graph,
                                average_iterations=5,
                                n_time_iterations = 200,
                                initial_infected_prop = 0.05,
                                gamma = 0.01,
                                beta_start = 0.00001,
                                current_jump=0.001,
                                pandemic_threshold = 0.7,
                                pandemic_dev = 0.01,
                                max_iterations = 50,
                                use_optimized_beta_finder=False
                              ):
    arg_dict = dict(
        n_time_iterations = n_time_iterations,
        initial_infected_prop = initial_infected_prop,
        gamma = gamma,
        beta_start = beta_start,
        current_jump=current_jump,
        pandemic_threshold = pandemic_threshold,
        pandemic_dev = pandemic_dev,
        max_iterations=max_iterations
            )
    
    pandemic_beta = []
    for i in range(0,average_iterations):
        print(f"\n    Working on Run {i}")
        percent_affected_history = []
        retry_counter = 0
        max_retry = 3
        while len(percent_affected_history)<= 1:
            percent_affected_history,beta_history = find_pandemic_beta(graph,**arg_dict)
            retry_counter += 1
            if retry_counter > max_retry:
                print(f"Could not find right Beta after {max_retry} tries: returning Last value hit: {beta_history[0]}")
                return beta_history[0]
                #raise Exception(f"Could not find right Beta after {max_retry} tries")
        optimal_beta = beta_history[-1]
        
        pandemic_beta.append(optimal_beta)
        """
        now adjust the beta_start and the current jump based on the history 
        before the next run
        
        Rule: Find the optimal beta
        starting_beta = optimal_beta - 3*last_jump
        
        
        """ 
        if use_optimized_beta_finder:
            if len(beta_history) >= 2:
                print(f"beta_history= {beta_history}")
                last_jump_size = np.abs(beta_history[-2] - beta_history[-1])
                arg_dict["beta_start"] = optimal_beta - 2*last_jump_size
                arg_dict["current_jump"] = 2*last_jump_size
            
        
    return np.mean(pandemic_beta)

def find_pandemic_beta(graph,
                       n_time_iterations = 200,
                        initial_infected_prop = 0.05,
                        gamma = 0.01,
                        beta_start = 0.00001,
                        current_jump=0.001,
                        pandemic_threshold = 0.7,
                        pandemic_dev = 0.01,
                       max_iterations=50
                       ):
    
    print_flag = False
    
    def _calculate_percent_affected(trends):
        n_recovered = trends[0]["trends"]["node_count"][2][-1]
        n_infected = trends[0]["trends"]["node_count"][1][-1]
        percent_affected = (n_recovered + n_infected)/len(graph.nodes)
        return percent_affected

    print(f"\n\n---- New Run: Finding Beta for [{pandemic_threshold - pandemic_dev}, {pandemic_threshold + pandemic_dev}]\n"
         f"    Starting with beta_start={beta_start},current_jump={current_jump}")
    percent_affected = 0
    counter = 0
    beta = beta_start
    
    beta_history=[]
    percent_affected_history = []
    
    
    while (percent_affected > pandemic_threshold + pandemic_dev
           or percent_affected < pandemic_threshold - pandemic_dev):
        if print_flag:
            print(f"Current loop {counter}")
        counter += 1
        if counter > max_iterations:
            print("Max iterations hit before convergence on Beta, going to try again")
            return [],[beta]

        model = ep.SIRModel(graph)
        #Setting the model configuration
        config = mc.Configuration()
        config.add_model_parameter('beta', beta)
        config.add_model_parameter('gamma', gamma)
        config.add_model_parameter("fraction_infected", initial_infected_prop) #not setting the initial nodes that are infected but just the initial fraction
        model.set_initial_status(config)

        # Simulation
        iterations = model.iteration_bunch(n_time_iterations) 
        trends = model.build_trends(iterations) # builds the  dict_keys(['node_count', 'status_delta']) time series
        percent_affected = _calculate_percent_affected(trends)

        beta_history.append(beta)
        percent_affected_history.append(percent_affected)
        
        if print_flag:
            print(f"With beta = {beta}, percent_affected = {percent_affected}, current_jump={current_jump}")
        #Adjust the Beta
        if percent_affected < pandemic_threshold - pandemic_dev:
            beta += np.min((1,current_jump))
        elif percent_affected > pandemic_threshold + pandemic_dev:
            #print("beta_history[-2] = " + str(beta_history[-2]))
            if percent_affected_history[-2] < pandemic_threshold - pandemic_dev: #if jumped over the answer
                if print_flag:
                    print("Jumped over answer")
                beta = np.max((beta - current_jump/2,0))
                current_jump = current_jump/2 - current_jump*0.01
                if current_jump < 0.000001:
                    current_jump = 0.000001
            else:
                beta = np.max((beta - current_jump,0))
            
        else:
            break
    return percent_affected_history,beta_history


@run_options(directional=False,multiedge=False)
def pandemic_beta(graph):
    print("\n\n-------Starting Pandemic Beta -------")
    print(f"n_edges = {len(graph.edges())}, n_nodes = {len(graph.nodes)}")
    start_time = time.time()
    final_beta_average = pandemic_beta_average(graph)
    print(f"final_beta_average = {final_beta_average}")
    print(f"Total time for optimized = {time.time() - start_time}\n")
    
    return final_beta_average



# - End of Beta Epidemic Stat -- #


#eigenvalue measurements
@run_options(directional=False,multiedge=False)
def largest_adj_eigen_value(G1):
    Adj = nx.convert_matrix.to_numpy_matrix(G1)
    return np.real(np.max(np.linalg.eigvals(Adj)))

@run_options(directional=False,multiedge=False)
def smallest_adj_eigen_value(G1):
    Adj = nx.convert_matrix.to_numpy_matrix(G1)
    return np.real(np.min(np.linalg.eigvals(Adj)))

@run_options(directional=False,multiedge=False)
def largest_laplacian_eigen_value(G1):
    laplacian = scipy.sparse.csr_matrix.toarray(nx.laplacian_matrix(G1))
    return np.real(np.max(np.linalg.eigvals(laplacian)))

"""
SHOULDN'T REALLY HAVE TO USE THIS BECAUSE IT WILL ALWAYS BE 0
"""
@run_options(directional=False,multiedge=False)
def smallest_laplacian_eigen_value(G1):
    laplacian = scipy.sparse.csr_matrix.toarray(nx.laplacian_matrix(G1))
    return np.real(np.min(np.linalg.eigvals(laplacian)))

#*** THIS IS ALSO CALLED THE ALGEBRAIC CONNECTIVITY
@run_options(directional=False,multiedge=False)
def second_smallest_laplacian_eigen_value(G1):
    laplacian = scipy.sparse.csr_matrix.toarray(nx.laplacian_matrix(G1))
    sorted_eig_vals = np.sort(np.real(np.linalg.eigvals(laplacian)))
    return sorted_eig_vals[1]

@run_options(directional=False,multiedge=False)
def top_heavy_percentage(G1,top_percentage = 0.90):
    degree_sequence = np.array(G1.degree())[:,1]
    ordered_nodes = np.argsort(degree_sequence)


    index_to_start = np.ceil(len(degree_sequence)*top_percentage).astype("int")
    #print(f"index_to_start = {index_to_start}")
    top_nodes_to_keep = ordered_nodes[index_to_start:]
    #print(f"top_nodes_to_keep = {top_nodes_to_keep}")

    nodes_nbrs = G1.adj.items()
    top_neighbors = [set(v_nbrs) for v,v_nbrs in nodes_nbrs if v in top_nodes_to_keep]
    top_neighbors.append(set(top_nodes_to_keep))

    unique_top_neighbors = set(chain.from_iterable(top_neighbors))
    return len(unique_top_neighbors)/len(G1)

@run_options(directional=False,multiedge=False)
def critical_occupation_probability(G1):
    degree_sequence = np.array(G1.degree())[:,1]
    return np.mean(degree_sequence)/(np.mean(degree_sequence)**2 - np.mean(degree_sequence))


@run_options(directional=False,multiedge=False)
def rich_club_transitivity(G):
    """
    Computes the triad closure percentage between only those nodes with same or higher degree
    """
    nodes_nbrs = G.adj.items()

    triads = 0
    triangles = 0
    degree_lookup = dict(G.degree())

    for v,v_nbrs in nodes_nbrs:
        v_nbrs_degree = [vnb for vnb in v_nbrs if degree_lookup[vnb] >= degree_lookup[v]]
        vs=set(v_nbrs_degree)-set([v]) #getting all the neighbors of the node (so when put in different combinations these could be triads)
        local_triangles=0
        local_triads = len(vs)*(len(vs) - 1)
        if local_triads<1:
            #print("No local triads so skipping")
            continue
        for w in vs:
            ws = set(G[w])-set([w]) #gets all the neighbors of a neighbor except itself
            local_triangles += len(vs.intersection(ws)) #finds out how many common neighbors has between itself and main node

        #print(f"For neuron {v}: Triads = {local_triads/2}, Triangles = {local_triangles/2}, transitivity = {local_triangles/local_triads}")
        triads += local_triads 
        triangles+= local_triangles
    
    #print(f"Total: Triads = {triads/2}, Triangles = {triangles/2}, transitivity = {triangles/triads}")
    if triads > 0:
        return triangles/triads
    else:
        return None


# --- Powerlaw stats --- #


def get_degree_distribution(G):
    return np.array(G.degree())[:,1]

@run_options(directional=False,multiedge=False)
def power_law_alpha_sigma(G):
    #get the degree distribution
    power_law_alpha_sigma.stat_names = ["power_law_alpha",
                                        "power_law_sigma"]
    fit = powerlaw.Fit(get_degree_distribution(G))
    return fit.power_law.alpha, fit.power_law.sigma

@run_options(directional=False,multiedge=False)
def power_law_sigma(G):
    #get the degree distribution

    fit = powerlaw.Fit(get_degree_distribution(G))
    return fit.power_law.sigma

@run_options(directional=False,multiedge=False)
def power_law_alpha(G):
    #get the degree distribution

    fit = powerlaw.Fit(get_degree_distribution(G))
    return fit.power_law.alpha

@run_options(directional=False,multiedge=False)
def power_exp_fit_ratio(G):
    """
    Will return the loglikelihood ratio of the power and exponential graph
    R:
    Will be positive if power is more likely
            negative    exponential
    
    p: significance of fit
    """
    #get the degree distribution
    power_exp_fit_ratio.stat_names = ["power_exp_LL_ratio",
                                        "power_exp_LL_ratio_sign"]
    
    fit = powerlaw.Fit(get_degree_distribution(G))
    R,p = fit.distribution_compare("power_law",
                                                 "exponential",
                                                normalized_ratio=True)
    return R

@run_options(directional=False,multiedge=False)
def trunc_power_stretched_exp_fit_ratio(G):
    """
    Will return the loglikelihood ratio of the power and exponential graph
    R:
    Will be positive if power is more likely
            negative    exponential
    
    p: significance of fit
    """
    #get the degree distribution
    trunc_power_stretched_exp_fit_ratio.stat_names = ["trunc_power_stretched_exp_LL_ratio",
                                        "trunc_power_stretched_exp_LL_ratio_sign"]
    
    fit = powerlaw.Fit(get_degree_distribution(G))
    R,p = fit.distribution_compare("truncated_power_law",
                                                 "stretched_exponential",
                                                normalized_ratio=True)
    return R





def eig_vals_vecs_from_matrix(
    array,
    verbose = False):
    """
    Eigenvectors are in the columns and 
    organized in ascending order (so last one is the largest)
    """
    st = time.time()
    eigvals, eigvecs = np.linalg.eigh(array)
    if verbose:
        print(f"Time for eigenvalue, vector  = {time.time() - st}")
        
    return eigvals, eigvecs
    

def laplacian_eig_vals_vecs(G,verbose = False):
    if xu.is_graph(G):
        lap = gs.laplacian(G)
    else:
        lap = G
    return gs.eig_vals_vecs_from_matrix(lap,verbose = verbose)

def adjacency_eig_vals_vecs(G,verbose = False):
    return gs.eig_vals_vecs_from_matrix(gs.adjacency(G),verbose = verbose)

def eigenvector_centrality(
    G,
    weight = None,
    **kwargs):
    return nx.eigenvector_centrality(
        G,
        weight=weight,
        **kwargs)

def betweenness_centrality(
    G,
    normalized=True,
    endpoints = False,
    **kwargs):
    return nx.algorithms.betweenness_centrality(
        G,
        normalized=normalized,
        endpoints = endpoints,
        **kwargs,
    )

def degree_centrality(
    G,
    weight = None,
    nodes = None,
    normalize = True,
    ):
    """
    Purpose: To compute the degree
    centrality for nodes (with an option for weighted)
    """
    if nodes is None:
        nodes = list(G.nodes())
        
    weight_dict = dict(G.degree(
        nodes,
        weight = weight
    ))
    
    if normalize:
        total_weight = np.linalg.norm(list(weight_dict.values()))
        weight_dict = {k:v/total_weight for k,v in weight_dict.items()}
        
    return weight_dict

def eigenvector_centrality_numpy(G, weight=None, max_iter=50, tol=0):
    r"""Compute the eigenvector centrality for the graph G.

    Eigenvector centrality computes the centrality for a node based on the
    centrality of its neighbors. The eigenvector centrality for node $i$ is

    .. math::

        Ax = \lambda x

    where $A$ is the adjacency matrix of the graph G with eigenvalue $\lambda$.
    By virtue of the Perron–Frobenius theorem, there is a unique and positive
    solution if $\lambda$ is the largest eigenvalue associated with the
    eigenvector of the adjacency matrix $A$ ([2]_).

    Parameters
    ----------
    G : graph
      A networkx graph

    weight : None or string, optional (default=None)
      The name of the edge attribute used as weight.
      If None, all edge weights are considered equal.
      In this measure the weight is interpreted as the connection strength.
    max_iter : integer, optional (default=100)
      Maximum number of iterations in power method.

    tol : float, optional (default=1.0e-6)
       Relative accuracy for eigenvalues (stopping criterion).
       The default value of 0 implies machine precision.

    Returns
    -------
    nodes : dictionary
       Dictionary of nodes with eigenvector centrality as the value.

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> centrality = nx.eigenvector_centrality_numpy(G)
    >>> print([f"{node} {centrality[node]:0.2f}" for node in centrality])
    ['0 0.37', '1 0.60', '2 0.60', '3 0.37']

    See Also
    --------
    eigenvector_centrality
    pagerank
    hits

    Notes
    -----
    The measure was introduced by [1]_.

    This algorithm uses the SciPy sparse eigenvalue solver (ARPACK) to
    find the largest eigenvalue/eigenvector pair.

    For directed graphs this is "left" eigenvector centrality which corresponds
    to the in-edges in the graph. For out-edges eigenvector centrality
    first reverse the graph with ``G.reverse()``.

    Raises
    ------
    NetworkXPointlessConcept
        If the graph ``G`` is the null graph.

    References
    ----------
    .. [1] Phillip Bonacich:
       Power and Centrality: A Family of Measures.
       American Journal of Sociology 92(5):1170–1182, 1986
       http://www.leonidzhukov.net/hse/2014/socialnetworks/papers/Bonacich-Centrality.pdf
    .. [2] Mark E. J. Newman:
       Networks: An Introduction.
       Oxford University Press, USA, 2010, pp. 169.
    """
    import numpy as np
    import scipy as sp
    import scipy.sparse.linalg  # call as sp.sparse.linalg

    if len(G) == 0:
        raise nx.NetworkXPointlessConcept(
            "cannot compute centrality for the null graph"
        )
    M = nx.to_scipy_sparse_array(G, nodelist=list(G), weight=weight, dtype=float)
    eigenvalue, eigenvector = sp.sparse.linalg.eigs(
        M.T, k=1, which="LR", maxiter=max_iter, tol=tol
    )
    largest = eigenvector.flatten().real
    norm = np.sign(largest.sum()) * sp.linalg.norm(largest)
    return dict(zip(G, largest / norm))

# ---- new graph statistics -----
def local_clustering_coefficients(
    G,
    nodes=None,
    **kwargs):
    return nx.clustering(G,nodes=nodes,**kwargs)

def modularity_matrix(G,nodelist=None,**kwargs):
    return nx.modularity_matrix(
        G,
        nodelist=nodelist,
        **kwargs
    )
    
    
# --- 11/28 ---
def node_attribute_stat(
    G,
    attribute,
    stat = "mean",
    nodes = None,
    verbose = False   
    ):
    """
    Purpose: To get a summary
    statistic of a node
    attribute from all nodes
    in graph

    Pseudocode: 
    1) Get the attribute for all nodes
    specified
    2) Run the summary statistic
    
    Ex: 
    gstat.node_attribute_stat(
        G = G_auto,
        attribute = "axon_skeletal_length",
        stat = "mean",
        verbose = True,
    )
    """
    if type(stat) == str:
        stat = getattr(np,stat)

    attrs = xu.get_node_attribute_for_all_nodes(
        G,
        name = attribute,
        nodes = nodes,
    )
    
    value = stat(attrs)
    if verbose:
        print(f"{stat.__name__} of {attribute} = {value}")

    
    return value

def node_attribute_mean(
    G,
    attribute,
    nodes = None,
    verbose = False  
    ):
    
    return node_attribute_stat(
    G,
    attribute,
    stat = "mean",
    nodes = nodes,
    verbose = verbose   
    )

def node_attribute_median(
    G,
    attribute,
    nodes = None,
    verbose = False  
    ):
    
    return node_attribute_stat(
    G,
    attribute,
    stat = "median",
    nodes = nodes,
    verbose = verbose   
    )
    
def degree_sequences(
    A,
    return_non_zero_degree_mask = True,
    filter_away_disconnected_nodes = False,
    ):
    out_degree_sequence = xu.out_degree_sequence_from_adj(A)
    in_degree_sequence = xu.in_degree_sequence_from_adj(A)
    
    if filter_away_disconnected_nodes:
        good_map = (out_degree_sequence > 0) | (in_degree_sequence > 0)
        in_degree_sequence = in_degree_sequence[good_map]
        out_degree_sequence = out_degree_sequence[good_map]
    else:
        good_map = np.arange(0,len(A))
    
    return_value = [in_degree_sequence,out_degree_sequence]
    
    if return_non_zero_degree_mask:
        return_value.append(good_map) 
        
    return return_value


def n_reciprocal(A):
    return int((A+A.T == 2).sum()/2)

def reciprocity(A):
    n = A.shape[0]
    return (A+A.T == 2).sum() / float(n*(n-1))

def sparsity(A,verbose = False):
    n = A.shape[0]
    return A.sum() / float(n*(n-1))

    
def n_edges_from_A(A):
    return A.sum()

#--- from graph_tools ---
from . import graph_visualizations as gviz

#--- from datasci_tools ---
from datasci_tools import networkx_utils as xu

adjacency_matrix = xu.adjacency_matrix
modularity_matrix = xu.modularity_matrix
laplacian = xu.laplacian

from . import graph_statistics as gs
import matplotlib.pyplot as plt
def graph_to_soma_to_soma_distance(G,
                                  nuc_id_to_center):
    total_unique_edges = np.array(list(G.edges()))
    edges_mask = np.all(total_unique_edges>0,axis=1)
    soma_distances = np.array([[nuc_id_to_center[k],nuc_id_to_center[v]] for k,v in total_unique_edges[edges_mask]])
    soma_distances_norm = np.linalg.norm(soma_distances[:,0,:] - soma_distances[:,1,:],axis=1)

    
    fig,ax = plt.subplots(1,1)
    ax.hist(soma_distances_norm/1000,bins=100,density=False)
    ax.set_title("Soma to Soma Distance of Unique Direct Connections \n After Auto Proofreading")
    ax.set_xlabel("Soma to Soma Distance (um)")
    ax.set_ylabel("Counts")
    plt.yscale("log")
    plt.xscale("linear")
    plt.show()

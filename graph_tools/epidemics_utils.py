
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
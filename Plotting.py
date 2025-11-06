import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np
from HydroCore import PrimativeIndex
import HydroCore

def plot_results(
    input_pkl_file: str = "snapshot.pkl",
    filename: str = "sod_shock_evolution.png",
    title: str = "Sod Shock Tube Evolution (HLL Flux)",
    xlabel: str = "x",
    show_mach: bool = False
):
# GPT generated w/ edits b/c plotting is a pain...
# Prompt was that after debugging the above, it offered to plot the results 
# I said yes, but use the history list, save it to a png filename
# I also cleaned up extraneous variables and the like
    with open(input_pkl_file, 'rb') as f:
        history, params = pkl.load(f)
    N = history[0][1].shape[0]
    support = params.grid_info.construct_grid_centers(0)
    assert(N==support.shape[0])

    if(show_mach):
        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    else:
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    t, U = history[-1]
    W = HydroCore.conservative_to_primitive(U, params)

    rho = W[:, PrimativeIndex.DENSITY.value]
    v = W[:, PrimativeIndex.VELOCITY.value]
    P = W[:, PrimativeIndex.PRESSURE.value]
    c_s = HydroCore.sound_speed(W, params)

    label = f"t = {t:.3f}"

    axes[0].plot(support, rho,label=label)
    axes[1].plot(support, v)
    axes[2].plot(support, P)
    if(show_mach):
        axes[3].plot(support, np.abs(v)/c_s )

    axes[0].set_ylabel(r"$\rho$")
    axes[1].set_ylabel(r"$v$")
    axes[2].set_ylabel(r"$P$")
    if(show_mach):
        axes[3].set_ylabel(r"$M$")
        axes[3].axhline(y=1, color='black', linestyle='--', linewidth=1)
    for i, idx in enumerate(axes):
        axes[i].set_xlabel(xlabel)

    axes[0].legend(loc="best", frameon=False)
    fig.suptitle(title, fontsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(filename, dpi=200)


def plot_Mdot_time(
    input_pkl_file: str
):
    with open(input_pkl_file, 'rb') as f:
        history, params = pkl.load(f)
    centers=  params.grid_info.construct_grid_centers(0)
    t = [] 
    data = []
    for time, profile in history:
        M_dot_val = HydroCore.M_dot(profile, params)[0]
        t.append(time)
        data.append(np.abs(M_dot_val))
    plt.scatter(t, data) 
    plt.xlabel("Time")
    plt.ylabel("$M_{dot}$")
    plt.show()

def plot_Mdot_position(
    input_pkl_file: str = "snapshot.pkl"
):
    with open(input_pkl_file, 'rb') as f:
        history, params = pkl.load(f)
    centers=  params.grid_info.construct_grid_centers(0)
    plt.scatter(centers, HydroCore.M_dot(history[-1][1], params) )
    plt.xlabel("R")
    plt.ylabel("$M_{dot}$")
    plt.show()



if __name__ =="__main__":
    pass

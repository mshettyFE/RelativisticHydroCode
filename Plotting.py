import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np
from matplotlib.animation import FuncAnimation
from HydroCore import PrimitiveIndex, SimulationState
from GridInfo import WeightType
from metrics import Metric
from GuessPrimitives import sound_speed
from HelperFunctions import index_primitive_var, VariableSet
from matplotlib import colors

def plot_results_1D(
    input_pkl_file: str = "snapshot.pkl",
    filename: str = "sod_shock_evolution.png",
    title: str = "Sod Shock Tube Evolution (HLL Flux)",
    xlabel: str = "x",
    show_mach: bool = False,
    which_slice: int = -1
):
# GPT generated w/ edits b/c plotting is a pain...
# Prompt was that after debugging the above, it offered to plot the results 
# I said yes, but use the history list, save it to a png filename
# I also cleaned up extraneous variables and the like
    with open(input_pkl_file, 'rb') as f:
        history, sim_state = pkl.load(f)
    sim_state.metric.sanity_check(sim_state.grid_info, sim_state.simulation_params)
    N = history[0][1].shape[0]
    support = sim_state.grid_info.construct_grid_centers(0)
    assert(N==support.shape[0])

    if(show_mach):
        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    else:
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    t, U = history[which_slice]
    sim_state.U = U
    U_cart = sim_state.metric.unweight_system(U, sim_state.grid_info, WeightType.CENTER, sim_state.simulation_params)
    U_cart_padded = sim_state.pad_array(U_cart, sim_state.U_initial_unweighted_padded, VariableSet.CONSERVATIVE)
    W = sim_state.conservative_to_primitive(U_cart_padded)

    rho = index_primitive_var( W,PrimitiveIndex.DENSITY,sim_state.n_variable_dimensions).flatten()
    v = index_primitive_var( W,PrimitiveIndex.X_VELOCITY,sim_state.n_variable_dimensions).flatten()
    P = index_primitive_var( W,PrimitiveIndex.PRESSURE,sim_state.n_variable_dimensions).flatten()
    c_s = sound_speed(sim_state.simulation_params, P, rho, sim_state.simulation_params.regime).flatten()

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
        saved_state = pkl.load(f)
    history = saved_state[0] 
    sim_state: SimulationState = saved_state[1]
    t = [] 
    data = []
    for time, profile in history:
        sim_state.U = profile
        M_dot_val = sim_state.M_dot()[0]
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
        history, sim_state= pkl.load(f)
    centers=  sim_state.grid_info.construct_grid_centers(0)
    sim_state.U = history[-1][1]
    plt.scatter(centers, sim_state.M_dot() )
    plt.xlabel("R")
    plt.ylabel("$M_{dot}$")
    plt.show()

def plot_2D_anim(
    input_pkl_file: str = "snapshot.pkl"   
        ):
    with open(input_pkl_file, 'rb') as f:
        data, last_state = pkl.load(f)
 
    fig, ax = plt.subplots()

    # Initialize image using the first array
    t0, Z0 = data[0]
    plot_var = Z0[...,0]
    vmin = plot_var.min()
    vmax = plot_var.max()
    grid_centers_x = last_state.grid_info.construct_grid_centers(0)
    grid_centers_y = last_state.grid_info.construct_grid_centers(1)
    xx,yy  = np.meshgrid(grid_centers_x, grid_centers_y)
    quad = ax.pcolormesh(xx, yy, plot_var.T, shading='auto', cmap='viridis', norm=colors.LogNorm(vmin=vmin, vmax=vmax))
    fig.colorbar(quad, ax=ax)
    ax.set_title(f"t = {data[0][0]:.3f}")

    def update(frame):
        t, arr = data[frame]
        quad.set_array(arr[...,0].ravel())
        ax.title.set_text(f"t = {t:.3f}")  
        return quad, ax.title

    frame_indices = list(range(0, len(data), 100))
    ani = FuncAnimation(fig, update, frames=frame_indices, interval=100, blit=False)

    plt.show()

if __name__ =="__main__":
    pass

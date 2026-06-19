# scripts/plotting_cifs_pole_decomposition.py

from source.utils import file_walker
from source.utils.plotting import (
    set_default_plot_style,
    set_default_axis_style,
    boldtext
)
from source.utils.config import load_config
from source.utils import file_walker, results_layout
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # default logging level
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
import os
from pathlib import Path

# directory you ran the command from
run_dir = Path.cwd()
os.chdir(run_dir)

def generate_state_single_plot(results_to_plot,cfg,results_layout):

    set_default_plot_style(cfg)
    vib_freq = cfg["vib_freq"]
    time_vec = results_to_plot["time_vec"]*vib_freq
    physical_state = results_to_plot["physical_state"][:,0,:]
    plot_cfg = cfg["plotting"]
    fig, ax = plt.subplots()
    ax.plot(time_vec,physical_state[:,0],linewidth=plot_cfg["linewidth"],color='b')
    ax.plot(time_vec,physical_state[:,1],linewidth=plot_cfg["linewidth"],color='r')
    ax.set_ylabel(
                     fr'$\displaystyle  \langle x \rangle, \langle p \rangle $',
                     fontsize=plot_cfg["label_fontsize"]
                     )
    ax.set_xlabel(
                     fr'$\displaystyle {boldtext("Time")}$',
                     fontsize=plot_cfg["label_fontsize"]
                     )
    set_default_axis_style(cfg,ax)
    ax.set_xlim(0,np.max(time_vec))
    plt.tight_layout()
    plt.savefig(results_layout.transient_plots / f"physical_state_transient.pdf")
    # plt.show()

def generate_energy_single_plot(results_to_plot,cfg,results_layout):

    set_default_plot_style(cfg)
    vib_freq = cfg["vib_freq"]
    temp = cfg["temperature"] * 8.617333262e-5
    time_vec = results_to_plot["time_vec"]*vib_freq
    kinetic_energy = results_to_plot["kinetic_energy"][:,0]
    potential_energy = results_to_plot["potential_energy"][:,0]
    plot_cfg = cfg["plotting"]
    fig, ax = plt.subplots()
    ax.plot(time_vec,kinetic_energy,linewidth=plot_cfg["linewidth"],color='b')
    ax.plot(time_vec,potential_energy,linewidth=plot_cfg["linewidth"],color='r')
    # ax.axhline(y=0.60495, color='g', linestyle='-')
    # ax.axhline(y=0.4425, color='g', linestyle='--')
    # ax.axhline(y=0.02255694990, color='g', linestyle='-')
    ax.axhline(y=temp / 2, color='g', linestyle='-')
    ax.set_ylabel(
                     fr'$\displaystyle  {boldtext("Energy")} $',
                     fontsize=plot_cfg["label_fontsize"]
                     )
    ax.set_xlabel(
                     fr'$\displaystyle {boldtext("Time")}$',
                     fontsize=plot_cfg["label_fontsize"]
                     )
    legend_elements = [
    Line2D([0],[0],color='r',linestyle='-',lw=2,label=fr"$\displaystyle \langle {boldtext('KE')} \rangle$"),
    Line2D([0],[0],color='b',linestyle='-',lw=2,label=fr"$\displaystyle \langle {boldtext('PE')} \rangle$"),
    Line2D([0],[0],color='g',linestyle='-',lw=2,label=fr"$\displaystyle \langle E_{{{boldtext('vib')}}} \rangle_{{{boldtext('qu. HEOM')}}}$")
    # Line2D([0],[0],color='g',linestyle='--',lw=2,label=fr"$\displaystyle \langle E_{{{boldtext('vib')}}} \rangle_{{{boldtext('Mark. Friction')}}}$")
    # Line2D([0],[0],color='g',linestyle='-',lw=2,label=r"$\displaystyle \frac{k_{B}T}{2}$")
    ]
    ax.legend(handles=legend_elements,loc='lower right',fontsize=18)
    set_default_axis_style(cfg,ax)
    set_default_axis_style(cfg,ax)
    ax.set_xlim(0,np.max(time_vec))
    plt.tight_layout()
    plt.savefig(results_layout.transient_plots / f"energy_transient.pdf")
    # plt.show()

def generate_current_single_plot(results_to_plot,cfg,results_layout):

    set_default_plot_style(cfg)
    vib_freq = cfg["vib_freq"]
    time_vec = results_to_plot["time_vec"]*vib_freq
    current_ad = results_to_plot["adiabatic_current"]
    plot_cfg = cfg["plotting"]
    fig, ax = plt.subplots()
    ax.plot(time_vec,current_ad,linewidth=plot_cfg["linewidth"],color='b')
    ax.set_ylabel(
                fr'$\displaystyle  \langle I_{{L}} \rangle $',
                fontsize=plot_cfg["label_fontsize"]
                )
    ax.set_xlabel(
                fr'$\displaystyle {boldtext("Time")}$',
                fontsize=plot_cfg["label_fontsize"]
                )
    set_default_axis_style(cfg,ax)
    ax.set_xlim(0,np.max(time_vec))
    plt.tight_layout()
    plt.savefig(results_layout.transient_plots / f"current_ad_transient.pdf")
    # plt.show()

def generate_corrfunc_single_plot(results_to_plot,cfg,results_layout):

    set_default_plot_style(cfg)
    vib_freq = cfg["vib_freq"]
    # n_terms = cfg["decomposition"]["n_terms"]
    time_vec = results_to_plot["time_vec"]#*vib_freq
    corrfunc_traj = results_to_plot["stoch_force"]
    corrfunc_exact = results_to_plot["corrfunc_exact"]
    # time_vec_exact = np.genfromtxt(cfg["raw_data_root"] / "corrfunc_integrand_heom.dat")[:,0]
    # corrfunc_exact = np.genfromtxt(cfg["raw_data_root"] / "corrfunc_integrand_heom.dat")[:,1]
    # processed_cifs_dir = cfg["results_cifs_root"] 
    # stoch_force_arr = load_force_array(processed_cifs_dir,f"unprocessed_weights_frequencies_corrfunc.npz")
    # corrfunc_poles_exact = np.zeros(len(time_vec_exact),dtype=complex)
    # for itr_pole in np.arange(len(stoch_force_arr["freq_real"])):
    #     weight = stoch_force_arr["weights_real"][itr_pole] + 1j*stoch_force_arr["weights_imag"][itr_pole]
    #     freq = stoch_force_arr["freq_real"][itr_pole] + 1j*stoch_force_arr["freq_imag"][itr_pole]
        # corrfunc_poles_exact += np.abs(weight) * np.exp(-freq.real * time_vec_exact) * \
        #                         np.cos(freq.imag * time_vec_exact + np.angle(weight))
        # corrfunc_poles_exact += weight * np.exp(-freq * time_vec_exact)
                                
    plot_cfg = cfg["plotting"]
    fig, ax = plt.subplots()
    ax.plot(time_vec*vib_freq,corrfunc_traj,linewidth=plot_cfg["linewidth"],color='b',
            label=r"$\displaystyle \langle f(t) f(0) \rangle$")
    ax.plot(time_vec*vib_freq,corrfunc_exact,linewidth=plot_cfg["linewidth"],color='r',
            label=r"$\displaystyle D(t)$")
    # ax.plot(time_vec_exact*vib_freq,corrfunc_poles_exact,linewidth=plot_cfg["linewidth"],color='r',
    #         linestyle='--')
    ax.set_ylabel(
                     fr'$\displaystyle  {boldtext("Corr. Function")}$',
                     fontsize=plot_cfg["label_fontsize"]
                     )
    ax.set_xlabel(
                     fr'$\displaystyle {boldtext("Time")}$',
                     fontsize=plot_cfg["label_fontsize"]
                     )
    set_default_axis_style(cfg,ax)
    ax.set_xlim(0,10)
    # ax.set_ylim(0,1.05*np.max(corrfunc_exact))
    
    handles,labels = ax.get_legend_handles_labels()
    ax.legend(handles,labels,loc="upper right",fontsize=18)
    plt.tight_layout()
    plt.savefig(results_layout.transient_plots / f"corrfunc_transient.pdf")
    plt.show()

def generate_plots_single_voltage(cfg,voltage):

    layout = results_layout.ResultsLayout(cfg,voltage)
    ensemble_av_root = layout.transient_ensemble_av
    data_storage_cfg = cfg["simulation"]["data_storage"]
    store_state = data_storage_cfg["store_state"]
    store_energy = data_storage_cfg["store_energy"]
    store_current = data_storage_cfg["store_current"]
    store_corrfunc = data_storage_cfg["store_stoch_force"]
    markovian_value = cfg["simulation"]["markovian"]
    results_to_plot = np.load(ensemble_av_root / "ensemble_av.npz")
    if store_state:
        generate_state_single_plot(results_to_plot,cfg,layout)
    if store_energy:
        generate_energy_single_plot(results_to_plot,cfg,layout)
    if store_current:
        generate_current_single_plot(results_to_plot,cfg,layout)
    if not markovian_value:
        if store_corrfunc:
            generate_corrfunc_single_plot(results_to_plot,cfg,layout)
    logger.info(f"Saved plots to {layout.transient_plots}")

if __name__ == "__main__":

    import numpy as np
    from pathlib import Path
    from source.utils.enums import FunctionType

    cfg = load_config()
    markovian_value = cfg["simulation"]["markovian"]
    voltage = cfg["voltage"]["min"]
    results_dir = Path(cfg["results_transient_root"] / cfg["system_identifier_dir"] / f"voltage_{voltage:.2f}eV")
    # results_dir = Path("results/transient/2l1m_negative_friction_w_time_dependence/gamma_100meV_temp_300K/delta_0.2eV/tier_3/voltage_0.40eV/")
    save_dir = np.copy(results_dir)
    if markovian_value:
        results_dir = results_dir / "ensemble_av_markovian.npz"
    else:
        results_dir = results_dir / "ensemble_av.npz"
    
    results_to_plot = np.load(results_dir)

    generate_state_single_plot(results_to_plot,cfg,save_dir)
    generate_energy_single_plot(results_to_plot,cfg,save_dir)
    generate_corrfunc_single_plot(results_to_plot,cfg,save_dir)
    generate_current_single_plot(results_to_plot,cfg,save_dir)

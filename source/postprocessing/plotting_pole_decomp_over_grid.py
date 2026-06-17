# scripts/plotting_el_forces_pole_decomposition.py

from source.utils import file_walker
from source.utils.plotting import (
    set_default_plot_style,
    set_default_axis_style,
    boldtext
)
from source.utils.config import load_config
from source.utils.enums import (
    FunctionType,
    PoleType
)
from source.utils import file_walker
from matplotlib import pyplot as plt
import numpy as np
import logging
import os
import matplotlib as mpl
from scipy.interpolate import CubicSpline
mpl.rcParams['savefig.directory'] = os.getcwd()

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # default logging level
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def adiabatic_force_single_plot(results_to_plot,cfg,save_dir):

    set_default_plot_style(cfg)
    x_coordinate_vec = results_to_plot["x"]
    plot_cfg = cfg["plotting"]
    fig, ax = plt.subplots()
    adiabatic_force = results_to_plot["Adiabatic_Force"]
    ax.plot(x_coordinate_vec,adiabatic_force,linewidth=plot_cfg["linewidth"])
    ax.set_ylabel(
                    fr'$\displaystyle  F^{{{boldtext("ad")}}}_{{{boldtext("el")}}}(x) $',
                    fontsize=plot_cfg["label_fontsize"]
                )
    ax.set_xlabel(
                    fr'$\displaystyle  {boldtext("Coordinate")} $',
                    fontsize=plot_cfg["label_fontsize"]
                )
    set_default_axis_style(cfg,ax)
    ax.set_xlim(np.min(x_coordinate_vec),np.max(x_coordinate_vec))
    plt.tight_layout()
    plt.savefig(f"{save_dir}/unprocessed_adiabatic_force.pdf")
    # plt.show()


def current_ad_single_plot(results_to_plot,cfg,save_dir):

    set_default_plot_style(cfg)
    x_coordinate_vec = results_to_plot["x"]
    plot_cfg = cfg["plotting"]
    fig, ax = plt.subplots()
    adiabatic_force = results_to_plot["Current_Ad"]
    ax.plot(x_coordinate_vec,adiabatic_force,linewidth=plot_cfg["linewidth"])
    ax.set_ylabel(
                    fr'$\displaystyle  \langle I_{{L}} \rangle^{{{boldtext("ad")}}}(x) $',
                    fontsize=plot_cfg["label_fontsize"]
                )
    ax.set_xlabel(
                    fr'$\displaystyle  {boldtext("Coordinate")} $',
                    fontsize=plot_cfg["label_fontsize"]
                )
    set_default_axis_style(cfg,ax)
    ax.set_xlim(np.min(x_coordinate_vec),np.max(x_coordinate_vec))
    plt.tight_layout()
    plt.savefig(f"{save_dir}/adiabatic_current.pdf")
    #plt.show()


def markovian_fdt_single_plot(results_to_plot,cfg,save_dir):

    set_default_plot_style(cfg)
    x_coordinate_vec = results_to_plot["x"]
    plot_cfg = cfg["plotting"]
    temp = cfg["temperature"] * 8.617333262e-5
    fig, ax = plt.subplots()
    markovian_friction = results_to_plot["Friction"]
    markovian_corrfunc = results_to_plot["Corrfunc"]
    ax.plot(x_coordinate_vec,markovian_friction,linewidth=plot_cfg["linewidth"])
    ax.plot(x_coordinate_vec,markovian_corrfunc / temp,linewidth=plot_cfg["linewidth"])
    ax.set_ylabel(
                    fr'$\displaystyle  {boldtext("FDT")} $',
                    fontsize=plot_cfg["label_fontsize"]
                )
    ax.set_xlabel(
                    fr'$\displaystyle  {boldtext("Coordinate")} $',
                    fontsize=plot_cfg["label_fontsize"]
                )
    set_default_axis_style(cfg,ax)
    ax.set_xlim(np.min(x_coordinate_vec),np.max(x_coordinate_vec))
    plt.tight_layout()
    plt.savefig(f"{save_dir}/markovian_fdt.pdf")
    #plt.show()

def generate_el_forces_decomposition_single_plot(results_to_plot,ft,pt,cfg,save_dir):

    set_default_plot_style(cfg)
    x_coordinate_vec = results_to_plot["x"]
    plot_cfg = cfg["plotting"]
    fig, ax = plt.subplots(2,1,sharex=True,figsize=(9,12),gridspec_kw={'hspace': 0.0},dpi=80)
    real_function = results_to_plot[f"{pt.tag}_real"]
    n_terms = real_function.shape[1]
    spline_mask = np.abs(x_coordinate_vec) > 0.1
    real_function_splined = [CubicSpline(x_coordinate_vec[spline_mask],real_function[spline_mask,itrp]) for itrp in range(n_terms)]
    real_function_splined = np.array([real_function_splined[itrp](x_coordinate_vec) for itrp in range(n_terms)]).T
    print(real_function.shape)
    print(real_function_splined.shape)
    imag_function = results_to_plot[f"{pt.tag}_imag"]
    if ft is FunctionType.FRICTION:
        text = fr"$\displaystyle {boldtext("Friction")}$"
    elif ft is FunctionType.CORRFUNC:
        text = fr"$\displaystyle {boldtext("Correlation Function")}$"
    ax[0].plot(x_coordinate_vec,real_function,linewidth=plot_cfg["linewidth"])
    # ax[0].plot(x_coordinate_vec,real_function_splined,linewidth=plot_cfg["linewidth"],linestyle='--')
    ax[0].text(0.1,0.9,text,transform=ax[0].transAxes,ha="left", va="top",fontsize=24)
    ax[0].set_ylabel(
                     fr'$\displaystyle  {boldtext("Re")}({boldtext(pt.tag)}) $',
                     fontsize=plot_cfg["label_fontsize"]
                     )
    ax[1].plot(x_coordinate_vec,imag_function,linewidth=plot_cfg["linewidth"])
    ax[1].set_ylabel(
                     fr'$\displaystyle {boldtext("Im")}({boldtext(pt.tag)})$',
                     fontsize=plot_cfg["label_fontsize"]
                     )
    ax[1].set_xlabel(
                     fr'$\displaystyle {boldtext("Coordinate: ")} x$',
                     fontsize=plot_cfg["label_fontsize"]
                     )
    set_default_axis_style(cfg,ax)
    ax[1].set_xlim(np.min(x_coordinate_vec),np.max(x_coordinate_vec))
    plt.tight_layout()
    plt.savefig(f"{save_dir}/unprocessed_{ft.tag}_{pt.tag}.pdf")
    # print(real_function)
    # print(imag_function)
    plt.show()

def generate_el_forces_decomposition_both_plots(results_to_plot,cfg,save_dir,ft):

    for pt in PoleType:
        generate_el_forces_decomposition_single_plot(results_to_plot,ft,pt,cfg,save_dir)

def generate_all_plots_single_voltage(cfg,voltage_dir):

    # results_dir_markovian_forces = voltage_dir / cfg["el_forces_root"] / "unprocessed_markovian_forces.npz"
    # results_to_plot_markovian_forces = np.load(results_dir_markovian_forces)
    # adiabatic_force_single_plot(
    #     results_to_plot_markovian_forces,
    #     cfg,
    #     voltage_dir
    # )
    # markovian_fdt_single_plot(
    #     results_to_plot_markovian_forces,
    #     cfg,
    #     voltage_dir
    # )
    # results_dir_el_obs = voltage_dir / cfg["el_forces_root"] / "unprocessed_el_observables.npz"
    # results_to_plot_el_obs = np.load(results_dir_el_obs)
    # current_ad_single_plot(
    #     results_to_plot_el_obs,
    #     cfg,
    #     voltage_dir
    # )
    for ft in FunctionType:
        results_dir_poles = voltage_dir / cfg["el_forces_root"] / f"unprocessed_weights_frequencies_{ft.tag}.npz"
        results_to_plot_poles = np.load(results_dir_poles)
        generate_el_forces_decomposition_both_plots(
            results_to_plot_poles,
            cfg,
            voltage_dir,
            ft
        )
    logger.info(f"Saved plots to {voltage_dir}")


def generate_all_plots(cfg):

    results_el_forces_root = cfg["results_root"] / cfg["system_identifier_dir"] 
    for voltage_dir in file_walker.iter_voltage_dirs(results_el_forces_root,cfg):
        # results_dir_markovian_forces = voltage_dir / cfg["el_forces_root"] / "unprocessed_markovian_forces.npz"
        # results_to_plot_markovian_forces = np.load(results_dir_markovian_forces)
        # adiabatic_force_single_plot(
        #      results_to_plot_markovian_forces,
        #      cfg,
        #      voltage_dir
        # )
        # markovian_fdt_single_plot(
        #      results_to_plot_markovian_forces,
        #      cfg,
        #      voltage_dir
        # )
        # results_dir_el_obs = voltage_dir / cfg["el_forces_root"] / "unprocessed_el_observables.npz"
        # results_to_plot_el_obs = np.load(results_dir_el_obs)
        # current_ad_single_plot(
        #      results_to_plot_el_obs,
        #      cfg,
        #      voltage_dir
        # )
        for ft in FunctionType:
            results_dir_poles = voltage_dir / cfg["el_forces_root"] / f"unprocessed_weights_frequencies_{ft.tag}.npz"
            results_to_plot_poles = np.load(results_dir_poles)
            generate_el_forces_decomposition_both_plots(
                results_to_plot_poles,
                cfg,
                voltage_dir,
                ft
            )
        logger.info(f"Saved plots to {voltage_dir.relative_to(results_el_forces_root)}")

if __name__ == "__main__":

    import numpy as np
    from pathlib import Path
    from source.utils.enums import FunctionType

    cfg = load_config()
    results_dir = Path("results/el_forces/2l1m_negative_friction_w_time_dependence/gamma_100meV_temp_300K/delta_0.2eV/tier_2/voltage_0.40eV/")
    save_dir = results_dir
    results_dir_ad_force = results_dir / "unprocessed_adiabatic_force.npz"
    
    results_dir_friction = results_dir / "unprocessed_weights_frequencies_friction.npz"


    results_to_plot_friction = np.load(results_dir_friction)
    generate_el_forces_decomposition_both_plots(results_to_plot_friction,cfg,save_dir,FunctionType.FRICTION)
    
    results_to_plot_ad_force = np.load(results_dir_ad_force)
    adiabatic_force_single_plot(results_to_plot_ad_force,cfg,save_dir)
    





    

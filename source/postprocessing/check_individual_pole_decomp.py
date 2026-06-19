import numpy as np
from pathlib import Path
from source.utils.enums import FunctionType
from source.utils.plotting import (
    set_default_plot_style,
    set_default_axis_style,
    boldtext
)
from matplotlib import pyplot as plt
from source.utils import file_walker
from matplotlib.lines import Line2D
import os
import matplotlib as mpl
mpl.rcParams['savefig.directory'] = os.getcwd()

def w_coth_w_over_T(w, T, tol=1e-6):
    
    w = np.asarray(w, dtype=float)
    out = np.empty_like(w)

    x = w / (2*T)
    small = np.abs(x) < tol
    large = ~small

    # Series expansion near zero
    x2 = x[small]**2
    out[small] = 2 * T * (1.0 + x2/3.0 - x2*x2/45.0)

    # Exact expression away from zero
    out[large] = w[large] / np.tanh(x[large])

    return out

def generate_freq_vec(cfg):

    decomposition_cfg = cfg["decomposition"]
    max_freq = decomposition_cfg["cutoff_frequency"] * decomposition_cfg["cutoff_factor"]
    n_freq = decomposition_cfg["Nsupport_points"]
    freq_vec = np.linspace(-max_freq,max_freq,n_freq)

    return freq_vec

def generate_estimate(poles,time_vec,freq_vec,function_type,itrx):

    freq_real = poles["freq_real"][itrx]
    freq_imag = poles["freq_imag"][itrx]
    freq = (freq_real + 1j*freq_imag)[:, None]
    weight_real = poles["weight_real"][itrx]
    weight_imag = poles["weight_imag"][itrx]
    weight = (weight_real + 1j*weight_imag)[:, None]

    time_arr = time_vec[None, :]
    freq_arr = freq_vec[None, :]
    if function_type is FunctionType.FRICTION:
        estimate_time = np.real(np.sum(weight * np.exp(-freq * time_arr),axis=0))
        estimate_freq = np.sum((weight /(freq - 1j*freq_arr)),axis=0)
    elif function_type is FunctionType.CORRFUNC:
        estimate_time = np.real(np.sum(weight * np.exp(-freq * time_arr),axis=0))
        estimate_freq = np.sum((weight /(freq - 1j*freq_arr)) + 
                               (weight /(freq + 1j*freq_arr)),axis=0)
    
    return estimate_time,estimate_freq,{
        "freq_real": freq_real,
        "freq_imag": freq_imag,
        "weight_real": weight_real,
        "weight_imag": weight_imag
    }


def prepare_data(cfg,voltage,itrx):

    raw_results_root = cfg["results_root"] / cfg["system_identifier_dir"]
    voltage_dir = raw_results_root / f"voltage_{voltage:.2f}eV" / cfg["el_forces_root"]
    corrfunc_poles = np.load(voltage_dir / "unprocessed_weights_frequencies_corrfunc.npz")
    friction_poles = np.load(voltage_dir / "unprocessed_weights_frequencies_friction.npz")
    print(friction_poles["freq_real"][itrx].shape)
    
    x_value = corrfunc_poles["x"][itrx]
    print(x_value)
    xdir = file_walker.x_dir_given_x_coordinate(x_value)
    exact_data_path = cfg["raw_data_root"] / Path(f"voltage_{voltage:.2f}eV") / xdir

    friction_exact = np.genfromtxt(exact_data_path / "friction_integrand_heom.dat")
    corrfunc_exact = np.genfromtxt(exact_data_path / "corrfunc_integrand_heom.dat")
    freq_vec = generate_freq_vec(cfg)
    friction_estimate_time,friction_estimate_freq,friction_poles_x = generate_estimate(
        friction_poles,friction_exact[:,0],freq_vec,FunctionType.FRICTION,itrx)
    corrfunc_estimate_time,corrfunc_estimate_freq,corrfunc_poles_x = generate_estimate(
        corrfunc_poles,corrfunc_exact[:,0],freq_vec,FunctionType.CORRFUNC,itrx)

    return {
        "freq_vec": freq_vec,
        "friction_exact": friction_exact,
        "friction_estimate_time": friction_estimate_time,
        "friction_estimate_freq": friction_estimate_freq,
        "corrfunc_exact": corrfunc_exact,
        "corrfunc_estimate_time": corrfunc_estimate_time,
        "corrfunc_estimate_freq": corrfunc_estimate_freq,
        "friction_poles": friction_poles_x,
        "corrfunc_poles": corrfunc_poles_x
    }

def plot_poles(cfg,voltage,itrx):

    temp = cfg["temperature"] * 8.617333262e-5

    data = prepare_data(cfg,voltage,itrx)
    
    set_default_plot_style(cfg)
    plot_cfg = cfg["plotting"]
    fig, ax = plt.subplots(2,2,figsize=(12,12))
    ax[0,0].plot(data["freq_vec"],data["corrfunc_estimate_freq"],'b')
    ax[0,0].plot(data["freq_vec"],data["friction_estimate_freq"] * w_coth_w_over_T(data["freq_vec"],temp),'r')
    ax[0,0].plot(data["freq_vec"],data["friction_estimate_freq"] * 2 * temp,'r--')
    ax[0,0].set_ylabel(
                    fr'$\displaystyle  {boldtext("FT(Forces)")} $',
                    fontsize=plot_cfg["label_fontsize"]
                )
    ax[0,0].set_xlabel(
                    fr'$\displaystyle  {boldtext("Frequency: ")} \omega $',
                    fontsize=plot_cfg["label_fontsize"]
                )
    ax[0,1].set_ylabel(
                    fr'$\displaystyle  {boldtext("Forces")} $',
                    fontsize=plot_cfg["label_fontsize"]
                )
    ax[0,1].set_xlabel(
                    fr'$\displaystyle  {boldtext("Time: ")} t $',
                    fontsize=plot_cfg["label_fontsize"]
                )
    time_labels = [Line2D([0],[0],color='r',linestyle='-',
                    lw=2,label=fr"$\displaystyle \gamma(t)$"),
                    Line2D([0],[0],color='b',linestyle='-',
                    lw=2,label=fr"$\displaystyle D(t)$")]
    generation_labels = [Line2D([0],[0],color='k',linestyle='-',
                    lw=2,label=fr"$\displaystyle {boldtext('Exact')}$"),
                    Line2D([0],[0],color='k',linestyle='--',
                    lw=2,label=fr"$\displaystyle {boldtext('ESPRIT')}$")]
    frequency_labels = [Line2D([0],[0],color='b',linestyle='-',
                    lw=2,label=r"$\displaystyle \tilde{D}(\omega)$"),
                    Line2D([0],[0],color='r',linestyle='-',
                    lw=2,label=r"$\displaystyle \omega \coth\left(\frac{\omega}{2k_{B}T}\right) \mbox{\textbf{Re}}\left(\tilde{\gamma}(\omega)\right)$"),
                    Line2D([0],[0],color='r',linestyle='--',
                    lw=2,label=r"$\displaystyle 2 k_{B}T \mbox{\textbf{Re}}(\tilde{\gamma}(\omega))$")]
    ax[0,0].legend(handles=frequency_labels,loc='upper right',fontsize=12)
    time_legend = ax[0,1].legend(handles=time_labels,loc='center right',fontsize=18)
    ax[0,1].legend(handles=generation_labels,loc='upper right',fontsize=18)
    ax[0,1].add_artist(time_legend)
    set_default_axis_style(cfg,ax)
    set_default_axis_style(cfg,ax)
    ax[0,0].set_xlim(-2,2)
    # ax[0,1].set_xlim(0,200)
    ax[0,1].plot(data["corrfunc_exact"][:,0],data["corrfunc_exact"][:,1],'b')
    ax[0,1].plot(data["corrfunc_exact"][:,0],data["corrfunc_estimate_time"],'b--')
    ax[0,1].plot(data["friction_exact"][:,0],-data["friction_exact"][:,1],'r')
    ax[0,1].plot(data["friction_exact"][:,0],data["friction_estimate_time"],'r--')

    print(len(data["friction_poles"]["freq_real"]))
    ax[1,0].scatter(data["corrfunc_poles"]["freq_real"],data["corrfunc_poles"]["freq_imag"]
                    ,color="b")
    ax[1,0].scatter(data["friction_poles"]["freq_real"],data["friction_poles"]["freq_imag"]
                    ,color="r")
    ax[1,1].scatter(data["corrfunc_poles"]["weight_real"],data["corrfunc_poles"]["weight_imag"]
                    ,color="b")
    ax[1,1].scatter(data["friction_poles"]["weight_real"],data["friction_poles"]["weight_imag"]
                    ,color="r")
    ax[1,0].set_xlabel(
                    fr'$\displaystyle  {boldtext("Re(freq)")} $',
                    fontsize=plot_cfg["label_fontsize"]
                )
    ax[1,0].set_ylabel(
                    fr'$\displaystyle  {boldtext("Im(freq)")} $',
                    fontsize=plot_cfg["label_fontsize"]
                )
    ax[1,1].set_xlabel(
                    fr'$\displaystyle  {boldtext("Re(weight)")} $',
                    fontsize=plot_cfg["label_fontsize"]
                )
    ax[1,1].set_ylabel(
                    fr'$\displaystyle  {boldtext("Im(weight)")} $',
                    fontsize=plot_cfg["label_fontsize"]
                )

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    from source.utils.config import load_config
    import argparse
    cfg = load_config()
    parser = argparse.ArgumentParser(
        description="Run preprocessing for all voltages or a single voltage."
    )
    parser.add_argument(
        "--voltage",
        type=float,
        default=None,
        help="If provided, preprocess only this voltage"
    )
    parser.add_argument(
        "--itrx",
        type=float,
        default=None,
        help="If provided, preprocess only this voltage"
    )

    args = parser.parse_args()
    voltage = float(args.voltage)
    itrx = int(args.itrx)
    # plot_power_spectrum(cfg,0,1)
    plot_poles(cfg,voltage,itrx)
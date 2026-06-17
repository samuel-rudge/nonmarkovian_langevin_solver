# source/utils/plotting.py

import numpy as np

def set_default_plot_style(cfg):

    from matplotlib import pyplot as plt

    plot_cfg = cfg["plotting"]
    if plot_cfg.get('use_tex', False):
        try:
            plt.rc('text', usetex=True)
            plt.rc('text.latex', preamble=r'\boldmath')
        except Exception as e:
            print(f"Warning: LaTeX not available, using matplotlib mathtext. ({e})")

    plt.rc('font', family=plot_cfg['font_family'])
    plt.rc('axes', linewidth=plot_cfg['axes_linewidth'])

def iter_axes(ax):

    if isinstance(ax, np.ndarray):
        return ax.flat
    else:
        return(ax,)

def set_default_axis_style(cfg,ax):

    from matplotlib import pyplot as plt

    plot_cfg = cfg["plotting"]
    for each_ax in iter_axes(ax):
        each_ax.tick_params(axis='x', 
                        labelcolor='black',
                        length=plot_cfg["tick_length"],
                        width=plot_cfg["tick_width"],
                        labelsize=plot_cfg["tick_labelsize"]
                    )
        each_ax.tick_params(axis='y', 
                        labelcolor='black',
                        length=plot_cfg["tick_length"],
                        width=plot_cfg["tick_width"],
                        labelsize=plot_cfg["tick_labelsize"]
                    )
    
def boldtext(str):

    output_str = fr'\mbox{{\textbf{{{str}}}}}'

    return output_str

